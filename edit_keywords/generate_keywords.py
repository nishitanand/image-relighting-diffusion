"""
Step 3: Generate Lighting Keywords/Captions using Vision-Language Model

This script takes the CSV output from relightingDataGen-parallel (Step 2) and generates
lighting description keywords for each original image using a VLM.

Supported VLM providers:
- qwen3vl (DEFAULT): Qwen3-VL-30B via vLLM (fast, high quality, free)
- qwen3vl-server: Qwen3-VL via vLLM server API (for distributed setup)
- mistral: Mistral Pixtral via API
- openai: OpenAI GPT-4o via API

Input CSV columns:
    - image_path: Path to original input image
    - lighting_score: CLIP lighting quality score
    - output_image_path: Path to albedo degraded output image

Output CSV columns (adds 1 new column):
    - image_path: Path to original input image
    - lighting_score: CLIP lighting quality score  
    - output_image_path: Path to albedo degraded output image
    - lighting_keywords: VLM-generated lighting/environment description

Usage:
    # Default: Use Qwen3-VL-30B with vLLM (recommended)
    python generate_keywords.py --csv path/to/csv --output_dir ./output
    
    # Use vLLM server (start server first, then run this)
    python generate_keywords.py --csv path/to/csv --provider qwen3vl-server --vllm_url http://localhost:8000/v1
    
    # Use Mistral API
    python generate_keywords.py --csv path/to/csv --provider mistral
    
    # Use OpenAI API
    python generate_keywords.py --csv path/to/csv --provider openai

References:
    - Qwen3-VL: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct
    - vLLM Qwen3-VL: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html
"""

import os
import sys
from pathlib import Path
import argparse
import pandas as pd
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

LIGHTING_KEYWORD_PROMPT = """Analyze this image and describe the lighting and environment in a SHORT phrase (5-15 words max).

Focus ONLY on:
1. Lighting type (sunlight, studio light, neon, golden hour, etc.)
2. Lighting direction (from left, from above, backlit, etc.)  
3. Lighting quality (soft, harsh, dramatic, diffused, etc.)
4. Environment/setting if relevant (indoor, outdoor, beach, forest, city, etc.)

Output format: A SHORT comma-separated phrase describing the lighting.

Examples of good outputs:
- "sunlight through the blinds, near window blinds"
- "sunlight from the left side, beach"
- "magic golden lit, forest"
- "neo punk, city night"
- "soft studio lighting, neutral background"
- "dramatic side lighting, dark moody"
- "warm sunset glow, outdoor"
- "cold blue lighting, futuristic"
- "natural daylight, overcast sky"
- "neon pink and blue, cyberpunk city"

IMPORTANT: Output ONLY the short descriptive phrase, nothing else. No explanations, no "The lighting is...", just the keywords."""


# ============================================================================
# VLM PROVIDERS
# ============================================================================

class Qwen3VLM:
    """
    Qwen3-VL-30B using vLLM for fast local inference.
    
    This is the recommended default provider - free, fast, and high quality.
    Requires: pip install vllm qwen-vl-utils transformers
    
    Reference: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct
    """
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        tensor_parallel_size: int = None,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 32768
    ):
        """
        Initialize Qwen3-VL with vLLM.
        
        Args:
            model_name: HuggingFace model name
            tensor_parallel_size: Number of GPUs for tensor parallelism (default: all available)
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length
        """
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.llm = None
        self.processor = None
        self.sampling_params = None
    
    def load_model(self):
        """Load the model with vLLM."""
        if self.llm is not None:
            return
        
        logger.info(f"Loading {self.model_name} with vLLM...")
        
        try:
            import torch
            from vllm import LLM, SamplingParams
            from transformers import AutoProcessor
            
            # Set multiprocessing method for vLLM
            os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
            
            # Determine tensor parallel size
            tp_size = self.tensor_parallel_size or torch.cuda.device_count()
            
            logger.info(f"Using {tp_size} GPUs with tensor parallelism")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Load model with vLLM
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                mm_encoder_tp_mode="data",  # Better performance for vision encoder
                trust_remote_code=True,
            )
            
            # Setup sampling parameters
            self.sampling_params = SamplingParams(
                temperature=0.1,  # Low temperature for consistent outputs
                max_tokens=100,
                top_k=-1,
                stop_token_ids=[],
            )
            
            logger.info(f"✅ Qwen3-VL loaded successfully with vLLM")
            
        except ImportError as e:
            raise ImportError(
                f"Missing dependencies: {e}\n"
                "Install with: pip install vllm qwen-vl-utils transformers"
            )
    
    def _prepare_input(self, image_path: str, prompt: str):
        """Prepare input for vLLM inference."""
        from qwen_vl_utils import process_vision_info
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process vision info
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
        )
        
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        
        return {
            'prompt': text,
            'multi_modal_data': mm_data,
            'mm_processor_kwargs': video_kwargs if video_kwargs else {}
        }
    
    def generate_keywords(self, image_path: str, prompt: str = None) -> str:
        """
        Generate lighting keywords for an image.
        
        Args:
            image_path: Path to the image
            prompt: Custom prompt (default: LIGHTING_KEYWORD_PROMPT)
            
        Returns:
            Lighting keywords string
        """
        self.load_model()
        prompt = prompt or LIGHTING_KEYWORD_PROMPT
        
        # Prepare input
        input_data = self._prepare_input(image_path, prompt)
        
        # Generate
        outputs = self.llm.generate([input_data], self.sampling_params)
        
        return outputs[0].outputs[0].text.strip()
    
    def generate_keywords_batch(self, image_paths: list, prompt: str = None) -> list:
        """
        Generate keywords for multiple images in a batch (more efficient).
        
        Args:
            image_paths: List of image paths
            prompt: Custom prompt
            
        Returns:
            List of keyword strings
        """
        self.load_model()
        prompt = prompt or LIGHTING_KEYWORD_PROMPT
        
        # Prepare all inputs
        inputs = [self._prepare_input(path, prompt) for path in image_paths]
        
        # Generate in batch
        outputs = self.llm.generate(inputs, self.sampling_params)
        
        return [output.outputs[0].text.strip() for output in outputs]


class Qwen3VLMServer:
    """
    Qwen3-VL via vLLM server API (OpenAI-compatible).
    
    Use this when you have a vLLM server running separately.
    Start server with:
        vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct --tensor-parallel-size 4 --port 8000
    
    Reference: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        api_key: str = "EMPTY"
    ):
        """
        Initialize Qwen3-VL server client.
        
        Args:
            base_url: vLLM server URL
            model_name: Model name as registered in vLLM server
            api_key: API key (use "EMPTY" for local vLLM)
        """
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.client = None
    
    def _get_client(self):
        """Get or create OpenAI client."""
        if self.client is None:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=3600
                )
            except ImportError:
                raise ImportError("Install openai: pip install openai")
        return self.client
    
    def encode_image_base64(self, image_path: str) -> str:
        """Encode image to base64 data URL."""
        with open(image_path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        
        ext = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mime_type = mime_types.get(ext, 'image/jpeg')
        
        return f"data:{mime_type};base64,{data}"
    
    def generate_keywords(self, image_path: str, prompt: str = None) -> str:
        """Generate lighting keywords for an image."""
        prompt = prompt or LIGHTING_KEYWORD_PROMPT
        client = self._get_client()
        
        # Encode image
        image_url = self.encode_image_base64(image_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=100,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()


class MistralVLM:
    """Mistral Vision-Language Model via API."""
    
    def __init__(self, api_key: str = None, model: str = "pixtral-large-latest"):
        """
        Initialize Mistral VLM.
        
        Args:
            api_key: Mistral API key (or set MISTRAL_API_KEY env var)
            model: Model name (pixtral-large-latest, pixtral-12b-latest, etc.)
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        
        self.model = model
        
        try:
            from mistralai import Mistral
            self.client = Mistral(api_key=self.api_key)
        except ImportError:
            raise ImportError("Install mistralai: pip install mistralai")
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
    
    def get_image_mime_type(self, image_path: str) -> str:
        """Get MIME type from image path."""
        ext = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return mime_types.get(ext, 'image/jpeg')
    
    def generate_keywords(self, image_path: str, prompt: str = None) -> str:
        """
        Generate lighting keywords for an image.
        
        Args:
            image_path: Path to the image
            prompt: Custom prompt (default: LIGHTING_KEYWORD_PROMPT)
            
        Returns:
            Lighting keywords string
        """
        prompt = prompt or LIGHTING_KEYWORD_PROMPT
        
        # Encode image
        base64_image = self.encode_image(image_path)
        mime_type = self.get_image_mime_type(image_path)
        
        # Create message
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:{mime_type};base64,{base64_image}"
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        # Call API
        response = self.client.chat.complete(
            model=self.model,
            messages=messages
        )
        
        return response.choices[0].message.content.strip()


class OpenAIVLM:
    """OpenAI GPT-4 Vision via API."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        """
        Initialize OpenAI VLM.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model name (gpt-4o, gpt-4o-mini, etc.)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.model = model
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Install openai: pip install openai")
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
    
    def get_image_mime_type(self, image_path: str) -> str:
        """Get MIME type from image path."""
        ext = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return mime_types.get(ext, 'image/jpeg')
    
    def generate_keywords(self, image_path: str, prompt: str = None) -> str:
        """Generate lighting keywords for an image."""
        prompt = prompt or LIGHTING_KEYWORD_PROMPT
        
        base64_image = self.encode_image(image_path)
        mime_type = self.get_image_mime_type(image_path)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()


# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def get_vlm(provider: str, model: str = None, **kwargs):
    """
    Get VLM instance based on provider.
    
    Args:
        provider: Provider name:
            - 'qwen3vl' (DEFAULT): Local Qwen3-VL with vLLM
            - 'qwen3vl-server': Qwen3-VL via vLLM server
            - 'mistral': Mistral API
            - 'openai': OpenAI API
        model: Model name (optional, uses default for provider)
        **kwargs: Additional arguments for the VLM
        
    Returns:
        VLM instance
    """
    if provider == "qwen3vl":
        model = model or "Qwen/Qwen3-VL-30B-A3B-Instruct"
        return Qwen3VLM(model_name=model, **kwargs)
    elif provider == "qwen3vl-server":
        model = model or "Qwen/Qwen3-VL-30B-A3B-Instruct"
        return Qwen3VLMServer(model_name=model, **kwargs)
    elif provider == "mistral":
        model = model or "pixtral-large-latest"
        return MistralVLM(model=model, **kwargs)
    elif provider == "openai":
        model = model or "gpt-4o"
        return OpenAIVLM(model=model, **kwargs)
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            "Use 'qwen3vl' (default), 'qwen3vl-server', 'mistral', or 'openai'"
        )


def process_single_image(vlm, image_path: str, retry_count: int = 3) -> str:
    """
    Process a single image with retries.
    
    Args:
        vlm: VLM instance
        image_path: Path to image
        retry_count: Number of retries on failure
        
    Returns:
        Lighting keywords string or error message
    """
    for attempt in range(retry_count):
        try:
            keywords = vlm.generate_keywords(image_path)
            return keywords
        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep(1)  # Wait before retry
                continue
            return f"ERROR: {str(e)}"


def process_csv(
    csv_path: str,
    output_dir: str,
    provider: str = "qwen3vl",
    model: str = None,
    batch_size: int = 8,
    num_workers: int = 4,
    resume: bool = True,
    vllm_url: str = None,
    tensor_parallel_size: int = None,
    **kwargs
):
    """
    Process CSV file and generate lighting keywords for all images.
    
    Args:
        csv_path: Path to input CSV from Step 2
        output_dir: Directory to save output CSV
        provider: VLM provider ('qwen3vl', 'qwen3vl-server', 'mistral', 'openai')
        model: Model name
        batch_size: Batch size for processing (for qwen3vl)
        num_workers: Number of parallel workers (for API-based providers)
        resume: Resume from checkpoint if available
        vllm_url: vLLM server URL (for qwen3vl-server provider)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        **kwargs: Additional arguments for VLM
    """
    # Load CSV
    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Validate columns
    required_cols = ['image_path']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column")
    
    logger.info(f"Found {len(df)} images to process")
    
    # Setup output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_name = Path(csv_path).stem
    output_csv_path = output_dir / f"{csv_name}_with_keywords.csv"
    checkpoint_path = output_dir / f"{csv_name}_checkpoint.json"
    
    # Check for checkpoint
    processed_indices = set()
    if resume and checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
            processed_indices = set(checkpoint.get('processed_indices', []))
            df['lighting_keywords'] = checkpoint.get('keywords', [None] * len(df))
        logger.info(f"Resuming from checkpoint: {len(processed_indices)} already processed")
    else:
        df['lighting_keywords'] = None
    
    # Initialize VLM with provider-specific options
    logger.info(f"Initializing VLM: provider={provider}, model={model}")
    
    vlm_kwargs = kwargs.copy()
    if provider == "qwen3vl-server" and vllm_url:
        vlm_kwargs['base_url'] = vllm_url
    if provider == "qwen3vl" and tensor_parallel_size:
        vlm_kwargs['tensor_parallel_size'] = tensor_parallel_size
    
    vlm = get_vlm(provider, model, **vlm_kwargs)
    
    # Process images
    to_process = [i for i in range(len(df)) if i not in processed_indices]
    logger.info(f"Processing {len(to_process)} images...")
    
    # Check if VLM supports batching
    supports_batching = hasattr(vlm, 'generate_keywords_batch')
    
    if provider == "qwen3vl" and supports_batching:
        # Use batched inference for Qwen3-VL (most efficient)
        logger.info(f"Using batched inference with batch_size={batch_size}")
        
        for batch_start in tqdm(range(0, len(to_process), batch_size), desc="Generating keywords (batched)"):
            batch_end = min(batch_start + batch_size, len(to_process))
            batch_indices = to_process[batch_start:batch_end]
            batch_paths = [df.loc[idx, 'image_path'] for idx in batch_indices]
            
            try:
                keywords_list = vlm.generate_keywords_batch(batch_paths)
                
                for idx, keywords in zip(batch_indices, keywords_list):
                    df.loc[idx, 'lighting_keywords'] = keywords
                    processed_indices.add(idx)
                
                # Save checkpoint
                if len(processed_indices) % 100 == 0:
                    with open(checkpoint_path, 'w') as f:
                        json.dump({
                            'processed_indices': list(processed_indices),
                            'keywords': df['lighting_keywords'].tolist()
                        }, f)
                        
            except Exception as e:
                logger.error(f"Batch failed: {e}")
                # Fallback to single processing for this batch
                for idx in batch_indices:
                    image_path = df.loc[idx, 'image_path']
                    keywords = process_single_image(vlm, image_path)
                    df.loc[idx, 'lighting_keywords'] = keywords
                    processed_indices.add(idx)
    
    elif provider in ["mistral", "openai", "qwen3vl-server"]:
        # Use parallel processing for API-based VLMs
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for idx in to_process:
                image_path = df.loc[idx, 'image_path']
                future = executor.submit(process_single_image, vlm, image_path)
                futures[future] = idx
            
            with tqdm(total=len(to_process), desc="Generating keywords") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        keywords = future.result()
                        df.loc[idx, 'lighting_keywords'] = keywords
                        processed_indices.add(idx)
                        
                        # Save checkpoint periodically
                        if len(processed_indices) % 100 == 0:
                            with open(checkpoint_path, 'w') as f:
                                json.dump({
                                    'processed_indices': list(processed_indices),
                                    'keywords': df['lighting_keywords'].tolist()
                                }, f)
                    except Exception as e:
                        logger.error(f"Failed to process index {idx}: {e}")
                        df.loc[idx, 'lighting_keywords'] = f"ERROR: {str(e)}"
                    
                    pbar.update(1)
    else:
        # Sequential processing
        for idx in tqdm(to_process, desc="Generating keywords"):
            image_path = df.loc[idx, 'image_path']
            keywords = process_single_image(vlm, image_path)
            df.loc[idx, 'lighting_keywords'] = keywords
            processed_indices.add(idx)
            
            # Save checkpoint periodically
            if len(processed_indices) % 50 == 0:
                with open(checkpoint_path, 'w') as f:
                    json.dump({
                        'processed_indices': list(processed_indices),
                        'keywords': df['lighting_keywords'].tolist()
                    }, f)
    
    # Save final CSV
    df.to_csv(output_csv_path, index=False)
    
    # Cleanup checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    
    # Print summary
    success_count = df['lighting_keywords'].notna().sum()
    error_count = df['lighting_keywords'].str.startswith('ERROR:', na=False).sum() if success_count > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"✅ KEYWORD GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Provider: {provider}")
    print(f"Model: {model or 'default'}")
    print(f"Total images: {len(df)}")
    print(f"Successful: {success_count - error_count}")
    print(f"Errors: {error_count}")
    print(f"\nOutput CSV: {output_csv_path}")
    print(f"\n📋 Next Step: Prepare training data")
    print(f"   The CSV now has 4 columns:")
    print(f"   - image_path: Original image (will be OUTPUT for training)")
    print(f"   - lighting_score: CLIP score")
    print(f"   - output_image_path: Degraded image (will be INPUT for training)")
    print(f"   - lighting_keywords: Edit instruction")
    print(f"{'='*60}\n")
    
    return output_csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate lighting keywords for images using VLM (Step 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Use Qwen3-VL-30B with vLLM (recommended, free)
  python generate_keywords.py --csv path/to/csv --output_dir ./output

  # Use vLLM server (start server first)
  # Server: vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct --port 8000
  python generate_keywords.py --csv path/to/csv --provider qwen3vl-server --vllm_url http://localhost:8000/v1

  # Use Mistral API
  export MISTRAL_API_KEY="your-key"
  python generate_keywords.py --csv path/to/csv --provider mistral

  # Use OpenAI API  
  export OPENAI_API_KEY="your-key"
  python generate_keywords.py --csv path/to/csv --provider openai --model gpt-4o-mini
        """
    )
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to CSV from Step 2 (with image_path and output_image_path)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save output CSV")
    parser.add_argument("--provider", type=str, default="qwen3vl",
                        choices=["qwen3vl", "qwen3vl-server", "mistral", "openai"],
                        help="VLM provider (default: qwen3vl)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: provider's default)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for qwen3vl (default: 8)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers for API providers")
    parser.add_argument("--vllm_url", type=str, default=None,
                        help="vLLM server URL (for qwen3vl-server provider)")
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                        help="Number of GPUs for tensor parallelism (qwen3vl)")
    parser.add_argument("--no_resume", action="store_true",
                        help="Don't resume from checkpoint")
    
    args = parser.parse_args()
    
    process_csv(
        csv_path=args.csv,
        output_dir=args.output_dir,
        provider=args.provider,
        model=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        vllm_url=args.vllm_url,
        tensor_parallel_size=args.tensor_parallel_size,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()
