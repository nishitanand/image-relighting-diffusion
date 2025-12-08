"""
Step 3: Generate Lighting Keywords/Captions using Vision-Language Model

This script takes the CSV output from relightingDataGen-parallel (Step 2) and generates
lighting description keywords for each original image using a VLM like Mistral 3.

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
    python generate_keywords.py --csv path/to/csv --output_dir ./output
    
    # With specific model
    python generate_keywords.py --csv path/to/csv --model mistral-small-latest
    
    # With batch processing
    python generate_keywords.py --csv path/to/csv --batch_size 8 --num_gpus 4
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


class TransformersVLM:
    """Local VLM using HuggingFace Transformers (e.g., LLaVA, Qwen-VL)."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct", device: str = "cuda"):
        """
        Initialize local VLM.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda, cpu)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
    
    def load_model(self):
        """Load model and processor."""
        if self.model is not None:
            return
            
        logger.info(f"Loading {self.model_name}...")
        
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            import torch
            
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
            logger.info(f"✅ Model loaded successfully")
        except ImportError:
            raise ImportError("Install: pip install transformers qwen-vl-utils")
    
    def generate_keywords(self, image_path: str, prompt: str = None) -> str:
        """Generate lighting keywords for an image."""
        self.load_model()
        
        prompt = prompt or LIGHTING_KEYWORD_PROMPT
        
        from qwen_vl_utils import process_vision_info
        import torch
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output.strip()


# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def get_vlm(provider: str, model: str = None, **kwargs):
    """
    Get VLM instance based on provider.
    
    Args:
        provider: 'mistral', 'openai', or 'local'
        model: Model name (optional, uses default for provider)
        **kwargs: Additional arguments for the VLM
        
    Returns:
        VLM instance
    """
    if provider == "mistral":
        model = model or "pixtral-large-latest"
        return MistralVLM(model=model, **kwargs)
    elif provider == "openai":
        model = model or "gpt-4o"
        return OpenAIVLM(model=model, **kwargs)
    elif provider == "local":
        model = model or "Qwen/Qwen2-VL-7B-Instruct"
        return TransformersVLM(model_name=model, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'mistral', 'openai', or 'local'")


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
    provider: str = "mistral",
    model: str = None,
    batch_size: int = 1,
    num_workers: int = 4,
    resume: bool = True,
    **kwargs
):
    """
    Process CSV file and generate lighting keywords for all images.
    
    Args:
        csv_path: Path to input CSV from Step 2
        output_dir: Directory to save output CSV
        provider: VLM provider ('mistral', 'openai', 'local')
        model: Model name
        batch_size: Batch size for processing
        num_workers: Number of parallel workers (for API-based providers)
        resume: Resume from checkpoint if available
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
    
    # Initialize VLM
    logger.info(f"Initializing VLM: provider={provider}, model={model}")
    vlm = get_vlm(provider, model, **kwargs)
    
    # Process images
    to_process = [i for i in range(len(df)) if i not in processed_indices]
    logger.info(f"Processing {len(to_process)} images...")
    
    if provider in ["mistral", "openai"]:
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
        # Sequential processing for local models
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
    error_count = df['lighting_keywords'].str.startswith('ERROR:').sum() if success_count > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"✅ KEYWORD GENERATION COMPLETE")
    print(f"{'='*60}")
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
        description="Generate lighting keywords for images using VLM (Step 3)"
    )
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to CSV from Step 2 (with image_path and output_image_path)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save output CSV")
    parser.add_argument("--provider", type=str, default="mistral",
                        choices=["mistral", "openai", "local"],
                        help="VLM provider (default: mistral)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: provider's default)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers for API providers")
    parser.add_argument("--no_resume", action="store_true",
                        help="Don't resume from checkpoint")
    
    args = parser.parse_args()
    
    process_csv(
        csv_path=args.csv,
        output_dir=args.output_dir,
        provider=args.provider,
        model=args.model,
        num_workers=args.num_workers,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()

