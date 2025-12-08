"""
Stage 4: VLM captioning using Qwen2.5-VL-7B.
"""

import torch
from PIL import Image
from typing import Dict, Any
import logging

from ..pipeline.base_stage import BaseStage

logger = logging.getLogger(__name__)


class CaptioningStage(BaseStage):
    """
    Generate detailed captions with lighting descriptions using Qwen2.5-VL.
    """

    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        super().__init__(config, device)
        self.model_config = config.get('qwen_vl', {})
        self.caption_prompt = config.get('captioning_prompt', self._default_prompt())
        self.processor = None

    def _default_prompt(self) -> str:
        """Default captioning prompt."""
        return """Describe this portrait image in detail. Include:
1. The person's appearance and facial features
2. The lighting conditions (direction, quality, intensity)
3. Shadow characteristics (hard/soft, direction)
4. Overall color palette and mood
5. Composition and framing

Be specific about lighting and shadows."""

    def load_model(self):
        """
        Load Qwen2.5-VL model with 8-bit quantization.
        """
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from transformers import BitsAndBytesConfig

            logger.info("Loading Qwen2.5-VL model...")

            # Get configuration
            model_name = self.model_config.get('model_name', 'Qwen/Qwen2.5-VL-7B-Instruct')
            use_8bit = self.model_config.get('use_8bit', True)
            max_memory_gb = self.config.get('memory', {}).get('max_gpu_memory_gb', 22)

            logger.info(f"Model: {model_name}")
            logger.info(f"Using 8-bit quantization: {use_8bit}")

            # Configure quantization
            if use_8bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16,
                    bnb_8bit_use_double_quant=True
                )
            else:
                bnb_config = None

            # Load model
            logger.info("Loading model weights (this may take a while)...")

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=bnb_config if use_8bit else None,
                max_memory={0: f"{max_memory_gb}GiB"},
                trust_remote_code=True
            )

            # Load processor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            logger.info("Qwen2.5-VL model loaded successfully")
            self.log_memory_usage("After loading Qwen2.5-VL: ")

        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            logger.error("Please install: pip install transformers bitsandbytes")
            raise
        except Exception as e:
            logger.error(f"Failed to load Qwen2.5-VL model: {e}")
            raise

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate detailed caption for shadow image.

        Args:
            input_data: Dictionary with 'shadow_image' (PIL Image) and 'image_id'

        Returns:
            Dictionary with caption and other metadata
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        shadow_image = input_data['shadow_image']
        image_id = input_data['image_id']

        logger.info(f"Generating caption for image {image_id}")

        # Ensure image is RGB
        if isinstance(shadow_image, Image.Image) and shadow_image.mode != 'RGB':
            shadow_image = shadow_image.convert('RGB')

        # Prepare messages for chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": shadow_image},
                    {"type": "text", "text": self.caption_prompt}
                ]
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Prepare inputs
        logger.debug("Preparing model inputs...")
        inputs = self.processor(
            text=[text],
            images=[shadow_image],
            return_tensors="pt",
            padding=True
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Get generation parameters
        max_new_tokens = self.model_config.get('max_new_tokens', 256)
        temperature = self.model_config.get('temperature', 0.7)
        do_sample = self.model_config.get('do_sample', True)

        # Generate caption
        logger.debug(f"Generating caption (max_tokens={max_new_tokens})...")

        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )

            # Decode caption
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs['input_ids'], output_ids)
            ]

            caption = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]

            # Clean up caption
            caption = caption.strip()

        except Exception as e:
            logger.error(f"Error during caption generation: {e}")
            logger.warning("Using fallback caption")
            caption = "A portrait image with lighting and shadows."

        logger.info(f"Caption generated ({len(caption)} chars)")
        logger.debug(f"Caption: {caption[:100]}...")

        # Pass through previous data and add caption
        output = input_data.copy()
        output.update({
            'caption': caption,
            'caption_length': len(caption)
        })

        return output

    def unload_model(self):
        """
        Unload Qwen2.5-VL model and processor.
        """
        if self.processor is not None:
            del self.processor
            self.processor = None

        super().unload_model()
