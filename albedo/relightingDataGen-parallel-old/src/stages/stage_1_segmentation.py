"""
Stage 1: Background removal using SAM2 (Segment Anything Model 2).
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Any
import logging

from ..pipeline.base_stage import BaseStage

logger = logging.getLogger(__name__)


class SAM2SegmentationStage(BaseStage):
    """
    SAM2-based foreground/background segmentation.

    Uses automatic face detection to segment faces from backgrounds.
    """

    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        super().__init__(config, device)
        self.model_config = config.get('sam2', {})
        self.predictor = None

    def load_model(self):
        """
        Load SAM2 model and predictor.
        """
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            logger.info("Loading SAM2 model...")

            # # Get model configuration
            # model_type = self.model_config.get('model_type', 'sam2.1_hiera_l')
            # checkpoint_name = self.model_config.get('checkpoint_name', 'sam2.1_hiera_large.pt')
            # config_name = self.model_config.get('config_name', 'sam2.1_hiera_l.yaml')

            # # Build checkpoint path
            # models_root = self.config['paths']['models_root']
            # checkpoint_path = f"{models_root}/sam2/checkpoints/{checkpoint_name}"
            # config_path = f"{models_root}/sam2/configs/sam2.1/{config_name}"

            logger.info(f"Loading SAM2 checkpoint from: HuggingFace")


            self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")

            logger.info("SAM2 model loaded successfully")
            self.log_memory_usage("After loading SAM2: ")

        except ImportError as e:
            logger.error(f"Failed to import SAM2: {e}")
            logger.error("Please install SAM2: pip install git+https://github.com/facebookresearch/sam2.git")
            raise
        except Exception as e:
            logger.error(f"Failed to load SAM2 model: {e}")
            raise

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Segment foreground (face) from background.

        Args:
            input_data: Dictionary with 'image' (PIL Image) and 'image_id'

        Returns:
            Dictionary with foreground, background, mask, and image_id
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        image = input_data['image']
        image_id = input_data['image_id']

        logger.info(f"Processing image {image_id} with SAM2")

        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        # Set image for prediction
        self.predictor.set_image(image_np)

        # Get configuration
        multimask = self.model_config.get('multimask_output', False)

        # SAM2 doesn't support text prompts natively, so we use point prompts
        # For face segmentation, we use the center point which typically contains the face
        height, width = image_np.shape[:2]

        # Use center point as a positive prompt for face
        # Can also be configured via config
        point_coords = self.model_config.get('point_coords', [[width // 2, height // 2]])
        point_labels = self.model_config.get('point_labels', [1])  # 1 = foreground

        logger.debug(f"Running segmentation with point prompts at: {point_coords}")

        with torch.no_grad():
            masks, scores, logits = self.predictor.predict(
                point_coords=np.array(point_coords),
                point_labels=np.array(point_labels),
                multimask_output=multimask
            )

        # Use the best mask (highest score)
        if multimask:
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = scores[best_idx]
        else:
            mask = masks[0]
            score = scores[0] if len(scores) > 0 else 1.0

        logger.info(f"Segmentation score: {score:.3f}")

        # Create foreground and background images
        foreground_np = image_np.copy()
        background_np = image_np.copy()

        

        # Expand mask to 3 channels if needed
        if len(mask.shape) == 2:
            mask_3ch = np.stack([mask] * 3, axis=-1)
            print(1, type(mask_3ch))
            logger.info(f"{1}, {type(mask_3ch)}")
        else:
            mask_3ch = mask
            print(2, type(mask_3ch))
            logger.info(f"{2}, {type(mask_3ch)}")

        # Ensure mask is boolean (single-channel)
        mask_bool = mask.astype(bool)

        # Expand to 3 channels
        mask_3ch = np.repeat(mask_bool[:, :, None], 3, axis=2)

        # Apply mask
        foreground_np = np.where(mask_3ch, foreground_np, 0)
        background_np = np.where(~mask_3ch, background_np, 0)

        # Convert back to PIL Images
        foreground = Image.fromarray(foreground_np.astype(np.uint8))
        background = Image.fromarray(background_np.astype(np.uint8))
        mask_img = Image.fromarray((mask_bool * 255).astype(np.uint8))


        # Calculate coverage statistics
        coverage = mask.sum() / mask.size
        logger.info(f"Mask coverage: {coverage:.1%} of image")

        return {
            'foreground': foreground,
            'background': background,
            'mask': mask_img,
            'mask_array': mask,
            'segmentation_score': float(score),
            'mask_coverage': float(coverage),
            'image_id': image_id,
            'original_image': image
        }

    def unload_model(self):
        """
        Unload SAM3 model and predictor.
        """
        if self.predictor is not None:
            del self.predictor
            self.predictor = None

        super().unload_model()
