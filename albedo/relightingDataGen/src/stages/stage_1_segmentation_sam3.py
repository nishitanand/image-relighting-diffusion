"""
Stage 1: Background removal using SAM3 (Segment Anything Model 3) with text prompting.

SAM3 supports native text prompting, enabling more robust segmentation of "entire person"
without requiring manual point-based prompts.
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List
import logging

from ..pipeline.base_stage import BaseStage

logger = logging.getLogger(__name__)


class SAM3SegmentationStage(BaseStage):
    """
    SAM3-based foreground/background segmentation with text prompting.

    Uses text prompts like "person" to automatically segment all people in the image.
    Falls back to SAM2 if SAM3 is not available.
    """

    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        super().__init__(config, device)
        self.model_config = config.get('sam3', {})
        self.processor = None
        self.model = None
        self.use_sam3 = self.model_config.get('enabled', True)

    def load_model(self):
        """
        Load SAM3 model with text prompting support.
        Falls back to SAM2 if SAM3 is unavailable.
        """
        if not self.use_sam3:
            logger.info("SAM3 disabled, using SAM2 fallback")
            self._load_sam2_fallback()
            return

        try:
            logger.info("Attempting to load SAM3 with text prompting...")

            # Try to import SAM3
            try:
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
                sam3_available = True
            except ImportError:
                logger.warning("SAM3 not installed, falling back to SAM2")
                sam3_available = False

            if not sam3_available:
                self._load_sam2_fallback()
                return

            # Load SAM3 model (no arguments needed)
            logger.info("Loading SAM3 model...")
            self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model)
            
            # Move model to device
            self.model = self.model.to(self.device)

            logger.info("SAM3 model loaded successfully with text prompting support")
            self.log_memory_usage("After loading SAM3: ")

        except Exception as e:
            logger.error(f"Failed to load SAM3: {e}")
            logger.warning("Falling back to SAM2 point-based prompting")
            self._load_sam2_fallback()

    def _load_sam2_fallback(self):
        """Load SAM2 as fallback when SAM3 is unavailable."""
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            logger.info("Loading SAM2 fallback from HuggingFace...")
            self.processor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")
            self.use_sam3 = False

            logger.info("SAM2 fallback loaded successfully")
            self.log_memory_usage("After loading SAM2: ")

        except ImportError as e:
            logger.error(f"Failed to import SAM2: {e}")
            logger.error("Please install SAM2: pip install git+https://github.com/facebookresearch/sam2.git")
            raise
        except Exception as e:
            logger.error(f"Failed to load SAM2 fallback: {e}")
            raise

    def _segment_with_sam3_text(
        self,
        image_np: np.ndarray,
        text_prompt: str = "person"
    ) -> tuple:
        """
        Segment using SAM3 text prompting.

        Args:
            image_np: Image as numpy array (H, W, 3)
            text_prompt: Text description (e.g., "person", "entire person")

        Returns:
            Tuple of (mask, score)
        """
        logger.info(f"Segmenting with SAM3 text prompt: '{text_prompt}'")

        # Set image for inference
        inference_state = self.processor.set_image(Image.fromarray(image_np))

        # Use text prompt to segment ALL instances
        output = self.processor.set_text_prompt(
            state=inference_state,
            prompt=text_prompt
        )

        # Get masks, scores, and bounding boxes
        masks = output['masks']  # Shape: (N, H, W) where N = number of instances
        scores = output.get('scores', np.ones(len(masks)))
        
        # Convert masks to numpy if they are tensors and squeeze extra dimensions
        if len(masks) > 0:
            if torch.is_tensor(masks[0]):
                masks = [mask.cpu().numpy() if mask.is_cuda else mask.numpy() for mask in masks]
            if torch.is_tensor(scores):
                scores = scores.cpu().numpy() if scores.is_cuda else scores.numpy()
            
            # Squeeze any extra dimensions from masks (SAM3 might return (1, H, W) or (H, W, 1))
            masks = [np.squeeze(mask) for mask in masks]
            
            # Ensure masks are 2D (H, W)
            masks = [mask if mask.ndim == 2 else mask.reshape(image_np.shape[:2]) for mask in masks]

        logger.info(f"SAM3 detected {len(masks)} instances of '{text_prompt}'")

        # Combine all masks into a single mask (union of all persons)
        if len(masks) > 0:
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = np.logical_or(combined_mask, mask)

            # Use average score
            score = float(np.mean(scores))
        else:
            # No instances found
            logger.warning(f"No instances of '{text_prompt}' detected")
            combined_mask = np.zeros(image_np.shape[:2], dtype=bool)
            score = 0.0

        return combined_mask, score

    def _segment_with_sam2_points(
        self,
        image_np: np.ndarray,
        point_coords: Optional[List[List[int]]] = None,
        point_labels: Optional[List[int]] = None
    ) -> tuple:
        """
        Segment using SAM2 point prompting (fallback).

        Args:
            image_np: Image as numpy array (H, W, 3)
            point_coords: List of [x, y] coordinates
            point_labels: List of labels (1=foreground, 0=background)

        Returns:
            Tuple of (mask, score)
        """
        logger.info("Segmenting with SAM2 point prompts")

        # Set image
        self.processor.set_image(image_np)

        # Default to center point if not provided
        if point_coords is None:
            height, width = image_np.shape[:2]
            point_coords = [[width // 2, height // 2]]
            point_labels = [1]

        multimask = self.model_config.get('multimask_output', False)

        # Predict
        with torch.no_grad():
            masks, scores, logits = self.processor.predict(
                point_coords=np.array(point_coords),
                point_labels=np.array(point_labels),
                multimask_output=multimask
            )

        # Use best mask
        if multimask:
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = scores[best_idx]
        else:
            mask = masks[0]
            score = scores[0] if len(scores) > 0 else 1.0

        return mask, float(score)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Segment foreground (person) from background.

        Uses SAM3 text prompting if available, otherwise falls back to SAM2 points.

        Args:
            input_data: Dictionary with 'image' (PIL Image) and 'image_id'

        Returns:
            Dictionary with foreground, background, mask, and segmentation metadata
        """
        if self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        image = input_data['image']
        image_id = input_data['image_id']

        logger.info(f"Processing image {image_id} for segmentation")

        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        # Segment based on available model
        if self.use_sam3:
            # SAM3: Use text prompting
            text_prompt = self.model_config.get('text_prompt', 'person')
            mask, score = self._segment_with_sam3_text(image_np, text_prompt)
            segmentation_method = 'sam3_text'
        else:
            # SAM2: Use point prompting
            point_coords = self.model_config.get('point_coords', None)
            point_labels = self.model_config.get('point_labels', None)
            mask, score = self._segment_with_sam2_points(image_np, point_coords, point_labels)
            segmentation_method = 'sam2_points'

        logger.info(f"Segmentation score: {score:.3f} (method: {segmentation_method})")

        # Create foreground and background images
        foreground_np = image_np.copy()
        background_np = image_np.copy()

        # Ensure mask is boolean
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
        coverage = mask_bool.sum() / mask_bool.size
        logger.info(f"Mask coverage: {coverage:.1%} of image")

        return {
            'foreground': foreground,
            'background': background,
            'mask': mask_img,
            'mask_array': mask,
            'segmentation_score': float(score),
            'mask_coverage': float(coverage),
            'segmentation_method': segmentation_method,
            'image_id': image_id,
            'original_image': image
        }

    def unload_model(self):
        """Unload segmentation model."""
        if self.processor is not None:
            del self.processor
            self.processor = None

        if self.model is not None:
            del self.model
            self.model = None

        super().unload_model()


# Backwards compatibility alias
SAM2SegmentationStage = SAM3SegmentationStage
