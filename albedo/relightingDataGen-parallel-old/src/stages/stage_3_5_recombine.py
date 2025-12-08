"""
Stage 3.5: Background Recombination

Combines degraded foreground with original background to create complete images.
"""

import numpy as np
from PIL import Image
from typing import Dict, Any
import logging

from ..pipeline.base_stage import BaseStage

logger = logging.getLogger(__name__)


class BackgroundRecombinationStage(BaseStage):
    """
    Recombine degraded foreground with original background.

    Takes the degraded foreground from Stage 3 and composites it with
    the original background from Stage 1 using the segmentation mask.
    """

    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        super().__init__(config, device)
        # Get background configuration
        self.background_config = config.get('background', {})
        self.use_gray_background = self.background_config.get('use_gray', True)
        self.gray_color = tuple(self.background_config.get('gray_color', [128, 128, 128]))

    def load_model(self):
        """
        No model to load for background recombination.
        """
        logger.info("Background recombination stage - no model needed")
        pass

    def recombine_with_background(
        self,
        degraded_foreground: Image.Image,
        background: Image.Image,
        mask: Image.Image,
        use_gray_background: bool = True,
        gray_color: tuple = (128, 128, 128)
    ) -> Image.Image:
        """
        Recombine degraded foreground with background.

        Uses alpha compositing: result = foreground * mask + background * (1 - mask)

        Args:
            degraded_foreground: Degraded foreground image (PIL)
            background: Original background image (PIL, used only for size reference)
            mask: Foreground mask (PIL, grayscale)
            use_gray_background: If True, use gray background instead of original (default: True)
            gray_color: RGB tuple for gray background (default: (128, 128, 128) = #808080)

        Returns:
            Composite image (PIL)
        """
        # Convert foreground and mask to numpy arrays
        foreground_np = np.array(degraded_foreground).astype(np.float32)
        mask_np = np.array(mask).astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Get image dimensions
        height, width = foreground_np.shape[:2]

        # Create background
        if use_gray_background:
            # Create solid gray background (#808080)
            logger.info(f"Using gray background (RGB: {gray_color})")
            background_np = np.full((height, width, 3), gray_color, dtype=np.float32)
        else:
            # Use original background
            logger.info("Using original background")
            background_np = np.array(background).astype(np.float32)

        # Ensure mask is 2D
        if len(mask_np.shape) == 3:
            mask_np = mask_np[:, :, 0]

        # Expand mask to 3 channels
        mask_3ch = mask_np[:, :, np.newaxis]

        # Alpha compositing: foreground * mask + background * (1 - mask)
        composite_np = foreground_np * mask_3ch + background_np * (1 - mask_3ch)

        # Convert back to PIL
        composite = Image.fromarray(composite_np.astype(np.uint8))

        return composite

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recombine degraded foreground with background.

        Args:
            input_data: Dictionary with:
                - 'degraded_image': Degraded foreground (PIL Image)
                - 'background': Original background (PIL Image)
                - 'mask': Segmentation mask (PIL Image)
                - 'image_id': Image ID

        Returns:
            Dictionary with composite image added
        """
        image_id = input_data['image_id']

        logger.info(f"Recombining foreground with background for image {image_id}")

        # Check if we have all required inputs
        if 'degraded_image' not in input_data:
            logger.error("No degraded_image found in input data")
            raise ValueError("degraded_image is required for background recombination")

        if 'background' not in input_data or 'mask' not in input_data:
            logger.warning("Background or mask not found, using degraded image as-is")
            composite = input_data['degraded_image']
        else:
            # Perform recombination with configured background type
            composite = self.recombine_with_background(
                degraded_foreground=input_data['degraded_image'],
                background=input_data['background'],
                mask=input_data['mask'],
                use_gray_background=self.use_gray_background,
                gray_color=self.gray_color
            )

        logger.info(f"Background recombination completed for image {image_id}")

        # Pass through all previous data and add composite
        output = input_data.copy()
        output.update({
            'composite_image': composite,  # Final composite with background
            'degraded_foreground': input_data['degraded_image']  # Keep reference to foreground-only
        })

        return output

    def unload_model(self):
        """
        No model to unload.
        """
        pass
