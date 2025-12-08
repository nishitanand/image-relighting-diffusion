"""
Stage 2: Albedo extraction using multiple methods with fallbacks.

Implements 3 albedo extraction methods:
1. IntrinsicAnything (advanced, diffusion-based)
2. Multi-Scale Retinex (traditional, robust)
3. LAB-based simple method (ultimate fallback)

Per IC-Light paper Section 3.1: Uses multiple albedo extraction methods
to generate diverse training data.
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional
import logging
import os

from ..pipeline.base_stage import BaseStage
from ..utils.albedo_methods import (
    extract_albedo_retinex,
    extract_albedo_lab_based,
    get_robust_albedo,
    blend_with_original
)

logger = logging.getLogger(__name__)


class AlbedoExtractionStage(BaseStage):
    """
    Extract albedo (material reflectance) from foreground images.

    Tries multiple methods in order of sophistication with fallback.
    """

    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        super().__init__(config, device)
        self.albedo_config = config.get('albedo_extraction', {})
        self.intrinsic_anything_model = None

    def load_model(self):
        """
        Load IntrinsicAnything model (Method 1).

        If this fails, we'll fall back to traditional methods.
        """
        try:
            logger.info("Attempting to load IntrinsicAnything model...")

            # Check if IntrinsicAnything is available
            intrinsic_config = self.albedo_config.get('methods', {}).get('intrinsic_anything', {})

            if not intrinsic_config.get('enabled', True):
                logger.info("IntrinsicAnything disabled in config, skipping")
                return

            checkpoint_dir = intrinsic_config.get('checkpoint_dir', 'models/intrinsic_anything')

            # Try to load IntrinsicAnything
            # Note: This requires the actual IntrinsicAnything package to be installed
            # We'll try to import it, but if it fails, we'll use fallback methods
            try:
                # Check if model files exist
                if not os.path.exists(checkpoint_dir):
                    logger.warning(f"IntrinsicAnything checkpoint not found at: {checkpoint_dir}")
                    logger.info("To use IntrinsicAnything, run:")
                    logger.info("  huggingface-cli download --repo-type space LittleFrog/IntrinsicAnything")
                    return

                # Try importing (this will fail if package not installed)
                # from intrinsic_anything import IntrinsicModel
                # self.intrinsic_anything_model = IntrinsicModel(model_dir=checkpoint_dir)
                # self.intrinsic_anything_model.to(self.device)

                logger.warning("IntrinsicAnything import not yet implemented")
                logger.info("Falling back to traditional albedo methods")

            except ImportError as e:
                logger.warning(f"IntrinsicAnything not available: {e}")
                logger.info("Install with: pip install git+https://github.com/zju3dv/IntrinsicAnything.git")
                logger.info("Falling back to traditional methods")

        except Exception as e:
            logger.warning(f"Could not load IntrinsicAnything: {e}")
            logger.info("Will use traditional albedo extraction methods")

        # Traditional methods don't need model loading
        logger.info("Albedo extraction stage ready (using traditional methods)")

    def _extract_with_intrinsic_anything(self, foreground: Image.Image) -> Optional[Image.Image]:
        """
        Extract albedo using IntrinsicAnything (Method 1).

        Args:
            foreground: PIL Image

        Returns:
            Albedo PIL Image, or None if failed
        """
        if self.intrinsic_anything_model is None:
            return None

        try:
            logger.info("Extracting albedo with IntrinsicAnything")

            # Run inference
            with torch.no_grad():
                albedo, specular = self.intrinsic_anything_model.predict(
                    foreground,
                    ddim_steps=100,
                    batch_size=1
                )

            logger.info("IntrinsicAnything extraction successful")
            return albedo

        except Exception as e:
            logger.error(f"IntrinsicAnything extraction failed: {e}")
            return None

    def _extract_with_retinex(self, foreground: Image.Image) -> Image.Image:
        """
        Extract albedo using Multi-Scale Retinex (Method 2).

        Args:
            foreground: PIL Image

        Returns:
            Albedo PIL Image
        """
        logger.info("Extracting albedo with Multi-Scale Retinex")

        scales = self.albedo_config.get('methods', {}).get('retinex', {}).get('scales', [15, 80, 250])

        albedo = extract_albedo_retinex(
            foreground,
            scales=scales,
            apply_color_balance=True
        )

        logger.info("Retinex extraction successful")
        return albedo

    def _extract_with_lab(self, foreground: Image.Image) -> Image.Image:
        """
        Extract albedo using LAB-based method (Method 3).

        Args:
            foreground: PIL Image

        Returns:
            Albedo PIL Image
        """
        logger.info("Extracting albedo with LAB-based method")

        albedo = extract_albedo_lab_based(foreground)

        logger.info("LAB extraction successful")
        return albedo

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract albedo from foreground image.

        Tries methods in order with fallback:
        1. IntrinsicAnything (if available)
        2. Multi-Scale Retinex
        3. LAB-based (always works)

        Args:
            input_data: Dictionary with 'foreground' (PIL Image) and 'image_id'

        Returns:
            Dictionary with albedo and method used
        """
        foreground = input_data['foreground']
        image_id = input_data['image_id']

        logger.info(f"Extracting albedo for image {image_id}")

        albedo = None
        method_used = None

        # Method 1: Try IntrinsicAnything
        if self.intrinsic_anything_model is not None:
            albedo = self._extract_with_intrinsic_anything(foreground)
            if albedo is not None:
                method_used = "intrinsic_anything"

        # Method 2: Try Retinex if Method 1 failed
        if albedo is None:
            try:
                albedo = self._extract_with_retinex(foreground)
                method_used = "retinex"
            except Exception as e:
                logger.error(f"Retinex extraction failed: {e}")

        # Method 3: LAB-based fallback
        if albedo is None:
            try:
                albedo = self._extract_with_lab(foreground)
                method_used = "lab"
            except Exception as e:
                logger.error(f"LAB extraction failed: {e}")
                # Ultimate fallback: use foreground as albedo
                albedo = foreground
                method_used = "none"

        logger.info(f"Albedo extraction completed using method: {method_used}")

        # Apply blending with original to reduce whiteness
        blend_config = self.albedo_config.get('blend_with_original', {})
        enabled = blend_config.get('enabled', True)  # Enabled by default
        blend_ratio_range = blend_config.get('ratio_range', [0.15, 0.25])

        if enabled and albedo is not None:
            # Random blend ratio from range
            import random
            blend_ratio = random.uniform(*blend_ratio_range)

            # Blend albedo with original foreground
            albedo_blended = blend_with_original(
                albedo=albedo,
                original=foreground,
                blend_ratio=blend_ratio
            )

            logger.info(f"Applied blending with ratio {blend_ratio:.3f}")
            albedo = albedo_blended
            blend_ratio_used = blend_ratio
        else:
            blend_ratio_used = 0.0
            logger.info("Blending disabled, using pure albedo")

        # Pass through previous data and add albedo
        output = input_data.copy()
        output.update({
            'albedo': albedo,
            'albedo_method': method_used,
            'albedo_blend_ratio': blend_ratio_used,
            'foreground': foreground
        })

        return output

    def unload_model(self):
        """Unload IntrinsicAnything model if loaded."""
        if self.intrinsic_anything_model is not None:
            del self.intrinsic_anything_model
            self.intrinsic_anything_model = None

        super().unload_model()
