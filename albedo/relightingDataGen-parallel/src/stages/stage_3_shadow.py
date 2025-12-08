"""
Stage 3: Degradation synthesis using multiple methods.

Implements 3 degradation types per IC-Light paper Section 3.1:
A. Soft shading (normal-based, Lambertian)
B. Hard shadow (procedural shadow patterns)
C. Specular reflection (Phong model)

Randomly selects one degradation type per image to create diverse training data.
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional
import logging
import random

from ..pipeline.base_stage import BaseStage
from ..utils.normal_estimation import NormalEstimator, get_normal_map
from ..utils.shading_synthesis import generate_random_soft_shading
from ..utils.shadow_generation import (
    generate_random_hard_shadow,
    generate_normal_aware_shadow_degradation
)
from ..utils.advanced_shading import generate_random_advanced_shading

logger = logging.getLogger(__name__)


class DegradationSynthesisStage(BaseStage):
    """
    Generate degradation images with altered illumination.

    Per IC-Light paper Section 3.1: Generate degradation images that share
    the same intrinsic albedo but have completely altered illuminations.
    """

    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        super().__init__(config, device)
        self.degradation_config = config.get('degradation', {})
        self.normal_estimator = None

    def load_model(self):
        """
        Load MiDaS model for normal estimation (needed for soft shading).
        """
        try:
            logger.info("Loading MiDaS model for normal estimation...")

            # Get normal estimation configuration
            soft_shading_config = self.degradation_config.get('soft_shading', {})
            model_type = soft_shading_config.get('normal_estimation', 'MiDaS_small')

            # Map config names to actual MiDaS model types
            model_map = {
                'midas_small': 'MiDaS_small',
                'midas_dpt_hybrid': 'DPT_Hybrid',
                'midas_dpt_large': 'DPT_Large',
                'fast': 'MiDaS_small',
                'quality': 'DPT_Hybrid'
            }

            midas_model = model_map.get(model_type.lower(), 'MiDaS_small')

            # Initialize normal estimator
            self.normal_estimator = NormalEstimator(
                model_type=midas_model,
                device=self.device
            )

            # Load the model
            self.normal_estimator.load_model()

            # Store as model for base class
            self.model = self.normal_estimator

            logger.info(f"MiDaS model ({midas_model}) loaded successfully")
            self.log_memory_usage("After loading MiDaS: ")

        except Exception as e:
            logger.warning(f"Could not load MiDaS model: {e}")
            logger.info("Soft shading will be disabled")
            self.normal_estimator = None

    def _select_degradation_type(self) -> str:
        """
        Randomly select degradation type based on weights from config.

        Returns:
            Degradation type: 'soft_shading', 'hard_shadow', 'specular', or 'advanced_shading'
        """
        # Get weights from config
        soft_weight = self.degradation_config.get('soft_shading', {}).get('weight', 0.3)
        hard_weight = self.degradation_config.get('hard_shadow', {}).get('weight', 0.3)
        specular_weight = self.degradation_config.get('specular', {}).get('weight', 0.1)
        advanced_weight = self.degradation_config.get('advanced_shading', {}).get('weight', 0.3)

        # Create weighted choices
        choices = []
        weights = []

        if soft_weight > 0 and self.normal_estimator is not None:
            choices.append('soft_shading')
            weights.append(soft_weight)

        if hard_weight > 0:
            choices.append('hard_shadow')
            weights.append(hard_weight)

        if specular_weight > 0 and self.normal_estimator is not None:
            choices.append('specular')
            weights.append(specular_weight)

        if advanced_weight > 0 and self.normal_estimator is not None:
            choices.append('advanced_shading')
            weights.append(advanced_weight)

        # Fallback to hard shadow if no choices
        if not choices:
            return 'hard_shadow'

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Random selection
        return random.choices(choices, weights=weights)[0]

    def _generate_soft_shading(
        self,
        albedo: Image.Image,
        image_id: int
    ) -> Tuple[Image.Image, dict]:
        """
        Generate soft shading degradation (Method A).

        Args:
            albedo: Albedo image (PIL)
            image_id: Image ID for logging

        Returns:
            Tuple of (degraded image, metadata)
        """
        logger.info(f"Generating soft shading for image {image_id}")

        # Estimate normal map
        normal_map = self.normal_estimator.estimate_normal(albedo)

        # Get configuration
        soft_config = self.degradation_config.get('soft_shading', {})

        # Generate shading with random parameters
        degraded, metadata = generate_random_soft_shading(
            albedo=albedo,
            normal_map=normal_map,
            config=soft_config
        )

        logger.info(f"Soft shading completed for image {image_id}")
        return degraded, metadata

    def _generate_hard_shadow(
        self,
        albedo: Image.Image,
        image_id: int
    ) -> Tuple[Image.Image, dict]:
        """
        Generate 3D-aware shadow degradation (Method B).

        Uses depth and normal maps to cast realistic shadows based on geometry.

        Args:
            albedo: Albedo image (PIL)
            image_id: Image ID for logging

        Returns:
            Tuple of (degraded image, metadata)
        """
        logger.info(f"Generating 3D-aware shadow for image {image_id}")

        # Get configuration
        hard_config = self.degradation_config.get('hard_shadow', {})

        # Check if we can use 3D-aware shadows (requires MiDaS)
        if self.normal_estimator is not None:
            logger.info("Using depth/normal-aware shadow generation")

            # Estimate depth and normals
            depth_map = self.normal_estimator.estimate_depth(albedo)
            normal_map = self.normal_estimator.depth_to_normal(depth_map)

            # Normalize depth to [0, 1]
            depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

            # Generate 3D-aware shadow
            shadow_softness = hard_config.get('shadow_softness', 0.6)
            degraded, metadata = generate_normal_aware_shadow_degradation(
                albedo=albedo,
                normal_map=normal_map,
                depth_map=depth_normalized,
                shadow_softness=shadow_softness
            )
        else:
            # Fallback to procedural shadows if MiDaS not available
            logger.warning("MiDaS not available, using procedural shadows")
            degraded, metadata = generate_random_hard_shadow(
                albedo=albedo,
                config=hard_config
            )

        logger.info(f"Shadow generation completed for image {image_id}")
        return degraded, metadata

    def _generate_specular(
        self,
        albedo: Image.Image,
        image_id: int
    ) -> Tuple[Image.Image, dict]:
        """
        Generate specular reflection degradation (Method C).

        Combines soft shading with specular highlights.

        Args:
            albedo: Albedo image (PIL)
            image_id: Image ID for logging

        Returns:
            Tuple of (degraded image, metadata)
        """
        logger.info(f"Generating specular reflection for image {image_id}")

        # Estimate normal map
        normal_map = self.normal_estimator.estimate_normal(albedo)

        # Get configuration
        soft_config = self.degradation_config.get('soft_shading', {})
        soft_config['add_specular'] = True  # Force specular

        # Generate shading with specular
        degraded, metadata = generate_random_soft_shading(
            albedo=albedo,
            normal_map=normal_map,
            config=soft_config
        )

        # Update degradation type in metadata
        metadata['degradation_type'] = 'specular'

        logger.info(f"Specular generation completed for image {image_id}")
        return degraded, metadata

    def _generate_advanced_shading(
        self,
        albedo: Image.Image,
        image_id: int
    ) -> Tuple[Image.Image, dict]:
        """
        Generate advanced shading with AO, environment lighting, and multi-light.

        Uses depth and normal maps for realistic illumination with:
        - Screen-Space Ambient Occlusion (contact shadows)
        - Spherical Harmonics environment lighting
        - Multi-light setups (3-point, 2-point, single)
        - Optional specular highlights

        Args:
            albedo: Albedo image (PIL)
            image_id: Image ID for logging

        Returns:
            Tuple of (degraded image, metadata)
        """
        logger.info(f"Generating advanced shading for image {image_id}")

        if self.normal_estimator is None:
            logger.warning("MiDaS not available, falling back to soft shading")
            return self._generate_soft_shading(albedo, image_id)

        # Estimate depth and normals
        depth_map = self.normal_estimator.estimate_depth(albedo)
        normal_map = self.normal_estimator.depth_to_normal(depth_map)

        # Normalize depth to [0, 1]
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

        # Generate advanced shading with random parameters
        degraded, metadata = generate_random_advanced_shading(
            albedo=albedo,
            normal_map=normal_map,
            depth_map=depth_normalized
        )

        logger.info(f"Advanced shading completed for image {image_id}")
        return degraded, metadata

    def _recombine_with_background(
        self,
        degraded_foreground: Image.Image,
        background: Image.Image,
        mask: Image.Image
    ) -> Image.Image:
        """
        Recombine degraded foreground with original background.

        Args:
            degraded_foreground: Degraded foreground image (PIL)
            background: Original background image (PIL)
            mask: Foreground mask (PIL, grayscale)

        Returns:
            Composite image (PIL)
        """
        # Convert all to numpy arrays
        foreground_np = np.array(degraded_foreground).astype(np.float32)
        background_np = np.array(background).astype(np.float32)
        mask_np = np.array(mask).astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Ensure mask is 2D
        if len(mask_np.shape) == 3:
            mask_np = mask_np[:, :, 0]

        # Expand mask to 3 channels
        mask_3ch = mask_np[:, :, np.newaxis]

        # Composite: foreground * mask + background * (1 - mask)
        composite_np = foreground_np * mask_3ch + background_np * (1 - mask_3ch)

        # Convert back to PIL
        composite = Image.fromarray(composite_np.astype(np.uint8))

        logger.info("Background recombination completed")

        return composite

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate degraded image with altered illumination.

        Randomly selects one degradation type:
        - Soft shading (normal-based)
        - Hard shadow (pattern-based)
        - Specular reflection (shading + highlights)

        Args:
            input_data: Dictionary with 'albedo' (PIL Image) and 'image_id'

        Returns:
            Dictionary with degraded image and metadata
        """
        albedo = input_data['albedo']
        image_id = input_data['image_id']

        logger.info(f"Generating degradation for image {image_id}")

        # Select degradation type
        degradation_type = self._select_degradation_type()
        logger.info(f"Selected degradation type: {degradation_type}")

        # Generate degradation based on type
        try:
            if degradation_type == 'soft_shading':
                degraded, metadata = self._generate_soft_shading(albedo, image_id)

            elif degradation_type == 'hard_shadow':
                degraded, metadata = self._generate_hard_shadow(albedo, image_id)

            elif degradation_type == 'specular':
                degraded, metadata = self._generate_specular(albedo, image_id)

            elif degradation_type == 'advanced_shading':
                degraded, metadata = self._generate_advanced_shading(albedo, image_id)

            else:
                logger.warning(f"Unknown degradation type: {degradation_type}")
                logger.info("Falling back to hard shadow")
                degraded, metadata = self._generate_hard_shadow(albedo, image_id)

        except Exception as e:
            logger.error(f"Degradation generation failed: {e}")
            logger.warning("Using albedo as fallback")
            degraded = albedo
            metadata = {
                'degradation_type': 'none',
                'error': str(e)
            }

        logger.info(f"Degradation generation completed for image {image_id}")

        # Pass through previous data and add degradation (foreground only)
        output = input_data.copy()
        output.update({
            'degraded_image': degraded,  # Degraded foreground only
            'degradation_metadata': metadata,
            'albedo': albedo
        })

        return output

    def unload_model(self):
        """
        Unload MiDaS model.
        """
        if self.normal_estimator is not None:
            self.normal_estimator.unload_model()
            self.normal_estimator = None

        super().unload_model()


# Keep backwards compatibility with old name
ShadowGenerationStage = DegradationSynthesisStage
