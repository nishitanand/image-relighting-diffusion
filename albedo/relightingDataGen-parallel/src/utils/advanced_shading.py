"""
Advanced shading synthesis with modern lighting techniques.

Implements:
- Ambient Occlusion (SSAO)
- Spherical Harmonics environment lighting
- Multi-light setups
- Physically-based rendering components

This module provides more realistic lighting than basic Lambertian shading.
"""

import numpy as np
from PIL import Image
import random
import logging
from typing import Tuple, Optional, List, Dict

from .ambient_occlusion import compute_ssao
from .environment_lighting import (
    compute_sh_lighting,
    sample_random_environment_sh,
    generate_outdoor_sh_coeffs,
    generate_indoor_sh_coeffs,
    generate_studio_sh_coeffs
)
from .shading_synthesis import (
    sample_hemisphere_uniform,
    compute_lambertian_shading,
    compute_phong_specular
)

logger = logging.getLogger(__name__)


class LightSource:
    """Represents a single light source."""

    def __init__(
        self,
        direction: np.ndarray,
        intensity: float = 1.0,
        color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ):
        """
        Initialize light source.

        Args:
            direction: Light direction vector (will be normalized)
            intensity: Light brightness multiplier
            color: Light color (R, G, B) in [0, 1]
        """
        self.direction = direction / (np.linalg.norm(direction) + 1e-8)
        self.intensity = intensity
        self.color = np.array(color, dtype=np.float32)


def setup_multilight_environment(
    key_direction: Optional[np.ndarray] = None,
    preset: str = 'three_point'
) -> List[LightSource]:
    """
    Setup multi-light environment for realistic portrait lighting.

    Args:
        key_direction: Main light direction (if None, random)
        preset: Lighting preset - 'three_point', 'two_point', 'single'

    Returns:
        List of LightSource objects
    """
    lights = []

    # Sample key light direction if not provided
    if key_direction is None:
        key_direction = sample_hemisphere_uniform(
            elevation_range=(30, 60),
            azimuth_range=(0, 360)
        )

    key_dir = key_direction / (np.linalg.norm(key_direction) + 1e-8)

    if preset == 'three_point':
        # Key light (main directional)
        key = LightSource(
            direction=key_dir,
            intensity=random.uniform(0.9, 1.2),
            color=(1.0, 1.0, 1.0)
        )
        lights.append(key)

        # Fill light (opposite side, lower intensity)
        fill_dir = -key_dir * np.array([1.0, 0.5, 0.5])
        fill_dir = fill_dir / (np.linalg.norm(fill_dir) + 1e-8)
        fill = LightSource(
            direction=fill_dir,
            intensity=random.uniform(0.3, 0.5),
            color=(1.0, 0.98, 0.95)  # Slightly warm
        )
        lights.append(fill)

        # Rim light (from behind/side)
        rim_dir = -key_dir * np.array([0.5, 1.0, -1.0])
        rim_dir = rim_dir / (np.linalg.norm(rim_dir) + 1e-8)
        rim = LightSource(
            direction=rim_dir,
            intensity=random.uniform(0.4, 0.7),
            color=(1.0, 1.0, 1.05)  # Slightly cool
        )
        lights.append(rim)

    elif preset == 'two_point':
        # Key light
        key = LightSource(
            direction=key_dir,
            intensity=random.uniform(0.9, 1.2),
            color=(1.0, 1.0, 1.0)
        )
        lights.append(key)

        # Fill light
        fill_dir = -key_dir * np.array([1.0, 0.6, 0.6])
        fill_dir = fill_dir / (np.linalg.norm(fill_dir) + 1e-8)
        fill = LightSource(
            direction=fill_dir,
            intensity=random.uniform(0.3, 0.6),
            color=(1.0, 0.98, 0.95)
        )
        lights.append(fill)

    elif preset == 'single':
        # Just key light
        key = LightSource(
            direction=key_dir,
            intensity=random.uniform(0.8, 1.2),
            color=(1.0, 1.0, 1.0)
        )
        lights.append(key)

    else:
        raise ValueError(f"Unknown lighting preset: {preset}")

    return lights


def compute_multilight_shading(
    normal_map: np.ndarray,
    lights: List[LightSource]
) -> np.ndarray:
    """
    Compute shading from multiple light sources.

    Args:
        normal_map: Surface normals, shape (H, W, 3)
        lights: List of LightSource objects

    Returns:
        Accumulated shading, shape (H, W, 3) in [0, 1]
    """
    h, w = normal_map.shape[:2]
    total_shading = np.zeros((h, w, 3), dtype=np.float32)

    for light in lights:
        # Compute Lambertian for this light
        light_dir = light.direction.reshape(1, 1, 3)
        dot_NL = np.sum(normal_map * light_dir, axis=2)
        diffuse = np.maximum(0, dot_NL)

        # Add color and intensity
        shading_3ch = diffuse[:, :, np.newaxis] * light.color * light.intensity

        # Accumulate
        total_shading += shading_3ch

    # Clip to valid range
    total_shading = np.clip(total_shading, 0, 1)

    return total_shading


def generate_advanced_shading_degradation(
    albedo: Image.Image,
    normal_map: np.ndarray,
    depth_map: np.ndarray,
    config: Optional[Dict] = None
) -> Tuple[Image.Image, dict]:
    """
    Generate realistic shading degradation with advanced techniques.

    Implements:
    1. Ambient Occlusion for contact shadows
    2. Environment lighting via Spherical Harmonics
    3. Multi-light direct illumination
    4. Optional specular highlights

    Args:
        albedo: Albedo image (PIL)
        normal_map: Surface normals, shape (H, W, 3)
        depth_map: Normalized depth map, shape (H, W)
        config: Configuration dictionary

    Returns:
        Tuple of (degraded image, metadata dict)
    """
    if config is None:
        config = {}

    logger.info("Generating advanced shading degradation")

    # Convert albedo to numpy
    albedo_np = np.array(albedo).astype(np.float32) / 255.0
    h, w = albedo_np.shape[:2]

    # Ensure normal and depth have correct shape
    if normal_map.shape[:2] != (h, w):
        import cv2
        normal_map = cv2.resize(normal_map, (w, h), interpolation=cv2.INTER_LINEAR)
        normal_norms = np.linalg.norm(normal_map, axis=2, keepdims=True)
        normal_map = normal_map / (normal_norms + 1e-8)

    if depth_map.shape != (h, w):
        import cv2
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # === 1. Compute Ambient Occlusion ===
    use_ao = config.get('use_ambient_occlusion', True)
    ao_quality = config.get('ao_quality', 'fast')  # 'fast', 'medium', 'high'

    if use_ao:
        logger.info(f"Computing ambient occlusion (quality: {ao_quality})")
        ao_map = compute_ssao(depth_map, normal_map, quality=ao_quality)
    else:
        ao_map = np.ones((h, w), dtype=np.float32)

    # === 2. Setup Environment Lighting ===
    env_type = config.get('environment_type', 'outdoor')  # 'outdoor', 'indoor', 'studio'
    use_env_lighting = config.get('use_environment_lighting', True)

    # Sample light direction for both environment and direct lighting
    light_direction = sample_hemisphere_uniform(
        elevation_range=config.get('elevation_range', (20, 70)),
        azimuth_range=config.get('azimuth_range', (0, 360))
    )

    if use_env_lighting:
        logger.info(f"Computing environment lighting ({env_type})")
        sh_coeffs = sample_random_environment_sh(env_type, light_direction)
        env_lighting = compute_sh_lighting(normal_map, sh_coeffs)
    else:
        # Fallback to simple ambient
        ambient = config.get('ambient', 0.2)
        env_lighting = np.ones((h, w, 3), dtype=np.float32) * ambient

    # === 3. Compute Direct Lighting ===
    use_multilight = config.get('use_multilight', True)
    lighting_preset = config.get('lighting_preset', 'three_point')

    if use_multilight:
        logger.info(f"Computing multi-light shading ({lighting_preset})")
        lights = setup_multilight_environment(light_direction, lighting_preset)
        direct_lighting = compute_multilight_shading(normal_map, lights)
    else:
        # Single light Lambertian
        logger.info("Computing single-light Lambertian shading")
        light_dir = light_direction.reshape(1, 1, 3)
        dot_NL = np.sum(normal_map * light_dir, axis=2)
        diffuse = np.maximum(0, dot_NL)
        direct_lighting = diffuse[:, :, np.newaxis]

    # === 4. Combine Lighting Components ===
    # Scale environment vs direct lighting
    env_strength = config.get('env_strength', 0.3)
    direct_strength = config.get('direct_strength', 0.7)

    total_lighting = (
        env_lighting * env_strength +
        direct_lighting * direct_strength
    )

    # Apply ambient occlusion (multiply) - LIGHTER for relighting
    ao_strength = config.get('ao_strength', 0.4)  # Reduced from 0.7 to 0.4
    ao_3ch = ao_map[:, :, np.newaxis]
    ao_3ch = ao_strength * ao_3ch + (1 - ao_strength)  # Blend AO strength
    total_lighting = total_lighting * ao_3ch

    # === 5. Apply to Albedo ===
    shaded = albedo_np * total_lighting
    shaded = np.clip(shaded, 0, 1)

    # === 6. Add Specular Highlights (Optional) ===
    add_specular = config.get('add_specular', False)
    if add_specular:
        logger.info("Adding specular highlights")
        shininess = random.uniform(
            *config.get('shininess_range', [10, 100])
        )
        specular_intensity = random.uniform(
            *config.get('specular_intensity_range', [0.1, 0.4])
        )

        # Use key light direction for specular
        specular = compute_phong_specular(
            normal_map,
            light_direction,
            shininess=shininess
        )

        # Add specular (additive)
        specular_3ch = specular[:, :, np.newaxis] * specular_intensity
        shaded = shaded + specular_3ch
        shaded = np.clip(shaded, 0, 1)
    else:
        shininess = 0
        specular_intensity = 0

    # === 7. Convert to PIL ===
    degraded = Image.fromarray((shaded * 255).astype(np.uint8))

    # === 8. Create Metadata ===
    metadata = {
        'degradation_type': 'advanced_shading',
        'light_direction': light_direction.tolist(),
        'environment_type': env_type,
        'lighting_preset': lighting_preset if use_multilight else 'single',
        'use_ambient_occlusion': use_ao,
        'ao_quality': ao_quality if use_ao else None,
        'use_environment_lighting': use_env_lighting,
        'use_multilight': use_multilight,
        'env_strength': env_strength,
        'direct_strength': direct_strength,
        'ao_strength': ao_strength,
        'add_specular': add_specular,
        'specular_intensity': specular_intensity if add_specular else 0,
        'shininess': shininess if add_specular else 0
    }

    logger.info("Advanced shading completed")

    return degraded, metadata


def generate_random_advanced_shading(
    albedo: Image.Image,
    normal_map: np.ndarray,
    depth_map: np.ndarray
) -> Tuple[Image.Image, dict]:
    """
    Generate advanced shading with random parameters.

    This is the recommended function for data generation - it randomly
    samples realistic lighting configurations.

    Args:
        albedo: Albedo image (PIL)
        normal_map: Surface normals, shape (H, W, 3)
        depth_map: Normalized depth map, shape (H, W)

    Returns:
        Tuple of (degraded image, metadata dict)
    """
    # Random environment type
    env_type = random.choice(['outdoor', 'indoor', 'studio'])

    # Random lighting preset based on environment
    if env_type == 'studio':
        preset = random.choice(['three_point', 'two_point'])
    elif env_type == 'outdoor':
        preset = 'single'  # Sun is dominant
    else:  # indoor
        preset = random.choice(['single', 'two_point'])

    # Random AO quality (favor fast for speed)
    ao_quality = random.choices(
        ['fast', 'medium', 'high'],
        weights=[0.7, 0.25, 0.05]
    )[0]

    # Random lighting balance
    env_strength = random.uniform(0.2, 0.5)
    direct_strength = random.uniform(0.5, 1.0)

    # Normalize so they sum to ~1.0
    total = env_strength + direct_strength
    env_strength = env_strength / total
    direct_strength = direct_strength / total

    # Random AO strength (LIGHTER for relighting)
    ao_strength = random.uniform(0.2, 0.5)  # Reduced from 0.5-0.9 to 0.2-0.5

    # Random specular (20% chance)
    add_specular = random.random() < 0.2

    config = {
        'environment_type': env_type,
        'lighting_preset': preset,
        'use_ambient_occlusion': True,
        'ao_quality': ao_quality,
        'use_environment_lighting': True,
        'use_multilight': preset != 'single',
        'env_strength': env_strength,
        'direct_strength': direct_strength,
        'ao_strength': ao_strength,
        'add_specular': add_specular,
        'shininess_range': [10, 100],
        'specular_intensity_range': [0.1, 0.4]
    }

    return generate_advanced_shading_degradation(
        albedo,
        normal_map,
        depth_map,
        config
    )
