"""
Shading synthesis for degradation image generation.

Implements Lambertian and Phong shading models for realistic lighting effects.
"""

import numpy as np
from PIL import Image
import random
import math
import logging
import cv2
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

# Try to import shadow patterns for optional pattern overlay
try:
    from .shadow_patterns import generate_random_shadow_pattern
    PATTERNS_AVAILABLE = True
except ImportError:
    PATTERNS_AVAILABLE = False
    logger.info("shadow_patterns not available, soft shading will work without patterns")


def sample_hemisphere_uniform(
    elevation_range: Tuple[float, float] = (10, 80),
    azimuth_range: Tuple[float, float] = (0, 360)
) -> np.ndarray:
    """
    Sample a random light direction from the upper hemisphere.

    Args:
        elevation_range: Elevation angle range in degrees (0=horizon, 90=zenith)
        azimuth_range: Azimuth angle range in degrees (0-360)

    Returns:
        Unit light direction vector [x, y, z] where z > 0
    """
    # Sample elevation and azimuth
    elevation = random.uniform(elevation_range[0], elevation_range[1])
    azimuth = random.uniform(azimuth_range[0], azimuth_range[1])

    # Convert to radians
    elevation_rad = math.radians(elevation)
    azimuth_rad = math.radians(azimuth)

    # Convert spherical to Cartesian
    # x = cos(elevation) * cos(azimuth)
    # y = cos(elevation) * sin(azimuth)
    # z = sin(elevation)
    x = math.cos(elevation_rad) * math.cos(azimuth_rad)
    y = math.cos(elevation_rad) * math.sin(azimuth_rad)
    z = math.sin(elevation_rad)

    # Return as numpy array
    light_dir = np.array([x, y, z], dtype=np.float32)

    return light_dir


def sample_light_direction_preset() -> np.ndarray:
    """
    Sample from preset light directions (matching paper's approach).

    Returns:
        Unit light direction vector [x, y, z]
    """
    directions = [
        [0.5, 0.5, 1.0],    # Top-right
        [-0.5, 0.5, 1.0],   # Top-left
        [0.5, -0.5, 1.0],   # Bottom-right
        [-0.5, -0.5, 1.0],  # Bottom-left
        [0.0, 0.5, 1.0],    # Top
        [0.5, 0.0, 1.0],    # Right
        [-0.5, 0.0, 1.0],   # Left
        [0.0, -0.5, 1.0],   # Bottom
        [0.0, 0.0, 1.0],    # Center (from above)
    ]

    # Normalize
    direction = random.choice(directions)
    light_dir = np.array(direction, dtype=np.float32)
    light_dir = light_dir / np.linalg.norm(light_dir)

    return light_dir


def compute_lambertian_shading(
    normal_map: np.ndarray,
    light_direction: np.ndarray,
    ambient: float = 0.5  # MUCH higher ambient for lighter shadows
) -> np.ndarray:
    """
    Compute Lambertian (diffuse) shading with VERY LIGHT shadows.

    Lambertian model: I_diffuse = I_light * max(0, N · L)
    where N = surface normal, L = light direction

    Args:
        normal_map: Surface normals (H, W, 3) with values in [-1, 1]
        light_direction: Light direction vector [x, y, z] (unit length)
        ambient: Ambient light ratio (0-1) - higher = lighter shadows

    Returns:
        Shading map (H, W) with values in [0, 1]
    """
    # Ensure light direction is normalized
    light_dir = light_direction / (np.linalg.norm(light_direction) + 1e-8)

    # Reshape for broadcasting: (1, 1, 3)
    light_dir = light_dir.reshape(1, 1, 3)

    # Compute dot product N · L
    # normal_map is (H, W, 3), light_dir is (1, 1, 3)
    dot_product = np.sum(normal_map * light_dir, axis=2)

    # Lambertian: max(0, N · L)
    diffuse = np.maximum(0, dot_product)

    # Add ambient term: ambient + (1 - ambient) * diffuse
    # VERY HIGH AMBIENT = VERY LIGHT SHADOWS
    shading = ambient + (1 - ambient) * diffuse

    # Ensure in [0, 1] range
    shading = np.clip(shading, 0, 1)

    return shading


def compute_phong_specular(
    normal_map: np.ndarray,
    light_direction: np.ndarray,
    view_direction: np.ndarray = np.array([0, 0, 1]),
    shininess: float = 32.0
) -> np.ndarray:
    """
    Compute Phong specular highlights.

    Phong model: I_specular = I_light * (R · V)^n
    where R = 2(N·L)N - L (reflection vector)
          V = view direction
          n = shininess exponent

    Args:
        normal_map: Surface normals (H, W, 3) with values in [-1, 1]
        light_direction: Light direction vector [x, y, z]
        view_direction: View direction vector [x, y, z]
        shininess: Specular exponent (higher = sharper highlights)

    Returns:
        Specular map (H, W) with values in [0, 1]
    """
    # Normalize directions
    light_dir = light_direction / (np.linalg.norm(light_direction) + 1e-8)
    view_dir = view_direction / (np.linalg.norm(view_direction) + 1e-8)

    # Reshape for broadcasting: (1, 1, 3)
    light_dir = light_dir.reshape(1, 1, 3)
    view_dir = view_dir.reshape(1, 1, 3)

    # Compute N · L
    dot_NL = np.sum(normal_map * light_dir, axis=2, keepdims=True)

    # Reflection vector: R = 2(N·L)N - L
    reflection = 2 * dot_NL * normal_map - light_dir

    # Compute R · V
    dot_RV = np.sum(reflection * view_dir, axis=2)

    # Phong specular: (R · V)^shininess
    specular = np.maximum(0, dot_RV) ** shininess

    # Normalize to [0, 1]
    if specular.max() > 0:
        specular = specular / specular.max()

    return specular


def apply_shading_to_albedo(
    albedo: Image.Image,
    shading: np.ndarray,
    blend_mode: str = "multiply"
) -> Image.Image:
    """
    Apply shading to albedo image.

    Args:
        albedo: Albedo image (PIL)
        shading: Shading map (H, W) with values in [0, 1]
        blend_mode: "multiply" or "add"

    Returns:
        Shaded image (PIL)
    """
    # Convert albedo to numpy
    albedo_np = np.array(albedo).astype(np.float32) / 255.0

    # Ensure shading has same shape
    if shading.shape != albedo_np.shape[:2]:
        # Resize shading to match albedo
        import cv2
        shading = cv2.resize(shading, (albedo_np.shape[1], albedo_np.shape[0]))

    # Add channel dimension to shading
    shading_3ch = shading[:, :, np.newaxis]

    # Apply shading
    if blend_mode == "multiply":
        shaded = albedo_np * shading_3ch
    elif blend_mode == "add":
        shaded = albedo_np + shading_3ch
        shaded = np.clip(shaded, 0, 1)
    else:
        raise ValueError(f"Unknown blend mode: {blend_mode}")

    # Convert back to PIL
    shaded = (shaded * 255).astype(np.uint8)
    shaded_pil = Image.fromarray(shaded)

    return shaded_pil


def add_specular_to_image(
    image: Image.Image,
    specular: np.ndarray,
    intensity: float = 0.3
) -> Image.Image:
    """
    Add specular highlights to an image.

    Args:
        image: Base image (PIL)
        specular: Specular map (H, W) with values in [0, 1]
        intensity: Specular intensity multiplier

    Returns:
        Image with specular highlights (PIL)
    """
    # Convert to numpy
    img_np = np.array(image).astype(np.float32) / 255.0

    # Resize specular if needed
    if specular.shape != img_np.shape[:2]:
        import cv2
        specular = cv2.resize(specular, (img_np.shape[1], img_np.shape[0]))

    # Add specular (additive blending)
    specular_3ch = specular[:, :, np.newaxis] * intensity
    result = img_np + specular_3ch

    # Clip to [0, 1]
    result = np.clip(result, 0, 1)

    # Convert back to PIL
    result_pil = Image.fromarray((result * 255).astype(np.uint8))

    return result_pil


def generate_soft_shading_degradation(
    albedo: Image.Image,
    normal_map: np.ndarray,
    light_direction: Optional[np.ndarray] = None,
    ambient: float = 0.2,
    add_specular: bool = False,
    specular_intensity: float = 0.3,
    shininess: float = 32.0
) -> Tuple[Image.Image, dict]:
    """
    Generate soft shading degradation (Method A from paper).

    Creates degraded image with altered illumination using normal-based shading.

    Args:
        albedo: Albedo image (PIL)
        normal_map: Surface normals (H, W, 3)
        light_direction: Light direction vector (if None, random)
        ambient: Ambient light ratio
        add_specular: Whether to add specular highlights
        specular_intensity: Specular highlight intensity
        shininess: Specular shininess exponent

    Returns:
        Tuple of (degraded image, metadata dict)
    """
    logger.info("Generating soft shading degradation")

    # Sample light direction if not provided
    if light_direction is None:
        light_direction = sample_hemisphere_uniform()

    # Compute Lambertian shading
    shading = compute_lambertian_shading(normal_map, light_direction, ambient)

    # Apply shading to albedo
    degraded = apply_shading_to_albedo(albedo, shading, blend_mode="multiply")

    # Add specular if requested
    if add_specular:
        specular = compute_phong_specular(
            normal_map,
            light_direction,
            shininess=shininess
        )
        degraded = add_specular_to_image(degraded, specular, specular_intensity)

    # Create metadata
    metadata = {
        'degradation_type': 'soft_shading',
        'light_direction': light_direction.tolist(),
        'ambient': ambient,
        'has_specular': add_specular,
        'specular_intensity': specular_intensity if add_specular else 0,
        'shininess': shininess if add_specular else 0
    }

    logger.info(f"Soft shading completed with light direction: {light_direction}")

    return degraded, metadata


def generate_random_soft_shading(
    albedo: Image.Image,
    normal_map: np.ndarray,
    config: Optional[dict] = None
) -> Tuple[Image.Image, dict]:
    """
    Generate soft shading with random parameters and optional subtle patterns.

    Args:
        albedo: Albedo image (PIL)
        normal_map: Surface normals (H, W, 3)
        config: Configuration dict with parameter ranges

    Returns:
        Tuple of (degraded image, metadata dict)
    """
    if config is None:
        config = {}

    # Random ambient (VERY HIGH for light shadows)
    ambient_range = config.get('ambient_range', [0.6, 0.85])  # Much higher!
    ambient = random.uniform(*ambient_range)

    # Disable specular by default (too strong)
    add_specular = False
    specular_intensity = 0
    shininess = 32

    # Generate base soft shading
    degraded, metadata = generate_soft_shading_degradation(
        albedo=albedo,
        normal_map=normal_map,
        light_direction=None,  # Random
        ambient=ambient,
        add_specular=add_specular,
        specular_intensity=specular_intensity,
        shininess=shininess
    )

    # Add subtle shadow patterns if enabled
    add_patterns = config.get('add_subtle_patterns', False)
    if add_patterns and PATTERNS_AVAILABLE:
        try:
            h, w = albedo.size[1], albedo.size[0]

            # Generate subtle pattern
            pattern, pattern_meta = generate_random_shadow_pattern((h, w))

            # Apply sharper blur for more visible patterns
            pattern_blur_range = config.get('pattern_blur_range', [21, 51])
            blur_size = random.choice([21, 31, 41, 51])
            if blur_size % 2 == 0:
                blur_size += 1
            pattern = cv2.GaussianBlur(pattern, (blur_size, blur_size), 0)

            # Apply with higher opacity (darker, more visible)
            pattern_opacity_range = config.get('pattern_opacity', [0.35, 0.6])
            pattern_opacity = random.uniform(*pattern_opacity_range)

            # Convert to numpy
            degraded_np = np.array(degraded).astype(np.float32) / 255.0

            # Resize pattern if needed
            if pattern.shape[:2] != (h, w):
                pattern = cv2.resize(pattern, (w, h))

            # Apply pattern: slightly darken where pattern exists
            pattern_3ch = pattern[:, :, np.newaxis]
            degraded_np = degraded_np * (1 - pattern_3ch * pattern_opacity)

            # Convert back
            degraded = Image.fromarray((degraded_np * 255).astype(np.uint8))

            # Update metadata
            metadata['has_pattern'] = True
            metadata['pattern_type'] = pattern_meta.get('pattern_type', 'unknown')
            metadata['pattern_opacity'] = pattern_opacity
            metadata['pattern_blur'] = blur_size

        except Exception as e:
            logger.warning(f"Could not add pattern to soft shading: {e}")
            metadata['has_pattern'] = False
    else:
        metadata['has_pattern'] = False

    return degraded, metadata
