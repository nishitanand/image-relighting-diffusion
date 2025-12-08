"""
Shadow generation for hard shadow synthesis.

Generates procedural shadow patterns without requiring purchased shadow materials.
Implements geometric and noise-based shadow patterns.
"""

import numpy as np
from PIL import Image
import random
import cv2
import logging
from typing import Tuple, Optional
import math
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)

# Try to import shadow patterns, fallback if not available
try:
    from .shadow_patterns import generate_random_shadow_pattern
    PATTERNS_AVAILABLE = True
except ImportError:
    PATTERNS_AVAILABLE = False
    logger.warning("shadow_patterns module not available, using basic procedural shadows only")


def generate_perlin_noise_2d(shape: Tuple[int, int], scale: int = 10) -> np.ndarray:
    """
    Generate 2D Perlin-like noise using interpolated random gradients.

    Simplified Perlin noise implementation for organic shadow patterns.

    Args:
        shape: Output shape (H, W)
        scale: Noise scale (smaller = smoother)

    Returns:
        Noise array (H, W) with values in [0, 1]
    """
    h, w = shape

    # Generate random gradients at grid points
    grid_h = h // scale + 1
    grid_w = w // scale + 1

    # Random gradients
    gradients = np.random.randn(grid_h, grid_w, 2)

    # Interpolate
    noise = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            # Grid cell
            cell_i = i // scale
            cell_j = j // scale

            # Position within cell
            local_i = (i % scale) / scale
            local_j = (j % scale) / scale

            # Bilinear interpolation of gradients
            if cell_i < grid_h - 1 and cell_j < grid_w - 1:
                # Get 4 corner gradients
                g00 = gradients[cell_i, cell_j]
                g01 = gradients[cell_i, cell_j + 1]
                g10 = gradients[cell_i + 1, cell_j]
                g11 = gradients[cell_i + 1, cell_j + 1]

                # Distance vectors
                d00 = np.array([local_i, local_j])
                d01 = np.array([local_i, local_j - 1])
                d10 = np.array([local_i - 1, local_j])
                d11 = np.array([local_i - 1, local_j - 1])

                # Dot products
                n00 = np.dot(g00, d00)
                n01 = np.dot(g01, d01)
                n10 = np.dot(g10, d10)
                n11 = np.dot(g11, d11)

                # Interpolate
                nx0 = n00 * (1 - local_j) + n01 * local_j
                nx1 = n10 * (1 - local_j) + n11 * local_j
                noise[i, j] = nx0 * (1 - local_i) + nx1 * local_i

    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

    return noise


def generate_geometric_shadow(
    shape: Tuple[int, int],
    shadow_type: str = "rectangle"
) -> np.ndarray:
    """
    Generate geometric shadow pattern.

    Args:
        shape: Output shape (H, W)
        shadow_type: "rectangle", "ellipse", "triangle", or "stripe"

    Returns:
        Shadow mask (H, W) with values in [0, 1]
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)

    if shadow_type == "rectangle":
        # Random rectangle
        x1 = random.randint(0, w // 2)
        y1 = random.randint(0, h // 2)
        x2 = random.randint(w // 2, w)
        y2 = random.randint(h // 2, h)
        mask[y1:y2, x1:x2] = 1.0

    elif shadow_type == "ellipse":
        # Random ellipse
        center_x = random.randint(w // 4, 3 * w // 4)
        center_y = random.randint(h // 4, 3 * h // 4)
        radius_x = random.randint(w // 8, w // 3)
        radius_y = random.randint(h // 8, h // 3)
        cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 1.0, -1)

    elif shadow_type == "triangle":
        # Random triangle
        pts = np.array([
            [random.randint(0, w), random.randint(0, h)],
            [random.randint(0, w), random.randint(0, h)],
            [random.randint(0, w), random.randint(0, h)]
        ], np.int32)
        cv2.fillPoly(mask, [pts], 1.0)

    elif shadow_type == "stripe":
        # Vertical or horizontal stripes
        if random.random() < 0.5:
            # Vertical
            num_stripes = random.randint(3, 10)
            stripe_width = w // num_stripes
            for i in range(0, num_stripes, 2):
                x1 = i * stripe_width
                x2 = min((i + 1) * stripe_width, w)
                mask[:, x1:x2] = 1.0
        else:
            # Horizontal
            num_stripes = random.randint(3, 10)
            stripe_width = h // num_stripes
            for i in range(0, num_stripes, 2):
                y1 = i * stripe_width
                y2 = min((i + 1) * stripe_width, h)
                mask[y1:y2, :] = 1.0

    return mask


def generate_blob_shadow(shape: Tuple[int, int], num_blobs: int = 5) -> np.ndarray:
    """
    Generate organic blob-like shadows.

    Args:
        shape: Output shape (H, W)
        num_blobs: Number of blob components

    Returns:
        Shadow mask (H, W) with values in [0, 1]
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)

    for _ in range(num_blobs):
        # Random blob center and size
        center_x = random.randint(0, w)
        center_y = random.randint(0, h)
        radius_x = random.randint(w // 10, w // 4)
        radius_y = random.randint(h // 10, h // 4)

        # Create blob with soft edges
        y, x = np.ogrid[:h, :w]
        dist = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2
        blob = np.exp(-dist)

        mask = np.maximum(mask, blob)

    return mask


def apply_shadow_transform(
    shadow_mask: np.ndarray,
    rotate: bool = True,
    scale: bool = True,
    skew: bool = False
) -> np.ndarray:
    """
    Apply random transformations to shadow mask.

    Args:
        shadow_mask: Shadow mask (H, W)
        rotate: Whether to apply random rotation
        scale: Whether to apply random scaling
        skew: Whether to apply random skew

    Returns:
        Transformed shadow mask
    """
    h, w = shadow_mask.shape

    # Create transformation matrix
    center = (w // 2, h // 2)
    M = np.eye(2, 3, dtype=np.float32)

    if rotate:
        angle = random.uniform(-30, 30)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        M = M_rot

    if scale:
        scale_x = random.uniform(0.7, 1.3)
        scale_y = random.uniform(0.7, 1.3)
        M_scale = np.array([[scale_x, 0, 0], [0, scale_y, 0]], dtype=np.float32)
        # Combine affine transformations: M_combined = M @ M_scale_3x3
        # Convert both to 3x3, multiply, then take top 2 rows
        M_3x3 = np.vstack([M, [0, 0, 1]])
        M_scale_3x3 = np.vstack([M_scale, [0, 0, 1]])
        M = (M_3x3 @ M_scale_3x3)[:2, :]

    # Apply transformation
    transformed = cv2.warpAffine(shadow_mask, M, (w, h), flags=cv2.INTER_LINEAR)

    return transformed


def soften_shadow_edges(shadow_mask: np.ndarray, blur_size: int = 15) -> np.ndarray:
    """
    Soften shadow edges with Gaussian blur.

    Args:
        shadow_mask: Shadow mask (H, W)
        blur_size: Gaussian blur kernel size

    Returns:
        Softened shadow mask
    """
    # Apply Gaussian blur
    softened = cv2.GaussianBlur(shadow_mask, (blur_size, blur_size), 0)

    return softened


def generate_random_shadow_pattern(
    shape: Tuple[int, int],
    pattern_type: Optional[str] = None,
    soften: bool = True
) -> np.ndarray:
    """
    Generate random shadow pattern.

    Args:
        shape: Output shape (H, W)
        pattern_type: Shadow type (if None, random selection)
        soften: Whether to soften edges

    Returns:
        Shadow mask (H, W) with values in [0, 1]
    """
    # Select pattern type
    if pattern_type is None:
        pattern_type = random.choice(["geometric", "blob", "noise"])

    # Generate base pattern
    if pattern_type == "geometric":
        geo_type = random.choice(["rectangle", "ellipse", "triangle", "stripe"])
        mask = generate_geometric_shadow(shape, geo_type)

    elif pattern_type == "blob":
        num_blobs = random.randint(2, 5)
        mask = generate_blob_shadow(shape, num_blobs)

    elif pattern_type == "noise":
        scale = random.randint(5, 20)
        mask = generate_perlin_noise_2d(shape, scale)

        # Threshold to create sharper shadows
        threshold = random.uniform(0.4, 0.6)
        mask = (mask > threshold).astype(np.float32)

    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    # Apply transformations
    mask = apply_shadow_transform(mask, rotate=True, scale=True)

    # Soften edges (much softer for realistic relighting)
    if soften:
        # Use larger blur sizes for soft, realistic shadows
        blur_size = random.choice([31, 51, 71, 91, 111])
        mask = soften_shadow_edges(mask, blur_size)

    return mask


def apply_shadow_to_image(
    image: Image.Image,
    shadow_mask: np.ndarray,
    opacity: float = 0.5
) -> Image.Image:
    """
    Apply shadow mask to image.

    Args:
        image: Input image (PIL)
        shadow_mask: Shadow mask (H, W) with values in [0, 1]
        opacity: Shadow opacity (0=transparent, 1=black)

    Returns:
        Shadowed image (PIL)
    """
    # Convert to numpy
    img_np = np.array(image).astype(np.float32) / 255.0

    # Resize mask if needed
    if shadow_mask.shape != img_np.shape[:2]:
        shadow_mask = cv2.resize(shadow_mask, (img_np.shape[1], img_np.shape[0]))

    # Apply shadow: img * (1 - mask * opacity)
    shadow_3ch = shadow_mask[:, :, np.newaxis] * opacity
    shadowed = img_np * (1 - shadow_3ch)

    # Convert back to PIL
    shadowed = (shadowed * 255).astype(np.uint8)
    shadowed_pil = Image.fromarray(shadowed)

    return shadowed_pil


def generate_hard_shadow_degradation(
    albedo: Image.Image,
    shadow_pattern: Optional[np.ndarray] = None,
    opacity: Optional[float] = None,
    pattern_type: Optional[str] = None
) -> Tuple[Image.Image, dict]:
    """
    Generate hard shadow degradation (Method B from paper).

    Creates degraded image with hard shadow patterns.

    Args:
        albedo: Albedo image (PIL)
        shadow_pattern: Pre-generated shadow mask (if None, generate random)
        opacity: Shadow opacity (if None, random in [0.3, 0.8])
        pattern_type: Shadow pattern type (if None, random)

    Returns:
        Tuple of (degraded image, metadata dict)
    """
    logger.info("Generating hard shadow degradation")

    # Get image shape
    h, w = albedo.size[1], albedo.size[0]

    # Generate shadow pattern if not provided
    if shadow_pattern is None:
        shadow_pattern = generate_random_shadow_pattern((h, w), pattern_type)
        pattern_generated = True
    else:
        pattern_generated = False

    # Sample opacity if not provided (softer for relighting)
    if opacity is None:
        opacity = random.uniform(0.2, 0.5)

    # Apply shadow to albedo
    degraded = apply_shadow_to_image(albedo, shadow_pattern, opacity)

    # Create metadata
    metadata = {
        'degradation_type': 'hard_shadow',
        'opacity': opacity,
        'pattern_type': pattern_type,
        'pattern_generated': pattern_generated
    }

    logger.info(f"Hard shadow completed with opacity: {opacity}")

    return degraded, metadata


def generate_random_hard_shadow(
    albedo: Image.Image,
    config: Optional[dict] = None
) -> Tuple[Image.Image, dict]:
    """
    Generate hard shadow with random parameters.

    Uses realistic shadow patterns (venetian blinds, trees, windows) when available,
    falls back to procedural patterns otherwise.

    Args:
        albedo: Albedo image (PIL)
        config: Configuration dict with parameter ranges

    Returns:
        Tuple of (degraded image, metadata dict)
    """
    if config is None:
        config = {}

    # Random opacity (LIGHTER for relighting)
    opacity_range = config.get('opacity_range', [0.1, 0.3])  # Reduced from [0.2, 0.5]
    opacity = random.uniform(*opacity_range)

    # Get image shape
    h, w = albedo.size[1], albedo.size[0]

    # Use realistic patterns if available and enabled
    use_patterns = config.get('use_patterns', True)

    if PATTERNS_AVAILABLE and use_patterns:
        # Generate realistic shadow pattern
        shadow_pattern, pattern_metadata = generate_random_shadow_pattern((h, w))

        # Apply blur for softness
        blur_range = config.get('blur_range', [31, 111])
        blur_size = random.choice([31, 51, 71, 91, 111])  # From blur_range
        if blur_size % 2 == 0:
            blur_size += 1
        shadow_pattern = cv2.GaussianBlur(shadow_pattern, (blur_size, blur_size), 0)

        # Apply to image
        degraded = apply_shadow_to_image(albedo, shadow_pattern, opacity)

        # Create metadata
        metadata = {
            'degradation_type': 'pattern_shadow',
            'opacity': opacity,
            'blur_size': blur_size,
            **pattern_metadata
        }

        logger.info(f"Realistic pattern shadow completed: {pattern_metadata['pattern_type']}")
        return degraded, metadata
    else:
        # Fallback to basic procedural
        logger.info("Using basic procedural shadow (realistic patterns not available)")
        return generate_hard_shadow_degradation(
            albedo=albedo,
            shadow_pattern=None,  # Random
            opacity=opacity,
            pattern_type=None  # Random
        )


def generate_depth_aware_shadow(
    depth_map: np.ndarray,
    normal_map: np.ndarray,
    light_direction: np.ndarray,
    shadow_softness: float = 0.5
) -> np.ndarray:
    """
    Generate 3D-aware shadow mask using depth and normal maps.

    Uses depth discontinuities and surface orientation to cast realistic shadows
    based on light direction.

    Args:
        depth_map: Depth map (H, W) with values in [0, 1], closer = smaller
        normal_map: Surface normals (H, W, 3) with values in [-1, 1]
        light_direction: Light direction vector [x, y, z] (unit length)
        shadow_softness: Shadow edge softness (0-1)

    Returns:
        Shadow mask (H, W) with values in [0, 1], where 1 = full shadow
    """
    h, w = depth_map.shape

    # Normalize light direction
    light_dir = light_direction / (np.linalg.norm(light_direction) + 1e-8)
    light_dir = light_dir.reshape(1, 1, 3)

    # 1. Compute self-shadowing from surface orientation
    # Surfaces facing away from light should be shadowed
    dot_product = np.sum(normal_map * light_dir, axis=2)
    facing_away = np.clip(-dot_product, 0, 1)  # 0 = facing light, 1 = facing away

    # 2. Compute cast shadows from depth discontinuities
    # Find depth edges (object boundaries)
    depth_grad_x = np.abs(np.gradient(depth_map, axis=1))
    depth_grad_y = np.abs(np.gradient(depth_map, axis=0))
    depth_edges = np.sqrt(depth_grad_x**2 + depth_grad_y**2)
    depth_edges = np.clip(depth_edges * 10, 0, 1)  # Enhance edges

    # 3. Cast shadows in light direction
    # Project light direction onto image plane
    light_x, light_y = light_dir[0, 0, 0], light_dir[0, 0, 1]

    # Create shadow caster mask from depth edges
    occluder_mask = depth_edges > 0.2

    # Dilate occluders in direction opposite to light
    cast_distance = int(30 * shadow_softness)  # Shadow length
    angle = np.arctan2(-light_y, -light_x)  # Opposite direction

    # Create directional shadow kernel
    kernel_size = cast_distance * 2 + 1
    kernel = np.zeros((kernel_size, kernel_size))
    center = cast_distance

    # Draw line from center in shadow direction
    for i in range(cast_distance):
        offset_x = int(i * np.cos(angle))
        offset_y = int(i * np.sin(angle))
        if abs(offset_x) < cast_distance and abs(offset_y) < cast_distance:
            kernel[center + offset_y, center + offset_x] = 1.0 - (i / cast_distance)

    # Apply directional dilation
    cast_shadow = cv2.filter2D(occluder_mask.astype(np.float32), -1, kernel)
    cast_shadow = np.clip(cast_shadow, 0, 1)

    # 4. Combine self-shadowing and cast shadows
    shadow_mask = np.maximum(facing_away * 0.6, cast_shadow * 0.8)

    # 5. Apply depth-based attenuation (closer objects cast darker shadows)
    depth_factor = 1.0 - depth_map * 0.3  # Closer = darker
    shadow_mask = shadow_mask * depth_factor

    # 6. Soften shadow edges
    blur_size = int(15 + shadow_softness * 50)
    if blur_size % 2 == 0:
        blur_size += 1
    shadow_mask = cv2.GaussianBlur(shadow_mask, (blur_size, blur_size), 0)

    # Ensure in [0, 1] range
    shadow_mask = np.clip(shadow_mask, 0, 1)

    return shadow_mask


def generate_normal_aware_shadow_degradation(
    albedo: Image.Image,
    normal_map: np.ndarray,
    depth_map: Optional[np.ndarray] = None,
    light_direction: Optional[np.ndarray] = None,
    opacity: Optional[float] = None,
    shadow_softness: float = 0.5
) -> Tuple[Image.Image, dict]:
    """
    Generate 3D-aware shadow using normal and depth maps.

    More realistic than procedural shadows - uses surface geometry to cast shadows.

    Args:
        albedo: Albedo image (PIL)
        normal_map: Surface normals (H, W, 3)
        depth_map: Depth map (H, W) - if None, derived from normals
        light_direction: Light direction (if None, random)
        opacity: Shadow opacity (if None, random 0.2-0.5)
        shadow_softness: Shadow edge softness (0-1)

    Returns:
        Tuple of (degraded image, metadata)
    """
    logger.info("Generating 3D-aware shadow from normal/depth maps")

    # Get image dimensions
    h, w = normal_map.shape[:2]

    # Sample light direction if not provided
    if light_direction is None:
        # Sample from hemisphere
        elevation = random.uniform(20, 70)  # degrees
        azimuth = random.uniform(0, 360)
        elevation_rad = math.radians(elevation)
        azimuth_rad = math.radians(azimuth)

        light_direction = np.array([
            math.cos(elevation_rad) * math.cos(azimuth_rad),
            math.cos(elevation_rad) * math.sin(azimuth_rad),
            math.sin(elevation_rad)
        ], dtype=np.float32)

    # Create simple depth map from normals if not provided
    if depth_map is None:
        # Integrate normals to get rough depth
        # This is a simplification - proper depth integration is complex
        depth_map = np.ones((h, w), dtype=np.float32) * 0.5

        # Add some depth variation based on normal z-component
        z_component = normal_map[:, :, 2]
        depth_map = depth_map + z_component * 0.2
        depth_map = np.clip(depth_map, 0, 1)

    # Generate 3D-aware shadow mask
    shadow_mask = generate_depth_aware_shadow(
        depth_map=depth_map,
        normal_map=normal_map,
        light_direction=light_direction,
        shadow_softness=shadow_softness
    )

    # Sample opacity if not provided (LIGHTER for relighting)
    if opacity is None:
        opacity = random.uniform(0.1, 0.3)  # Reduced from 0.2-0.5

    # Apply shadow to albedo
    degraded = apply_shadow_to_image(albedo, shadow_mask, opacity)

    # Create metadata
    metadata = {
        'degradation_type': 'depth_aware_shadow',
        'light_direction': light_direction.tolist(),
        'opacity': opacity,
        'shadow_softness': shadow_softness,
        'uses_3d_geometry': True
    }

    logger.info(f"3D-aware shadow completed with light direction: {light_direction}")

    return degraded, metadata
