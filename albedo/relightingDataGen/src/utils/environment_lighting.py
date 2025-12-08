"""
Environment lighting using Spherical Harmonics.

Provides natural ambient illumination from sky dome and environment maps.
Based on Ravi Ramamoorthi's work on Spherical Harmonic lighting.
"""

import numpy as np
from typing import Optional, Tuple


def compute_sh_basis(normal_map: np.ndarray) -> np.ndarray:
    """
    Compute Spherical Harmonics basis functions for surface normals.

    Uses first 3 bands (9 coefficients) which is sufficient for diffuse lighting.

    Args:
        normal_map: Surface normals, shape (H, W, 3), normalized to unit length

    Returns:
        SH basis values, shape (H, W, 9)
    """
    h, w = normal_map.shape[:2]

    # Reshape for batch processing
    n = normal_map.reshape(-1, 3)
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]

    # Initialize basis functions
    Y = np.zeros((n.shape[0], 9), dtype=np.float32)

    # Band 0 (l=0, constant)
    Y[:, 0] = 0.282095  # Y(0,0)

    # Band 1 (l=1, linear)
    Y[:, 1] = 0.488603 * ny      # Y(1,-1)
    Y[:, 2] = 0.488603 * nz      # Y(1,0)
    Y[:, 3] = 0.488603 * nx      # Y(1,1)

    # Band 2 (l=2, quadratic)
    Y[:, 4] = 1.092548 * nx * ny              # Y(2,-2)
    Y[:, 5] = 1.092548 * ny * nz              # Y(2,-1)
    Y[:, 6] = 0.315392 * (3 * nz**2 - 1)      # Y(2,0)
    Y[:, 7] = 1.092548 * nx * nz              # Y(2,1)
    Y[:, 8] = 0.546274 * (nx**2 - ny**2)      # Y(2,2)

    return Y.reshape(h, w, 9)


def compute_sh_lighting(
    normal_map: np.ndarray,
    sh_coeffs: np.ndarray
) -> np.ndarray:
    """
    Compute environment lighting using Spherical Harmonics.

    Args:
        normal_map: Surface normals, shape (H, W, 3)
        sh_coeffs: SH coefficients, shape (9, 3) for RGB

    Returns:
        Environment lighting contribution, shape (H, W, 3)
    """
    h, w = normal_map.shape[:2]

    # Compute SH basis
    Y = compute_sh_basis(normal_map)  # (H, W, 9)

    # Reshape for matrix multiplication
    Y_flat = Y.reshape(-1, 9)  # (H*W, 9)

    # Compute lighting: Y @ coeffs
    lighting = Y_flat @ sh_coeffs  # (H*W, 9) @ (9, 3) = (H*W, 3)

    # Reshape back to image
    lighting = lighting.reshape(h, w, 3)

    # Ensure non-negative (clamp to [0, inf])
    lighting = np.maximum(lighting, 0.0)

    return lighting


def generate_outdoor_sh_coeffs(
    sun_direction: Optional[np.ndarray] = None,
    sun_intensity: float = 1.0,
    sky_color: Tuple[float, float, float] = (0.5, 0.7, 1.0),
    ground_color: Tuple[float, float, float] = (0.3, 0.25, 0.2)
) -> np.ndarray:
    """
    Generate SH coefficients for outdoor lighting environment.

    Simulates:
    - Sky dome (blue ambient from above)
    - Ground reflection (brown/green from below)
    - Directional sun (if provided)

    Args:
        sun_direction: Optional sun direction vector (will be normalized)
        sun_intensity: Sun brightness multiplier
        sky_color: Sky color (R, G, B)
        ground_color: Ground reflection color (R, G, B)

    Returns:
        SH coefficients, shape (9, 3)
    """
    sh = np.zeros((9, 3), dtype=np.float32)

    # Band 0: Ambient (average environment color)
    ambient = np.array(sky_color) * 0.6 + np.array(ground_color) * 0.4
    sh[0] = ambient * 0.8  # Y(0,0) - constant term

    # Band 1: Directional components
    # Y(1,-1) corresponds to +Y (up/down)
    # Y(1,0) corresponds to +Z (forward/back)
    # Y(1,1) corresponds to +X (right/left)

    # Sky from above (+Y direction)
    sh[1] = np.array(sky_color) * 0.4  # +Y component

    # Ground from below (negative in Y)
    # (Note: -Y contribution goes into positive Y(1,-1) with negative weight)

    # If sun direction provided, add directional component
    if sun_direction is not None:
        sun_dir = sun_direction / (np.linalg.norm(sun_direction) + 1e-8)
        sun_color = np.array([1.0, 0.98, 0.95]) * sun_intensity  # Warm sun

        # Project sun direction into SH bands
        sh[1] += 0.488603 * sun_dir[1] * sun_color  # Y component
        sh[2] += 0.488603 * sun_dir[2] * sun_color  # Z component
        sh[3] += 0.488603 * sun_dir[0] * sun_color  # X component

    # Band 2: Subtle variation for sky gradient
    sh[6] = np.array(sky_color) * 0.1  # Y(2,0) - zenith/nadir gradient

    return sh


def generate_indoor_sh_coeffs(
    light_direction: Optional[np.ndarray] = None,
    light_intensity: float = 0.8,
    ambient_color: Tuple[float, float, float] = (1.0, 0.95, 0.9)
) -> np.ndarray:
    """
    Generate SH coefficients for indoor lighting environment.

    Simulates:
    - Uniform ambient from ceiling lights
    - Optional directional key light

    Args:
        light_direction: Optional key light direction vector
        light_intensity: Key light brightness
        ambient_color: Ambient light color (typically warm white)

    Returns:
        SH coefficients, shape (9, 3)
    """
    sh = np.zeros((9, 3), dtype=np.float32)

    # Band 0: Strong uniform ambient
    sh[0] = np.array(ambient_color) * 0.6

    # Band 1: Key light if provided
    if light_direction is not None:
        light_dir = light_direction / (np.linalg.norm(light_direction) + 1e-8)
        light_color = np.array(ambient_color) * light_intensity

        sh[1] += 0.488603 * light_dir[1] * light_color
        sh[2] += 0.488603 * light_dir[2] * light_color
        sh[3] += 0.488603 * light_dir[0] * light_color

    # Subtle top-down gradient (ceiling lights)
    sh[1] += np.array(ambient_color) * 0.2

    return sh


def generate_studio_sh_coeffs(
    key_direction: np.ndarray,
    key_intensity: float = 1.0,
    fill_ratio: float = 0.3,
    rim_ratio: float = 0.5
) -> np.ndarray:
    """
    Generate SH coefficients for studio 3-point lighting setup.

    Args:
        key_direction: Main light direction (will be normalized)
        key_intensity: Key light brightness
        fill_ratio: Fill light intensity relative to key (typically 0.3-0.5)
        rim_ratio: Rim light intensity relative to key (typically 0.5-0.8)

    Returns:
        SH coefficients, shape (9, 3)
    """
    sh = np.zeros((9, 3), dtype=np.float32)

    # Normalize key direction
    key_dir = key_direction / (np.linalg.norm(key_direction) + 1e-8)

    # Key light (main directional)
    key_color = np.array([1.0, 1.0, 1.0]) * key_intensity
    sh[1] += 0.488603 * key_dir[1] * key_color
    sh[2] += 0.488603 * key_dir[2] * key_color
    sh[3] += 0.488603 * key_dir[0] * key_color

    # Fill light (opposite side, from below/front)
    fill_dir = -key_dir * np.array([1.0, 0.5, 0.5])  # Opposite X, lower Y, same Z
    fill_dir = fill_dir / (np.linalg.norm(fill_dir) + 1e-8)
    fill_color = np.array([1.0, 0.98, 0.95]) * key_intensity * fill_ratio
    sh[1] += 0.488603 * fill_dir[1] * fill_color
    sh[2] += 0.488603 * fill_dir[2] * fill_color
    sh[3] += 0.488603 * fill_dir[0] * fill_color

    # Rim light (from behind)
    rim_dir = -key_dir * np.array([0.5, 1.0, -1.0])  # Behind and above
    rim_dir = rim_dir / (np.linalg.norm(rim_dir) + 1e-8)
    rim_color = np.array([1.0, 1.0, 1.05]) * key_intensity * rim_ratio  # Slightly cool
    sh[1] += 0.488603 * rim_dir[1] * rim_color
    sh[2] += 0.488603 * rim_dir[2] * rim_color
    sh[3] += 0.488603 * rim_dir[0] * rim_color

    # Soft ambient base
    sh[0] = np.array([0.2, 0.2, 0.25])

    return sh


def sample_random_environment_sh(
    environment_type: str = 'outdoor',
    light_direction: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Sample random SH coefficients for diverse environments.

    Args:
        environment_type: 'outdoor', 'indoor', or 'studio'
        light_direction: Optional directional light (used for all types)

    Returns:
        SH coefficients, shape (9, 3)
    """
    if environment_type == 'outdoor':
        # Random sky and ground colors
        sky_color = (
            np.random.uniform(0.4, 0.7),   # R
            np.random.uniform(0.6, 0.9),   # G
            np.random.uniform(0.8, 1.0)    # B (sky is bluish)
        )
        ground_color = (
            np.random.uniform(0.2, 0.4),   # R
            np.random.uniform(0.2, 0.35),  # G
            np.random.uniform(0.15, 0.3)   # B
        )
        sun_intensity = np.random.uniform(0.8, 1.5)

        return generate_outdoor_sh_coeffs(
            sun_direction=light_direction,
            sun_intensity=sun_intensity,
            sky_color=sky_color,
            ground_color=ground_color
        )

    elif environment_type == 'indoor':
        # Random warm white ambient
        warmth = np.random.uniform(0.85, 1.0)
        ambient_color = (
            1.0,
            np.random.uniform(0.9, 0.98),
            warmth
        )
        light_intensity = np.random.uniform(0.6, 1.0)

        return generate_indoor_sh_coeffs(
            light_direction=light_direction,
            light_intensity=light_intensity,
            ambient_color=ambient_color
        )

    elif environment_type == 'studio':
        if light_direction is None:
            # Default studio key light position
            light_direction = np.array([0.5, 0.7, 0.5])

        key_intensity = np.random.uniform(0.8, 1.2)
        fill_ratio = np.random.uniform(0.2, 0.5)
        rim_ratio = np.random.uniform(0.4, 0.8)

        return generate_studio_sh_coeffs(
            key_direction=light_direction,
            key_intensity=key_intensity,
            fill_ratio=fill_ratio,
            rim_ratio=rim_ratio
        )

    else:
        raise ValueError(f"Unknown environment type: {environment_type}")


def visualize_sh_environment(sh_coeffs: np.ndarray, size: int = 256) -> np.ndarray:
    """
    Visualize SH environment as a normal map sphere.

    Useful for debugging and understanding the lighting environment.

    Args:
        sh_coeffs: SH coefficients, shape (9, 3)
        size: Output image size (will be square)

    Returns:
        Rendered environment visualization, shape (size, size, 3)
    """
    # Create sphere normal map
    y, x = np.meshgrid(
        np.linspace(-1, 1, size),
        np.linspace(-1, 1, size),
        indexing='ij'
    )

    # Compute z from unit sphere
    r_squared = x**2 + y**2
    mask = r_squared <= 1.0
    z = np.zeros_like(x)
    z[mask] = np.sqrt(1.0 - r_squared[mask])

    # Create normal map
    normal_map = np.stack([x, y, z], axis=2)

    # Compute SH lighting
    lighting = compute_sh_lighting(normal_map, sh_coeffs)

    # Apply mask
    lighting[~mask] = 0.0

    # Tone map and convert to uint8
    lighting = np.clip(lighting, 0, 1)
    vis = (lighting * 255).astype(np.uint8)

    return vis
