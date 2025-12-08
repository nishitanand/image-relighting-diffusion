"""
Ambient Occlusion computation for realistic contact shadows.

Implements Screen-Space Ambient Occlusion (SSAO) using depth and normal maps.
Based on Crytek's SSAO technique and LearnOpenGL tutorial.
"""

import numpy as np
import cv2
from typing import Tuple


def generate_hemisphere_samples(num_samples: int = 64) -> np.ndarray:
    """
    Generate random samples on a hemisphere for SSAO.

    Samples are weighted towards the normal direction (cosine distribution).

    Args:
        num_samples: Number of sample points to generate

    Returns:
        Array of shape (num_samples, 3) with hemisphere sample vectors
    """
    samples = []

    for i in range(num_samples):
        # Random point on hemisphere
        sample = np.array([
            np.random.uniform(-1.0, 1.0),  # x
            np.random.uniform(-1.0, 1.0),  # y
            np.random.uniform(0.0, 1.0)    # z (always positive for hemisphere)
        ])

        # Normalize to unit sphere
        sample = sample / (np.linalg.norm(sample) + 1e-8)

        # Scale with accelerating distribution towards normal
        scale = i / num_samples
        scale = 0.1 + 0.9 * scale * scale  # Quadratic falloff
        sample = sample * scale

        samples.append(sample)

    return np.array(samples)


def depth_to_position(depth: float, x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Convert screen-space depth to 3D position.

    Assumes simple orthographic projection for efficiency.

    Args:
        depth: Depth value [0, 1]
        x, y: Screen coordinates
        width, height: Image dimensions

    Returns:
        3D position vector
    """
    # Normalize to [-1, 1]
    ndc_x = (2.0 * x / width) - 1.0
    ndc_y = (2.0 * y / height) - 1.0

    return np.array([ndc_x, ndc_y, depth])


def position_to_screen(pos: np.ndarray, width: int, height: int) -> Tuple[int, int]:
    """
    Convert 3D position back to screen coordinates.

    Args:
        pos: 3D position vector
        width, height: Image dimensions

    Returns:
        Tuple of (x, y) screen coordinates
    """
    x = int((pos[0] + 1.0) * 0.5 * width)
    y = int((pos[1] + 1.0) * 0.5 * height)

    return x, y


def create_tbn_from_normal(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create tangent and bitangent vectors from normal for TBN matrix.

    Args:
        normal: Normal vector (3,)

    Returns:
        Tuple of (tangent, bitangent) vectors
    """
    # Choose a vector not parallel to normal
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(normal, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])

    # Gram-Schmidt orthogonalization
    tangent = np.cross(up, normal)
    tangent = tangent / (np.linalg.norm(tangent) + 1e-8)

    bitangent = np.cross(normal, tangent)
    bitangent = bitangent / (np.linalg.norm(bitangent) + 1e-8)

    return tangent, bitangent


def compute_ssao_optimized(
    depth_map: np.ndarray,
    normal_map: np.ndarray,
    num_samples: int = 16,
    radius: float = 0.05,
    bias: float = 0.025,
    subsample_factor: int = 2
) -> np.ndarray:
    """
    Compute Screen-Space Ambient Occlusion (optimized version).

    This is a performance-optimized implementation that:
    - Subsamples the computation
    - Uses vectorized operations where possible
    - Applies bilateral upsampling

    Args:
        depth_map: Normalized depth map [0, 1], shape (H, W)
        normal_map: Surface normals [-1, 1], shape (H, W, 3)
        num_samples: Number of sample points per pixel (default: 16)
        radius: Sampling radius in normalized coordinates (default: 0.05)
        bias: Depth bias to prevent self-occlusion (default: 0.025)
        subsample_factor: Compute at 1/N resolution (default: 2)

    Returns:
        Ambient occlusion map [0, 1], shape (H, W)
        Higher values = less occluded (brighter)
    """
    h, w = depth_map.shape

    # Subsample for performance
    h_sub = h // subsample_factor
    w_sub = w // subsample_factor

    depth_sub = cv2.resize(depth_map, (w_sub, h_sub), interpolation=cv2.INTER_LINEAR)
    normal_sub = cv2.resize(normal_map, (w_sub, h_sub), interpolation=cv2.INTER_LINEAR)

    # Normalize normals after resize
    normal_norms = np.linalg.norm(normal_sub, axis=2, keepdims=True)
    normal_sub = normal_sub / (normal_norms + 1e-8)

    ao_map_sub = np.ones((h_sub, w_sub), dtype=np.float32)

    # Generate sample kernel
    kernel = generate_hemisphere_samples(num_samples)

    # Process each pixel (can be parallelized further)
    for y in range(h_sub):
        for x in range(w_sub):
            frag_depth = depth_sub[y, x]
            frag_normal = normal_sub[y, x]

            # Skip invalid normals
            if np.linalg.norm(frag_normal) < 0.1:
                continue

            # Build TBN matrix to orient samples
            tangent, bitangent = create_tbn_from_normal(frag_normal)
            tbn = np.column_stack([tangent, bitangent, frag_normal])

            occlusion = 0.0

            for sample in kernel:
                # Orient sample by surface normal
                sample_world = tbn @ sample

                # Calculate sample position
                sample_offset = sample_world[:2] * radius
                sample_x = int(x + sample_offset[0] * w_sub)
                sample_y = int(y + sample_offset[1] * h_sub)

                # Check bounds
                if not (0 <= sample_x < w_sub and 0 <= sample_y < h_sub):
                    continue

                # Get sample depth
                sample_depth = depth_sub[sample_y, sample_x]

                # Expected depth based on sample direction
                expected_depth = frag_depth + sample_world[2] * radius

                # Range check to reduce false occlusion from distant surfaces
                depth_diff = abs(sample_depth - frag_depth)
                range_check = depth_diff < radius

                # Check if sample is occluded
                if range_check and sample_depth > expected_depth + bias:
                    # Smooth falloff based on distance
                    falloff = 1.0 - (depth_diff / radius)
                    occlusion += falloff

            # Normalize and invert (1 = not occluded, 0 = fully occluded)
            ao_map_sub[y, x] = 1.0 - np.clip(occlusion / num_samples, 0.0, 1.0)

    # Bilateral upsampling to preserve edges
    ao_map = cv2.resize(ao_map_sub, (w, h), interpolation=cv2.INTER_LINEAR)

    # Blur to reduce noise while preserving depth edges
    # Use joint bilateral filter guided by depth
    ao_map = cv2.bilateralFilter(
        ao_map.astype(np.float32),
        d=5,
        sigmaColor=0.1,
        sigmaSpace=5
    )

    return ao_map


def compute_ssao_fast(
    depth_map: np.ndarray,
    normal_map: np.ndarray,
    num_samples: int = 8,
    radius: float = 0.08
) -> np.ndarray:
    """
    Fast approximation of SSAO using depth gradients.

    This is a simpler, faster alternative that works well for real-time applications.
    Based on depth curvature and local depth variance.

    Args:
        depth_map: Normalized depth map [0, 1], shape (H, W)
        normal_map: Surface normals [-1, 1], shape (H, W, 3)
        num_samples: Number of sample directions (default: 8)
        radius: Sampling radius in pixels (default: 0.08)

    Returns:
        Ambient occlusion map [0, 1], shape (H, W)
    """
    h, w = depth_map.shape

    # Compute depth gradients
    grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)

    # Compute curvature (Laplacian)
    laplacian = cv2.Laplacian(depth_map, cv2.CV_32F, ksize=3)

    # Convex regions (positive curvature) are less occluded
    # Concave regions (negative curvature) are more occluded
    convexity = -laplacian  # Negative curvature = occlusion
    convexity = np.clip(convexity, -1.0, 1.0)

    # Local depth variance (high variance = potential occlusion)
    kernel_size = int(radius * min(h, w))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, kernel_size)

    depth_blur = cv2.GaussianBlur(depth_map, (kernel_size, kernel_size), 0)
    depth_variance = np.abs(depth_map - depth_blur)

    # Combine curvature and variance
    ao_map = 1.0 - (convexity * 0.6 + depth_variance * 0.4)
    ao_map = np.clip(ao_map, 0.0, 1.0)

    # Smooth the result
    ao_map = cv2.GaussianBlur(ao_map, (5, 5), 0)

    return ao_map


def compute_ssao(
    depth_map: np.ndarray,
    normal_map: np.ndarray,
    quality: str = 'medium',
    **kwargs
) -> np.ndarray:
    """
    Compute Screen-Space Ambient Occlusion with quality presets.

    Args:
        depth_map: Normalized depth map [0, 1], shape (H, W)
        normal_map: Surface normals [-1, 1], shape (H, W, 3)
        quality: Quality preset - 'fast', 'medium', 'high' (default: 'medium')
        **kwargs: Additional parameters passed to SSAO implementation

    Returns:
        Ambient occlusion map [0, 1], shape (H, W)
    """
    # Normalize depth to [0, 1] if needed
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    if depth_max - depth_min > 1e-6:
        depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = depth_map

    # Quality presets
    if quality == 'fast':
        return compute_ssao_fast(
            depth_normalized,
            normal_map,
            num_samples=kwargs.get('num_samples', 8),
            radius=kwargs.get('radius', 0.08)
        )

    elif quality == 'medium':
        return compute_ssao_optimized(
            depth_normalized,
            normal_map,
            num_samples=kwargs.get('num_samples', 16),
            radius=kwargs.get('radius', 0.05),
            bias=kwargs.get('bias', 0.025),
            subsample_factor=kwargs.get('subsample_factor', 2)
        )

    elif quality == 'high':
        return compute_ssao_optimized(
            depth_normalized,
            normal_map,
            num_samples=kwargs.get('num_samples', 32),
            radius=kwargs.get('radius', 0.04),
            bias=kwargs.get('bias', 0.02),
            subsample_factor=kwargs.get('subsample_factor', 1)
        )

    else:
        raise ValueError(f"Unknown quality preset: {quality}. Use 'fast', 'medium', or 'high'.")
