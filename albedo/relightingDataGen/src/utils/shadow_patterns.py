"""
Realistic shadow pattern generation for image relighting.

Implements procedural generation of common shadow patterns:
- Venetian blinds (horizontal/vertical slats)
- Window frames (geometric grids)
- Tree leaves (organic Perlin noise)
- Lattice/architectural elements
- Voronoi cells (irregular organic patterns)

Based on research into IC-Light and photography shadow patterns.
"""

import numpy as np
import cv2
import random
from typing import Tuple, Optional


def generate_venetian_blind_pattern(
    shape: Tuple[int, int],
    num_slats: int = 10,
    slat_ratio: float = 0.4,
    orientation: str = 'horizontal',
    angle: float = 0
) -> np.ndarray:
    """
    Generate venetian blind shadow pattern.

    Common in product and portrait photography for dramatic effect.

    Args:
        shape: Output shape (H, W)
        num_slats: Number of slats (8-15 typical)
        slat_ratio: Width of slat vs gap (0.3-0.6 realistic)
        orientation: 'horizontal' or 'vertical'
        angle: Rotation angle in degrees (-30 to 30 realistic)

    Returns:
        Shadow mask [0, 1], shape (H, W)
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)

    if orientation == 'horizontal':
        slat_height = h // num_slats
        fill_height = int(slat_height * slat_ratio)

        for i in range(num_slats):
            y1 = i * slat_height
            y2 = min(y1 + fill_height, h)
            mask[y1:y2, :] = 1.0

    else:  # vertical
        slat_width = w // num_slats
        fill_width = int(slat_width * slat_ratio)

        for i in range(num_slats):
            x1 = i * slat_width
            x2 = min(x1 + fill_width, w)
            mask[:, x1:x2] = 1.0

    # Apply rotation
    if abs(angle) > 0.1:
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR)

    return mask


def generate_window_frame_pattern(
    shape: Tuple[int, int],
    grid_size: Tuple[int, int] = (3, 3),
    frame_width: int = 5,
    mullion_width: int = 3
) -> np.ndarray:
    """
    Generate window frame shadow pattern.

    Creates geometric grid pattern common in architectural photography.

    Args:
        shape: Output shape (H, W)
        grid_size: Number of panes (rows, cols) - (2, 2) to (4, 4) typical
        frame_width: Outer frame thickness in pixels (4-8 typical)
        mullion_width: Internal divider thickness (2-5 typical)

    Returns:
        Shadow mask [0, 1], shape (H, W)
    """
    h, w = shape
    mask = np.ones((h, w), dtype=np.float32)

    # Outer frame
    mask[:frame_width, :] = 0
    mask[-frame_width:, :] = 0
    mask[:, :frame_width] = 0
    mask[:, -frame_width:] = 0

    # Internal grid
    rows, cols = grid_size
    half_mullion = mullion_width // 2

    for i in range(1, rows):
        y = int(h * i / rows)
        y1 = max(0, y - half_mullion)
        y2 = min(h, y + half_mullion)
        mask[y1:y2, :] = 0

    for j in range(1, cols):
        x = int(w * j / cols)
        x1 = max(0, x - half_mullion)
        x2 = min(w, x + half_mullion)
        mask[:, x1:x2] = 0

    # Invert: frame is shadow
    return 1.0 - mask


def generate_perlin_noise_2d(
    shape: Tuple[int, int],
    scale: int = 10
) -> np.ndarray:
    """
    Generate 2D Perlin noise for organic shadow patterns.

    Uses grid-based gradient noise with smooth interpolation.

    Args:
        shape: Output shape (H, W)
        scale: Grid cell size (5-20 typical) - smaller = smoother

    Returns:
        Noise map [0, 1], shape (H, W)
    """
    h, w = shape

    def smoothstep(t):
        """Smooth interpolation function"""
        return 3 * t**2 - 2 * t**3

    def interpolate(a, b, t):
        """Smoothstep interpolation"""
        t = smoothstep(t)
        return a + t * (b - a)

    # Grid dimensions
    grid_h = (h // scale) + 2
    grid_w = (w // scale) + 2

    # Random gradients at grid points
    gradients = np.random.randn(grid_h, grid_w, 2)
    norms = np.linalg.norm(gradients, axis=2, keepdims=True) + 1e-8
    gradients = gradients / norms

    noise = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            # Grid cell coordinates
            cell_i = i // scale
            cell_j = j // scale

            # Position within cell [0, 1]
            local_i = (i % scale) / scale
            local_j = (j % scale) / scale

            if cell_i < grid_h - 1 and cell_j < grid_w - 1:
                # Corner gradients
                g00 = gradients[cell_i, cell_j]
                g01 = gradients[cell_i, cell_j + 1]
                g10 = gradients[cell_i + 1, cell_j]
                g11 = gradients[cell_i + 1, cell_j + 1]

                # Distance vectors from cell corners
                d00 = np.array([local_i, local_j])
                d01 = np.array([local_i, local_j - 1])
                d10 = np.array([local_i - 1, local_j])
                d11 = np.array([local_i - 1, local_j - 1])

                # Dot products
                n00 = np.dot(g00, d00)
                n01 = np.dot(g01, d01)
                n10 = np.dot(g10, d10)
                n11 = np.dot(g11, d11)

                # Bilinear interpolation
                nx0 = interpolate(n00, n01, local_j)
                nx1 = interpolate(n10, n11, local_j)
                noise[i, j] = interpolate(nx0, nx1, local_i)

    # Normalize to [0, 1]
    if noise.max() - noise.min() > 1e-8:
        noise = (noise - noise.min()) / (noise.max() - noise.min())

    return noise


def generate_fractal_brownian_motion(
    shape: Tuple[int, int],
    octaves: int = 5,
    persistence: float = 0.5,
    lacunarity: float = 2.0
) -> np.ndarray:
    """
    Generate fractal Brownian motion (fBm) for natural tree/foliage shadows.

    Layers multiple octaves of Perlin noise for realistic organic patterns.

    Args:
        shape: Output shape (H, W)
        octaves: Number of noise layers (4-6 typical)
        persistence: Amplitude decay per octave (0.4-0.6 typical)
        lacunarity: Frequency multiplier per octave (typically 2.0)

    Returns:
        Noise map [0, 1], shape (H, W)
    """
    result = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0

    for octave in range(octaves):
        scale = int(10 * frequency)
        if scale < 2:
            scale = 2

        # Generate noise at this frequency
        noise = generate_perlin_noise_2d(shape, scale=scale)

        # Add to result with current amplitude
        result += noise * amplitude

        # Track maximum for normalization
        max_value += amplitude

        # Update for next octave
        amplitude *= persistence
        frequency *= lacunarity

    # Normalize
    if max_value > 1e-8:
        result /= max_value

    return result


def generate_voronoi_cells(
    shape: Tuple[int, int],
    num_seeds: int = 80,
    min_distance: int = 15,
    cell_probability: float = 0.5
) -> np.ndarray:
    """
    Generate Voronoi cell pattern for irregular organic shadows.

    Creates cellular pattern useful for foliage and organic effects.

    Args:
        shape: Output shape (H, W)
        num_seeds: Number of seed points (50-150 typical)
        min_distance: Minimum spacing between seeds (10-25 pixels)
        cell_probability: Fraction of cells that are shadowed (0.3-0.7)

    Returns:
        Shadow mask [0, 1], shape (H, W)
    """
    h, w = shape

    # Generate seed points with minimum distance
    seeds = []
    attempts = 0
    max_attempts = num_seeds * 100

    while len(seeds) < num_seeds and attempts < max_attempts:
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)

        # Check minimum distance constraint
        if len(seeds) == 0 or all(
            np.sqrt((x - sx)**2 + (y - sy)**2) >= min_distance
            for sx, sy in seeds
        ):
            seeds.append((x, y))

        attempts += 1

    if len(seeds) == 0:
        return np.zeros(shape, dtype=np.float32)

    # Create distance map for each pixel to find closest seed
    y_coords, x_coords = np.meshgrid(
        np.arange(h),
        np.arange(w),
        indexing='ij'
    )

    closest_seed = np.zeros((h, w), dtype=np.int32)
    min_dist = np.full((h, w), float('inf'), dtype=np.float32)

    for seed_idx, (sx, sy) in enumerate(seeds):
        dist = np.sqrt((x_coords - sx)**2 + (y_coords - sy)**2)
        closer = dist < min_dist
        closest_seed[closer] = seed_idx
        min_dist[closer] = dist[closer]

    # Randomly select cells to be shadowed
    shadowed_cells = set(
        i for i in range(len(seeds))
        if random.random() < cell_probability
    )

    # Create mask
    mask = np.zeros(shape, dtype=np.float32)
    for seed_idx in shadowed_cells:
        mask[closest_seed == seed_idx] = 1.0

    return mask


def generate_lattice_pattern(
    shape: Tuple[int, int],
    spacing: int = 25,
    element_width: int = 3,
    pattern_type: str = 'grid'
) -> np.ndarray:
    """
    Generate lattice/railing shadow pattern.

    Creates architectural shadow patterns from railings, pergolas, etc.

    Args:
        shape: Output shape (H, W)
        spacing: Distance between elements in pixels (20-40 typical)
        element_width: Thickness of lines in pixels (2-5 typical)
        pattern_type: 'grid', 'diagonal', or 'cross'

    Returns:
        Shadow mask [0, 1], shape (H, W)
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)

    if pattern_type == 'grid':
        # Vertical lines
        for x in range(0, w, spacing):
            x1 = x
            x2 = min(x + element_width, w)
            mask[:, x1:x2] = 1.0

        # Horizontal lines
        for y in range(0, h, spacing):
            y1 = y
            y2 = min(y + element_width, h)
            mask[y1:y2, :] = 1.0

    elif pattern_type == 'diagonal':
        # Create diagonal lines
        for offset in range(-max(h, w), max(h, w), spacing):
            cv2.line(mask, (offset, 0), (offset + h, h), 1.0, element_width)
            cv2.line(mask, (offset, h), (offset + h, 0), 1.0, element_width)

    elif pattern_type == 'cross':
        # Diagonal cross-hatch
        for offset in range(-max(h, w), max(h, w), spacing):
            cv2.line(mask, (offset, 0), (offset + h, h), 1.0, element_width)

    return mask


def generate_curtain_pattern(
    shape: Tuple[int, int],
    num_folds: int = 8,
    fold_width_range: Tuple[int, int] = (20, 50),
    irregularity: float = 0.3
) -> np.ndarray:
    """
    Generate curtain/fabric fold shadow pattern.

    Creates vertical wavy patterns simulating curtain folds casting shadows.
    Based on typical fabric drapery in portrait photography.

    Args:
        shape: Output shape (H, W)
        num_folds: Number of vertical folds (6-12 typical)
        fold_width_range: Width variation of folds in pixels
        irregularity: Irregularity factor (0=smooth, 1=very irregular)

    Returns:
        Shadow mask [0, 1], shape (H, W)
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)

    # Generate fold positions
    fold_positions = np.linspace(0, w, num_folds + 1, dtype=int)

    for i in range(num_folds):
        # Random fold width
        fold_width = random.randint(*fold_width_range)

        # Create vertical gradient for this fold
        x_start = fold_positions[i]
        x_end = min(fold_positions[i] + fold_width, w)

        if x_end <= x_start:
            continue

        # Create sinusoidal variation along height
        y_coords = np.linspace(0, 4 * np.pi, h)
        wave = (np.sin(y_coords) + 1) / 2  # [0, 1]

        # Add irregularity
        if irregularity > 0:
            noise = np.random.rand(h) * irregularity
            wave = np.clip(wave + noise - irregularity/2, 0, 1)

        # Create gradient across fold width
        for x in range(x_start, x_end):
            # Gaussian-like profile
            pos = (x - x_start) / fold_width
            intensity = np.exp(-((pos - 0.5) ** 2) / 0.1)
            mask[:, x] = np.maximum(mask[:, x], wave * intensity)

    return mask


def generate_fence_pattern(
    shape: Tuple[int, int],
    picket_width: int = 15,
    gap_width: int = 10,
    num_horizontal: int = 2,
    horizontal_width: int = 8
) -> np.ndarray:
    """
    Generate fence/picket shadow pattern.

    Creates vertical picket fence patterns with horizontal rails.
    Common in outdoor portrait photography.

    Args:
        shape: Output shape (H, W)
        picket_width: Width of vertical pickets (10-20 typical)
        gap_width: Gap between pickets (8-15 typical)
        num_horizontal: Number of horizontal rails (1-3 typical)
        horizontal_width: Width of horizontal rails

    Returns:
        Shadow mask [0, 1], shape (H, W)
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)

    # Vertical pickets
    spacing = picket_width + gap_width
    for x in range(0, w, spacing):
        x1 = x
        x2 = min(x + picket_width, w)
        mask[:, x1:x2] = 1.0

    # Horizontal rails
    if num_horizontal > 0:
        rail_positions = np.linspace(h * 0.2, h * 0.8, num_horizontal, dtype=int)
        for y_pos in rail_positions:
            y1 = y_pos
            y2 = min(y_pos + horizontal_width, h)
            mask[y1:y2, :] = 1.0

    return mask


def generate_branch_pattern(
    shape: Tuple[int, int],
    num_branches: int = 6,
    branch_thickness_range: Tuple[int, int] = (3, 8),
    num_twigs_per_branch: int = 5
) -> np.ndarray:
    """
    Generate tree branch shadow pattern.

    Creates defined branch structures (more structured than tree foliage).
    Based on bare tree branches or palm fronds in photography.

    Args:
        shape: Output shape (H, W)
        num_branches: Number of main branches (4-8 typical)
        branch_thickness_range: Thickness variation for branches
        num_twigs_per_branch: Number of smaller twigs per branch

    Returns:
        Shadow mask [0, 1], shape (H, W)
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)

    # Generate main branches from edges toward center
    for _ in range(num_branches):
        # Random start position (usually from edges)
        if random.random() < 0.5:
            # From top or bottom
            start_x = random.randint(0, w)
            start_y = 0 if random.random() < 0.5 else h
        else:
            # From left or right
            start_x = 0 if random.random() < 0.5 else w
            start_y = random.randint(0, h)

        # Random end position
        end_x = random.randint(w // 4, 3 * w // 4)
        end_y = random.randint(h // 4, 3 * h // 4)

        # Draw main branch
        thickness = random.randint(*branch_thickness_range)
        cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, thickness)

        # Add smaller twigs
        for _ in range(num_twigs_per_branch):
            # Twig starts along main branch
            t = random.uniform(0.2, 0.8)
            twig_start_x = int(start_x + t * (end_x - start_x))
            twig_start_y = int(start_y + t * (end_y - start_y))

            # Twig ends nearby
            twig_end_x = twig_start_x + random.randint(-w//6, w//6)
            twig_end_y = twig_start_y + random.randint(-h//6, h//6)

            twig_thickness = max(1, thickness // 2)
            cv2.line(mask, (twig_start_x, twig_start_y), (twig_end_x, twig_end_y), 1.0, twig_thickness)

    return mask


def generate_cloud_shadow_pattern(
    shape: Tuple[int, int],
    num_clouds: int = 3,
    cloud_scale: float = 0.3,
    softness: float = 0.7
) -> np.ndarray:
    """
    Generate soft cloud shadow pattern.

    Creates large, soft, irregular shadows like clouds passing overhead.
    Common in outdoor photography.

    Args:
        shape: Output shape (H, W)
        num_clouds: Number of cloud-like regions (2-4 typical)
        cloud_scale: Size scale of clouds (0.2-0.5)
        softness: How soft/diffuse the edges are (0.5-0.9)

    Returns:
        Shadow mask [0, 1], shape (H, W)
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)

    for _ in range(num_clouds):
        # Random cloud center
        center_x = random.randint(-w//4, w + w//4)
        center_y = random.randint(-h//4, h + h//4)

        # Random cloud size
        radius_x = int(w * cloud_scale * random.uniform(0.8, 1.2))
        radius_y = int(h * cloud_scale * random.uniform(0.8, 1.2))

        # Create cloud mask using multiple overlapping ellipses
        cloud_mask = np.zeros((h, w), dtype=np.float32)
        num_lobes = random.randint(3, 6)

        for _ in range(num_lobes):
            lobe_x = center_x + random.randint(-radius_x//2, radius_x//2)
            lobe_y = center_y + random.randint(-radius_y//2, radius_y//2)
            lobe_rx = radius_x + random.randint(-radius_x//4, radius_x//4)
            lobe_ry = radius_y + random.randint(-radius_y//4, radius_y//4)

            cv2.ellipse(cloud_mask, (lobe_x, lobe_y), (lobe_rx, lobe_ry), 0, 0, 360, 1.0, -1)

        # Apply heavy blur for softness
        blur_size = int(min(h, w) * softness * 0.1)
        if blur_size % 2 == 0:
            blur_size += 1
        if blur_size > 1:
            cloud_mask = cv2.GaussianBlur(cloud_mask, (blur_size, blur_size), 0)

        # Add to main mask
        mask = np.maximum(mask, cloud_mask)

    # Normalize
    if mask.max() > 0:
        mask = mask / mask.max()

    return mask


def generate_screen_pattern(
    shape: Tuple[int, int],
    cell_size_range: Tuple[int, int] = (15, 30),
    element_width: int = 3,
    pattern_style: str = 'geometric'
) -> np.ndarray:
    """
    Generate architectural screen/mashrabiya shadow pattern.

    Creates geometric architectural screen patterns common in Middle Eastern
    and modern architecture photography.

    Args:
        shape: Output shape (H, W)
        cell_size_range: Size of repeating cells (15-40 typical)
        element_width: Width of screen elements
        pattern_style: 'geometric', 'star', or 'hexagon'

    Returns:
        Shadow mask [0, 1], shape (H, W)
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)

    cell_size = random.randint(*cell_size_range)

    if pattern_style == 'geometric':
        # Diamond/square grid pattern
        for y in range(0, h, cell_size):
            for x in range(0, w, cell_size):
                # Draw diamond shape
                pts = np.array([
                    [x + cell_size//2, y],
                    [x + cell_size, y + cell_size//2],
                    [x + cell_size//2, y + cell_size],
                    [x, y + cell_size//2]
                ], dtype=np.int32)
                cv2.polylines(mask, [pts], True, 1.0, element_width)

    elif pattern_style == 'star':
        # Star pattern in grid
        for y in range(0, h, cell_size):
            for x in range(0, w, cell_size):
                center_x = x + cell_size // 2
                center_y = y + cell_size // 2
                radius = cell_size // 3

                # 8-pointed star
                for angle in range(0, 360, 45):
                    rad = np.radians(angle)
                    end_x = int(center_x + radius * np.cos(rad))
                    end_y = int(center_y + radius * np.sin(rad))
                    cv2.line(mask, (center_x, center_y), (end_x, end_y), 1.0, element_width)

    elif pattern_style == 'hexagon':
        # Hexagonal tiling
        hex_height = cell_size
        hex_width = int(cell_size * 1.15)

        for row in range(-1, h // hex_height + 2):
            for col in range(-1, w // hex_width + 2):
                # Offset every other row
                offset_x = (hex_width // 2) if row % 2 == 1 else 0
                center_x = col * hex_width + offset_x + hex_width // 2
                center_y = row * hex_height + hex_height // 2

                # Draw hexagon
                pts = []
                for angle in range(0, 360, 60):
                    rad = np.radians(angle)
                    pt_x = int(center_x + (cell_size // 2) * np.cos(rad))
                    pt_y = int(center_y + (cell_size // 2) * np.sin(rad))
                    pts.append([pt_x, pt_y])

                pts = np.array(pts, dtype=np.int32)
                cv2.polylines(mask, [pts], True, 1.0, element_width)

    return mask


def apply_random_transform(
    mask: np.ndarray,
    max_rotation: float = 15.0,
    scale_range: Tuple[float, float] = (0.8, 1.2),
    translate_range: Tuple[float, float] = (-0.1, 0.1)
) -> np.ndarray:
    """
    Apply random affine transformations to shadow pattern.

    Args:
        mask: Input shadow mask
        max_rotation: Maximum rotation angle in degrees
        scale_range: (min, max) scale factor
        translate_range: (min, max) translation as fraction of image size

    Returns:
        Transformed mask
    """
    h, w = mask.shape[:2]
    center = (w // 2, h // 2)

    # Random rotation
    angle = random.uniform(-max_rotation, max_rotation)
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Random scale
    scale_x = random.uniform(*scale_range)
    scale_y = random.uniform(*scale_range)
    M_scale = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0]
    ], dtype=np.float32)

    # Random translation
    tx = int(random.uniform(*translate_range) * w)
    ty = int(random.uniform(*translate_range) * h)
    M_trans = np.array([
        [1, 0, tx],
        [0, 1, ty]
    ], dtype=np.float32)

    # Combine transformations
    M_rot_3x3 = np.vstack([M_rot, [0, 0, 1]])
    M_scale_3x3 = np.vstack([M_scale, [0, 0, 1]])
    M_trans_3x3 = np.vstack([M_trans, [0, 0, 1]])

    M_combined = M_rot_3x3 @ M_scale_3x3 @ M_trans_3x3
    M_final = M_combined[:2, :]

    # Apply transformation
    transformed = cv2.warpAffine(
        mask,
        M_final,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return transformed


def generate_random_shadow_pattern(
    shape: Tuple[int, int],
    pattern_type: Optional[str] = None
) -> Tuple[np.ndarray, dict]:
    """
    Generate random shadow pattern with realistic parameters.

    Randomly selects pattern type and parameters for diversity.

    Args:
        shape: Output shape (H, W)
        pattern_type: Force specific type, or None for random

    Returns:
        Tuple of (shadow_mask, metadata)
    """
    h, w = shape

    # Pattern type distribution - 10 types from photography research
    if pattern_type is None:
        pattern_types = [
            'tree_foliage',      # 20% - Organic dappled light (fBm)
            'venetian_blind',    # 18% - Horizontal/vertical slats
            'window_frame',      # 15% - Architectural grid
            'branch',            # 12% - Defined tree branches
            'curtain',           # 10% - Fabric fold patterns
            'fence',             # 8%  - Picket fence
            'voronoi',           # 7%  - Irregular cells
            'lattice',           # 5%  - Grid/diagonal
            'cloud',             # 3%  - Soft cloud shadows
            'screen'             # 2%  - Architectural screens
        ]
        weights = [0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.03, 0.02]
        pattern_type = random.choices(pattern_types, weights=weights)[0]

    metadata = {'pattern_type': pattern_type}

    # Generate pattern
    if pattern_type == 'venetian_blind':
        num_slats = random.randint(8, 15)
        slat_ratio = random.uniform(0.35, 0.55)
        orientation = random.choice(['horizontal', 'vertical'])
        angle = random.uniform(-20, 20)

        mask = generate_venetian_blind_pattern(
            shape,
            num_slats=num_slats,
            slat_ratio=slat_ratio,
            orientation=orientation,
            angle=angle
        )

        metadata.update({
            'num_slats': num_slats,
            'slat_ratio': slat_ratio,
            'orientation': orientation,
            'angle': angle
        })

    elif pattern_type == 'window_frame':
        grid_options = [(2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3)]
        grid_size = random.choice(grid_options)
        frame_width = random.randint(4, 8)
        mullion_width = random.randint(2, 5)

        mask = generate_window_frame_pattern(
            shape,
            grid_size=grid_size,
            frame_width=frame_width,
            mullion_width=mullion_width
        )

        metadata.update({
            'grid_size': grid_size,
            'frame_width': frame_width,
            'mullion_width': mullion_width
        })

    elif pattern_type == 'tree_foliage':
        octaves = random.randint(4, 6)
        persistence = random.uniform(0.45, 0.6)
        threshold = random.uniform(0.4, 0.65)

        mask = generate_fractal_brownian_motion(
            shape,
            octaves=octaves,
            persistence=persistence
        )

        # Convert to binary with threshold
        mask = (mask > threshold).astype(np.float32)

        metadata.update({
            'octaves': octaves,
            'persistence': persistence,
            'threshold': threshold
        })

    elif pattern_type == 'voronoi':
        num_seeds = random.randint(60, 120)
        min_distance = random.randint(12, 22)
        cell_probability = random.uniform(0.4, 0.65)

        mask = generate_voronoi_cells(
            shape,
            num_seeds=num_seeds,
            min_distance=min_distance,
            cell_probability=cell_probability
        )

        metadata.update({
            'num_seeds': num_seeds,
            'min_distance': min_distance,
            'cell_probability': cell_probability
        })

    elif pattern_type == 'lattice':
        spacing = random.randint(20, 40)
        element_width = random.randint(2, 5)
        pattern_subtype = random.choice(['grid', 'diagonal', 'cross'])

        mask = generate_lattice_pattern(
            shape,
            spacing=spacing,
            element_width=element_width,
            pattern_type=pattern_subtype
        )

        metadata.update({
            'spacing': spacing,
            'element_width': element_width,
            'pattern_subtype': pattern_subtype
        })

    elif pattern_type == 'curtain':
        num_folds = random.randint(6, 12)
        fold_width_range = (20, 50)
        irregularity = random.uniform(0.2, 0.4)

        mask = generate_curtain_pattern(
            shape,
            num_folds=num_folds,
            fold_width_range=fold_width_range,
            irregularity=irregularity
        )

        metadata.update({
            'num_folds': num_folds,
            'fold_width_range': fold_width_range,
            'irregularity': irregularity
        })

    elif pattern_type == 'fence':
        picket_width = random.randint(10, 20)
        gap_width = random.randint(8, 15)
        num_horizontal = random.randint(1, 3)
        horizontal_width = random.randint(6, 10)

        mask = generate_fence_pattern(
            shape,
            picket_width=picket_width,
            gap_width=gap_width,
            num_horizontal=num_horizontal,
            horizontal_width=horizontal_width
        )

        metadata.update({
            'picket_width': picket_width,
            'gap_width': gap_width,
            'num_horizontal': num_horizontal,
            'horizontal_width': horizontal_width
        })

    elif pattern_type == 'branch':
        num_branches = random.randint(4, 8)
        branch_thickness_range = (3, 8)
        num_twigs_per_branch = random.randint(3, 7)

        mask = generate_branch_pattern(
            shape,
            num_branches=num_branches,
            branch_thickness_range=branch_thickness_range,
            num_twigs_per_branch=num_twigs_per_branch
        )

        metadata.update({
            'num_branches': num_branches,
            'branch_thickness_range': branch_thickness_range,
            'num_twigs_per_branch': num_twigs_per_branch
        })

    elif pattern_type == 'cloud':
        num_clouds = random.randint(2, 4)
        cloud_scale = random.uniform(0.2, 0.4)
        softness = random.uniform(0.6, 0.8)

        mask = generate_cloud_shadow_pattern(
            shape,
            num_clouds=num_clouds,
            cloud_scale=cloud_scale,
            softness=softness
        )

        metadata.update({
            'num_clouds': num_clouds,
            'cloud_scale': cloud_scale,
            'softness': softness
        })

    elif pattern_type == 'screen':
        cell_size_range = (15, 30)
        element_width = random.randint(2, 4)
        pattern_style = random.choice(['geometric', 'star', 'hexagon'])

        mask = generate_screen_pattern(
            shape,
            cell_size_range=cell_size_range,
            element_width=element_width,
            pattern_style=pattern_style
        )

        metadata.update({
            'cell_size_range': cell_size_range,
            'element_width': element_width,
            'pattern_style': pattern_style
        })

    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    # Apply random transformations
    mask = apply_random_transform(mask)

    return mask, metadata
