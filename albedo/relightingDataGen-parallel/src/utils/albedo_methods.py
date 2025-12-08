"""
Albedo extraction methods for intrinsic image decomposition.

Provides multiple methods from sophisticated (IntrinsicAnything) to simple (grayscale-based).
Includes fallback mechanisms for robust operation.
"""

import numpy as np
import cv2
from PIL import Image
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def multi_scale_retinex(
    image: np.ndarray,
    scales: list = [15, 80, 250],
    epsilon: float = 1.0
) -> np.ndarray:
    """
    Multi-Scale Retinex algorithm for albedo estimation.

    Based on: "A Multiscale Retinex for Bridging the Gap Between Color Images and
    the Human Observation of Scenes" (Rahman et al., 1997)

    Formula: MSR(x,y) = Σ w_i * [log(I(x,y)) - log(G_i(x,y) * I(x,y))]
    where G_i is Gaussian blur with scale σ_i

    Args:
        image: Input image (H, W, 3) in range [0, 255]
        scales: List of Gaussian blur scales
        epsilon: Small constant to avoid log(0)

    Returns:
        Albedo estimate (H, W, 3) in range [0, 1]
    """
    # Convert to float
    img = image.astype(np.float32)

    # Initialize retinex output
    retinex = np.zeros_like(img)

    # Apply each scale
    for scale in scales:
        # Gaussian blur
        blurred = cv2.GaussianBlur(img, (0, 0), scale)

        # Retinex: log(I) - log(blur(I))
        retinex += np.log10(img + epsilon) - np.log10(blurred + epsilon)

    # Average across scales
    retinex = retinex / len(scales)

    # Normalize to [0, 1]
    retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min() + 1e-8)

    return retinex


def simplest_color_balance(
    image: np.ndarray,
    percent: float = 1.0
) -> np.ndarray:
    """
    Simplest color balance algorithm.

    Saturates percent% of darkest and brightest pixels.

    Args:
        image: Input image (H, W, 3) in range [0, 1]
        percent: Percentage to saturate (0-100)

    Returns:
        Balanced image (H, W, 3)
    """
    out = np.zeros_like(image)

    for channel in range(image.shape[2]):
        # Flatten channel
        flat = image[:, :, channel].flatten()

        # Calculate percentiles
        low_val = np.percentile(flat, percent)
        high_val = np.percentile(flat, 100 - percent)

        # Clip and rescale
        out[:, :, channel] = np.clip((image[:, :, channel] - low_val) / (high_val - low_val), 0, 1)

    return out


def extract_albedo_retinex(
    image: Image.Image,
    scales: list = [15, 80, 250],
    apply_color_balance: bool = True
) -> Image.Image:
    """
    Extract albedo using Multi-Scale Retinex.

    This is the primary traditional method (Method 2 in plan).

    Args:
        image: PIL Image (RGB)
        scales: Gaussian blur scales
        apply_color_balance: Whether to apply color balancing

    Returns:
        Albedo as PIL Image
    """
    logger.info("Extracting albedo using Multi-Scale Retinex")

    # Convert to numpy
    img_np = np.array(image.convert('RGB')).astype(np.float32)

    # Apply MSR
    albedo = multi_scale_retinex(img_np, scales=scales)

    # Apply color balance if requested
    if apply_color_balance:
        albedo = simplest_color_balance(albedo, percent=1.0)

    # Convert back to PIL
    albedo_pil = Image.fromarray((albedo * 255).astype(np.uint8))

    logger.info("Retinex albedo extraction completed")
    return albedo_pil


def extract_albedo_lab_based(image: Image.Image) -> Image.Image:
    """
    Extract albedo using LAB color space decomposition.

    This is the simple fallback method (Method 3 in plan).

    Approach:
    1. Convert to LAB color space
    2. Extract L (lightness) channel
    3. Separate low-frequency (illumination) using bilateral filter
    4. Divide original by illumination estimate
    5. Reconstruct with original AB channels

    Args:
        image: PIL Image (RGB)

    Returns:
        Albedo as PIL Image
    """
    logger.info("Extracting albedo using LAB-based method")

    # Convert to numpy and LAB
    img_np = np.array(image.convert('RGB'))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Extract L channel
    L = lab[:, :, 0]

    # Bilateral filter to get low-frequency illumination
    # Preserves edges while smoothing
    illumination = cv2.bilateralFilter(L, d=9, sigmaColor=75, sigmaSpace=75)

    # Estimate albedo: L / illumination
    # Add epsilon to avoid division by zero
    albedo_L = L / (illumination + 1e-8)

    # Normalize to [0, 255]
    albedo_L = cv2.normalize(albedo_L, None, 0, 255, cv2.NORM_MINMAX)

    # Replace L channel with albedo
    lab_albedo = lab.copy()
    lab_albedo[:, :, 0] = albedo_L

    # Convert back to RGB
    albedo_rgb = cv2.cvtColor(lab_albedo.astype(np.uint8), cv2.COLOR_LAB2RGB)

    # Convert to PIL
    albedo_pil = Image.fromarray(albedo_rgb)

    logger.info("LAB-based albedo extraction completed")
    return albedo_pil


def extract_albedo_simple_division(
    image: Image.Image,
    blur_size: int = 101
) -> Image.Image:
    """
    Simplest albedo extraction: image / blurred_image.

    This assumes illumination is low-frequency and can be removed by
    dividing by a heavily blurred version.

    Args:
        image: PIL Image (RGB)
        blur_size: Gaussian blur kernel size (must be odd)

    Returns:
        Albedo as PIL Image
    """
    logger.info("Extracting albedo using simple division method")

    # Convert to numpy
    img_np = np.array(image.convert('RGB')).astype(np.float32)

    # Heavy Gaussian blur to estimate illumination
    illumination = cv2.GaussianBlur(img_np, (blur_size, blur_size), 0)

    # Divide to get albedo
    albedo = img_np / (illumination + 1e-8)

    # Clip to [0, 1]
    albedo = np.clip(albedo, 0, 1)

    # Apply color balance
    albedo = simplest_color_balance(albedo, percent=1.0)

    # Convert back to PIL
    albedo_pil = Image.fromarray((albedo * 255).astype(np.uint8))

    logger.info("Simple division albedo extraction completed")
    return albedo_pil


def enhance_albedo_detail(albedo: Image.Image, strength: float = 1.5) -> Image.Image:
    """
    Enhance details in albedo using unsharp masking.

    Args:
        albedo: PIL Image
        strength: Enhancement strength (1.0 = no change, >1 = sharpen)

    Returns:
        Enhanced albedo as PIL Image
    """
    # Convert to numpy
    img = np.array(albedo).astype(np.float32) / 255.0

    # Gaussian blur
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)

    # Unsharp mask: original + strength * (original - blurred)
    enhanced = img + strength * (img - blurred)

    # Clip and convert back
    enhanced = np.clip(enhanced, 0, 1)
    enhanced_pil = Image.fromarray((enhanced * 255).astype(np.uint8))

    return enhanced_pil


# Main interface function
def extract_albedo(
    image: Image.Image,
    method: str = "retinex",
    **kwargs
) -> Image.Image:
    """
    Extract albedo using specified method.

    Args:
        image: PIL Image (RGB)
        method: "retinex", "lab", or "simple"
        **kwargs: Method-specific parameters

    Returns:
        Albedo as PIL Image

    Raises:
        ValueError: If method is unknown
    """
    if method == "retinex":
        return extract_albedo_retinex(image, **kwargs)
    elif method == "lab":
        return extract_albedo_lab_based(image)
    elif method == "simple":
        return extract_albedo_simple_division(image, **kwargs)
    else:
        raise ValueError(f"Unknown albedo extraction method: {method}")


def blend_with_original(
    albedo: Image.Image,
    original: Image.Image,
    blend_ratio: float = 0.2
) -> Image.Image:
    """
    Blend albedo with original image to reduce whiteness/over-brightening.

    Albedo extraction can make images too white by removing all illumination.
    Mixing back some of the original image preserves more natural colors and tones.

    Formula: result = (1 - blend_ratio) * albedo + blend_ratio * original

    Args:
        albedo: Extracted albedo image (PIL)
        original: Original input image (PIL)
        blend_ratio: How much original to mix in (0-1)
                     0 = pure albedo (brightest)
                     0.2 = 20% original (recommended)
                     0.5 = 50/50 mix
                     1.0 = pure original (no albedo extraction)

    Returns:
        Blended albedo as PIL Image
    """
    logger.info(f"Blending albedo with {blend_ratio*100:.1f}% original image")

    # Convert to numpy
    albedo_np = np.array(albedo.convert('RGB')).astype(np.float32) / 255.0
    original_np = np.array(original.convert('RGB')).astype(np.float32) / 255.0

    # Ensure same size
    if albedo_np.shape != original_np.shape:
        # Resize original to match albedo
        original_resized = cv2.resize(
            original_np,
            (albedo_np.shape[1], albedo_np.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    else:
        original_resized = original_np

    # Blend
    blended = (1 - blend_ratio) * albedo_np + blend_ratio * original_resized

    # Clip to valid range
    blended = np.clip(blended, 0, 1)

    # Convert back to PIL
    blended_pil = Image.fromarray((blended * 255).astype(np.uint8))

    logger.info(f"Blending completed (blend_ratio={blend_ratio})")
    return blended_pil


def get_robust_albedo(image: Image.Image) -> Tuple[Image.Image, str]:
    """
    Get albedo using fallback strategy.

    Tries methods in order of sophistication, returns first successful result.

    Order:
    1. Retinex (best quality)
    2. LAB-based (medium quality)
    3. Simple division (always works)

    Args:
        image: PIL Image (RGB)

    Returns:
        Tuple of (albedo PIL Image, method name used)
    """
    # Try Retinex first
    try:
        logger.info("Attempting Retinex albedo extraction")
        albedo = extract_albedo_retinex(image)
        return albedo, "retinex"
    except Exception as e:
        logger.warning(f"Retinex failed: {e}, falling back to LAB method")

    # Try LAB-based
    try:
        logger.info("Attempting LAB-based albedo extraction")
        albedo = extract_albedo_lab_based(image)
        return albedo, "lab"
    except Exception as e:
        logger.warning(f"LAB method failed: {e}, falling back to simple division")

    # Ultimate fallback
    try:
        logger.info("Using simple division method")
        albedo = extract_albedo_simple_division(image)
        return albedo, "simple"
    except Exception as e:
        logger.error(f"All albedo methods failed: {e}")
        logger.warning("Returning original image as albedo")
        return image, "none"
