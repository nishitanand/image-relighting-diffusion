"""
Normal Estimation using MiDaS depth estimation.

Converts RGB images to surface normal maps for shading synthesis.
Uses MiDaS for robust depth estimation, then converts depth to normals.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import logging
from typing import Optional, Tuple
import cv2

logger = logging.getLogger(__name__)


class NormalEstimator:
    """
    Estimates surface normals from RGB images using MiDaS depth estimation.
    """

    def __init__(self, model_type: str = "DPT_Hybrid", device: str = 'cuda'):
        """
        Initialize MiDaS model for depth estimation.

        Args:
            model_type: MiDaS model type ("DPT_Large", "DPT_Hybrid", "MiDaS_small")
            device: Device to run model on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.model = None
        self.transform = None

        logger.info(f"Initializing NormalEstimator with {model_type} on {self.device}")

    def load_model(self):
        """Load MiDaS model and transforms."""
        if self.model is not None:
            return

        try:
            logger.info(f"Loading MiDaS model: {self.model_type}")

            # Load model from torch hub
            self.model = torch.hub.load(
                "intel-isl/MiDaS",
                self.model_type,
                trust_repo=True
            )
            self.model.to(self.device)
            self.model.eval()

            # Load appropriate transform
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)

            if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform

            logger.info("MiDaS model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load MiDaS model: {e}")
            raise

    def unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.transform = None
            torch.cuda.empty_cache()
            logger.info("MiDaS model unloaded")

    def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """
        Estimate depth map from RGB image.

        Args:
            image: PIL Image (RGB)

        Returns:
            Depth map as numpy array (H, W)
        """
        if self.model is None:
            self.load_model()

        # Convert to numpy and ensure RGB
        if isinstance(image, Image.Image):
            img = np.array(image.convert('RGB'))
        else:
            img = image

        # Apply transform
        input_batch = self.transform(img).to(self.device)

        # Predict depth
        with torch.no_grad():
            prediction = self.model(input_batch)

            # Resize to original resolution
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

        return depth

    def depth_to_normal(self, depth: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Convert depth map to surface normals.

        Uses gradient-based method:
        N = normalize([-dz/dx, -dz/dy, 1])

        Args:
            depth: Depth map (H, W)
            normalize: Whether to normalize normal vectors to unit length

        Returns:
            Normal map (H, W, 3) with values in range [-1, 1]
        """
        # Compute gradients (dz/dx, dz/dy)
        zy, zx = np.gradient(depth)

        # Normal is perpendicular to surface
        # N = (-dz/dx, -dz/dy, 1)
        normal = np.dstack((-zx, -zy, np.ones_like(depth)))

        if normalize:
            # Normalize to unit length
            norm = np.linalg.norm(normal, axis=2, keepdims=True)
            norm = np.maximum(norm, 1e-8)  # avoid division by zero
            normal = normal / norm

        return normal

    def estimate_normal(self, image: Image.Image) -> np.ndarray:
        """
        Estimate surface normals directly from RGB image.

        Args:
            image: PIL Image (RGB)

        Returns:
            Normal map (H, W, 3) with values in range [-1, 1]
        """
        # Estimate depth
        depth = self.estimate_depth(image)

        # Convert to normals
        normal = self.depth_to_normal(depth)

        return normal

    def visualize_normal(self, normal: np.ndarray) -> np.ndarray:
        """
        Convert normal map to RGB visualization.

        Maps [-1, 1] to [0, 255] for visualization.

        Args:
            normal: Normal map (H, W, 3)

        Returns:
            RGB image (H, W, 3) in uint8 format
        """
        # Map from [-1, 1] to [0, 1]
        normal_vis = (normal + 1.0) / 2.0

        # Convert to uint8
        normal_vis = (normal_vis * 255).astype(np.uint8)

        return normal_vis


def estimate_normals_fast(image: Image.Image, device: str = 'cuda') -> np.ndarray:
    """
    Fast normal estimation using lightweight MiDaS model.

    Args:
        image: PIL Image (RGB)
        device: Device to run on

    Returns:
        Normal map (H, W, 3)
    """
    estimator = NormalEstimator(model_type="MiDaS_small", device=device)
    try:
        normal = estimator.estimate_normal(image)
        return normal
    finally:
        estimator.unload_model()


def estimate_normals_quality(image: Image.Image, device: str = 'cuda') -> np.ndarray:
    """
    High-quality normal estimation using DPT-Hybrid.

    Args:
        image: PIL Image (RGB)
        device: Device to run on

    Returns:
        Normal map (H, W, 3)
    """
    estimator = NormalEstimator(model_type="DPT_Hybrid", device=device)
    try:
        normal = estimator.estimate_normal(image)
        return normal
    finally:
        estimator.unload_model()


# Convenience function for the pipeline
def get_normal_map(
    image: Image.Image,
    method: str = "fast",
    device: str = 'cuda'
) -> np.ndarray:
    """
    Get normal map from image using specified method.

    Args:
        image: PIL Image (RGB)
        method: "fast" (MiDaS_small) or "quality" (DPT_Hybrid)
        device: Device to run on

    Returns:
        Normal map (H, W, 3) with values in [-1, 1]
    """
    if method == "fast":
        return estimate_normals_fast(image, device)
    elif method == "quality":
        return estimate_normals_quality(image, device)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'fast' or 'quality'")
