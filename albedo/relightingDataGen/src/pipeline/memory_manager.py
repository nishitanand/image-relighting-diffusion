"""
GPU memory management utilities for the relighting pipeline.
"""

import torch
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages GPU memory allocation and monitoring.
    """

    def __init__(self, max_memory_gb: float = 22.0):
        """
        Initialize memory manager.

        Args:
            max_memory_gb: Maximum GPU memory to use in GB (default: 22GB for A5000)
        """
        self.max_memory_gb = max_memory_gb
        self.max_memory = {0: f"{max_memory_gb}GiB"}

        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"Total GPU memory: {total_memory:.2f}GB")
            logger.info(f"Max memory limit set to: {max_memory_gb}GB")
        else:
            logger.warning("No GPU detected, will use CPU")

    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get detailed GPU memory statistics.

        Returns:
            Dictionary with memory stats in GB
        """
        if not torch.cuda.is_available():
            return {
                'allocated': 0.0,
                'reserved': 0.0,
                'free': 0.0,
                'total': 0.0,
                'peak': 0.0
            }

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        free = total - allocated

        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
            'total': total,
            'peak': peak
        }

    def log_memory_stats(self, prefix: str = ""):
        """
        Log current GPU memory statistics.

        Args:
            prefix: Prefix for log message
        """
        stats = self.get_memory_stats()
        logger.info(
            f"{prefix}GPU Memory - "
            f"Allocated: {stats['allocated']:.2f}GB, "
            f"Reserved: {stats['reserved']:.2f}GB, "
            f"Free: {stats['free']:.2f}GB, "
            f"Peak: {stats['peak']:.2f}GB"
        )

    def clear_cache(self):
        """
        Clear CUDA cache and synchronize.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("CUDA cache cleared")

    def reset_peak_stats(self):
        """
        Reset peak memory statistics.
        """
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            logger.debug("Peak memory statistics reset")

    def check_memory_available(self, required_gb: float) -> bool:
        """
        Check if required memory is available.

        Args:
            required_gb: Required memory in GB

        Returns:
            True if enough memory is available
        """
        if not torch.cuda.is_available():
            return True  # CPU fallback

        stats = self.get_memory_stats()
        available = stats['free']

        if available < required_gb:
            logger.warning(
                f"Insufficient memory: Required {required_gb:.2f}GB, "
                f"Available {available:.2f}GB"
            )
            return False

        return True

    def get_max_memory_config(self) -> Dict[int, str]:
        """
        Get max memory configuration for accelerate.

        Returns:
            Dictionary mapping device ID to memory limit
        """
        return self.max_memory

    @staticmethod
    def optimize_model_memory(model, enable_attention_slicing: bool = True):
        """
        Apply memory optimizations to a model.

        Args:
            model: Model to optimize
            enable_attention_slicing: Enable attention slicing for diffusion models

        Returns:
            Optimized model
        """
        # Enable attention slicing for diffusion models
        if enable_attention_slicing and hasattr(model, 'enable_attention_slicing'):
            model.enable_attention_slicing(1)
            logger.info("Enabled attention slicing")

        # Enable memory efficient attention if available
        if hasattr(model, 'enable_xformers_memory_efficient_attention'):
            try:
                model.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention")
            except Exception as e:
                logger.warning(f"Could not enable xformers: {e}")

        # Enable sequential CPU offload if available
        if hasattr(model, 'enable_sequential_cpu_offload'):
            logger.info("Sequential CPU offload available (not enabled by default)")

        return model

    def __repr__(self):
        return f"MemoryManager(max_memory={self.max_memory_gb}GB)"
