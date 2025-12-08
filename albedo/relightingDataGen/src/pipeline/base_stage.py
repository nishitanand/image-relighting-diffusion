"""
Base stage class for the relighting pipeline.
All pipeline stages inherit from this abstract base class.
"""

from abc import ABC, abstractmethod
import torch
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseStage(ABC):
    """
    Abstract base class for pipeline stages.

    Each stage must implement:
    - load_model(): Load the model to GPU
    - process(): Process input data and return output

    The base class provides:
    - unload_model(): Free GPU memory by deleting model
    - Memory management utilities
    """

    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        """
        Initialize stage with configuration.

        Args:
            config: Configuration dictionary
            device: Device to run on ('cuda' or 'cpu')
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None

        logger.info(f"Initialized {self.__class__.__name__} on device: {self.device}")

    @abstractmethod
    def load_model(self):
        """
        Load model to GPU/device.
        Must be implemented by each stage.
        """
        pass

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data through the stage.

        Args:
            input_data: Dictionary containing input data

        Returns:
            Dictionary containing output data
        """
        pass

    def unload_model(self):
        """
        Unload model from GPU to free memory.
        """
        if self.model is not None:
            logger.info(f"Unloading {self.__class__.__name__} model")

            # Move model to CPU first (helps with cleanup)
            if hasattr(self.model, 'to'):
                self.model.to('cpu')

            # Delete model
            del self.model
            self.model = None

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logger.info(f"Model unloaded successfully")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.

        Returns:
            Dictionary with memory stats in GB
        """
        if not torch.cuda.is_available():
            return {'allocated': 0.0, 'reserved': 0.0, 'free': 0.0}

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = total - allocated

        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
            'total': total
        }

    def log_memory_usage(self, prefix: str = ""):
        """
        Log current GPU memory usage.

        Args:
            prefix: Prefix string for log message
        """
        mem = self.get_memory_usage()
        logger.info(
            f"{prefix}GPU Memory - "
            f"Allocated: {mem['allocated']:.2f}GB, "
            f"Reserved: {mem['reserved']:.2f}GB, "
            f"Free: {mem['free']:.2f}GB"
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(device={self.device})"
