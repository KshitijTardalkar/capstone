"""
Base class for loading and managing AI models.

This file defines an abstract base class (ABC) `ModelHelper` which provides a
common interface for all models (STT, LLM, TTS) in the application.
It handles device selection (CPU/GPU) and provides abstract methods
for loading and running models, ensuring a consistent structure.
"""

import torch
from typing import Optional, Any
from abc import ABC, abstractmethod


class ModelHelper(ABC):
    """
    Abstract base class for a model service.

    Provides common functionality like device auto-selection and a standardized
    interface for loading and running models.

    Attributes:
        model_id (str): The Hugging Face model identifier.
        model_class (type): The Hugging Face `AutoModel` class (e.g., AutoModelForCausalLM).
        torch_dtype (torch.dtype): The data type for model weights (e.g., torch.float16).
        use_gpu (bool): Flag to attempt to use the GPU if available.
        use_flash_attn (bool): Flag to attempt to use Flash Attention 2.
        device (torch.device): The device (cuda or cpu) the model will run on.
        model (Optional[torch.nn.Module]): The loaded model instance.
    """

    def __init__(
        self,
        model_id: str,
        model_class: type,
        torch_dtype: torch.dtype = torch.float32,
        use_gpu: bool = True,
        use_flash_attn: bool = False
    ) -> None:
        self.model_id: str = model_id
        self.model_class: type = model_class
        self.torch_dtype: torch.dtype = torch_dtype
        self.use_gpu: bool = use_gpu
        self.use_flash_attn: bool = use_flash_attn
        self.device: torch.device = self._get_target_device()
        self.model: Optional[torch.nn.Module] = None

    @property
    def _is_gpu_available(self) -> bool:
        """Checks if a CUDA-compatible GPU is available."""
        return torch.cuda.is_available()

    def _get_target_device(self) -> torch.device:
        """Determines the target device (cuda or cpu) based on availability and user preference."""
        if self._is_gpu_available and self.use_gpu:
            return torch.device("cuda")
        return torch.device("cpu")

    @abstractmethod
    def load_model(self) -> None:
        """
        Abstract method for loading the model.

        Subclasses must implement this method to handle specific model
        loading logic, including quantization and compilation.
        """
        raise NotImplementedError("Subclasses must implement load_model()")

    @abstractmethod
    def run(self, inputs: Any, **kwargs) -> Any:
        """
        Abstract method for running model inference.

        Subclasses must implement this to define the inference logic.
        """
        raise NotImplementedError("Subclasses must implement run(inputs)")