import torch
import gc
from typing import Optional, Any


class ModelHelper:
    """
    Base helper class for managing pretrained models with device and memory management.

    Responsibilities:
    - Load pretrained model with specified dtype and options.
    - Manage model device transfers (CPU <-> GPU) with optional dtype conversion.
    - Clear GPU memory when moving model back to CPU to free resources.
    - Provide an abstract `run(inputs)` method to be implemented by subclasses for task-specific inference.

    Attributes:
        model_id (str): Identifier or path for the pretrained model.
        model_class (type): Hugging Face model class to instantiate.
        torch_dtype (torch.dtype): Desired data type for model weights and tensors.
        device (torch.device): Current device where the model resides. Defaults to CPU.
        model (torch.nn.Module, optional): The loaded model instance.
    """

    def __init__(
        self,
        model_id: str,
        model_class: type,
        torch_dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize the ModelHelper with model ID, model class, and torch dtype.

        Args:
            model_id (str): Model identifier or local path for pretrained weights.
            model_class (type): Hugging Face model class to load.
            torch_dtype (torch.dtype, optional): Data type for model weights. Defaults to torch.float32.
        """
        self.model_id: str = model_id
        self.model_class: type = model_class
        self.torch_dtype: torch.dtype = torch_dtype
        self.device: torch.device = torch.device(
            "cpu"
        )  # Default device is CPU on initialization
        self.model: Optional[torch.nn.Module] = (
            None  # Will hold the loaded model instance
        )

    @property
    def has_gpu(self) -> bool:
        """
        Check if a CUDA-capable GPU is available on the system.

        Returns:
            bool: True if CUDA GPU is available, False otherwise.
        """
        return torch.cuda.is_available()

    def load_model(self, device_map: Optional[Any] = None) -> None:
        """
        Load the pretrained model from the Hugging Face Hub or local directory.

        Args:
            device_map (Optional[Any], optional): Device mapping for model loading (e.g., for multi-GPU/offloading).
                                                  Defaults to None, which disables automatic device placement.

        Raises:
            Exception: Propagates exceptions raised during model loading.
        """
        try:
            self.model = self.model_class.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,  # Load weights in the desired precision
                low_cpu_mem_usage=True,  # Use memory-efficient loading on CPU
                use_safetensors=True,  # Prefer safe tensor format for faster loading
                device_map=device_map,  # Custom device mapping if provided
            )
            self.model.to(self.device)  # Move model to the current device (default CPU)
            print(f"Model loaded on {self.device} with dtype {self.torch_dtype}")
        except Exception as exc:
            print(f"Error loading model: {exc}")
            raise

    def to_gpu(self, dtype: Optional[torch.dtype] = None) -> None:
        """
        Move the model to GPU device, optionally converting to a specified dtype.

        Args:
            dtype (Optional[torch.dtype], optional): Desired dtype during GPU transfer.
                                                     Defaults to None, meaning use self.torch_dtype.

        Notes:
            - If GPU is not available, this method does nothing except printing a message.
            - Raises error if model is not loaded before calling.
        """
        if not self.has_gpu:
            print("No GPU available; to_gpu() skipped.")
            return
        if self.device.type == "cuda":
            print("Model already on GPU.")
            return
        if self.model is None:
            raise ValueError("Model not loaded yet; call load_model() before to_gpu().")

        self.device = torch.device("cuda")
        dtype = dtype or self.torch_dtype
        self.model.to(self.device, dtype=dtype)
        print(f"Model moved to {self.device} with dtype {dtype}")

    def to_cpu(self, dtype: Optional[torch.dtype] = None) -> None:
        """
        Move the model back to CPU device, optionally converting to specified dtype, and clear GPU cache.

        Args:
            dtype (Optional[torch.dtype], optional): Desired dtype during CPU transfer.
                                                     Defaults to None, meaning use self.torch_dtype.

        Notes:
            - Raises error if model is not loaded before calling.
            - Clears GPU memory cache and runs garbage collection if a GPU is available.
        """
        dtype = dtype or self.torch_dtype
        if self.device.type == "cpu":
            print("Model already on CPU.")
            return
        if self.model is None:
            raise ValueError("Model not loaded yet; call load_model() before to_cpu().")

        self.model.to("cpu", dtype=dtype)
        self.device = torch.device("cpu")

        # Clear GPU RAM if GPU available to free resources
        if self.has_gpu:
            gc.collect()
            torch.cuda.empty_cache()
            print("Cleared GPU memory.")

        print("Model moved to CPU")

    def run(self, inputs: Any) -> Any:
        """
        Abstract method for performing inference with the model.

        Subclasses must override this method with task-specific inference logic.

        Args:
            inputs (Any): Input data for the model inference; type depends on the specific model subclass.

        Raises:
            NotImplementedError: If called directly from this base class.
        """
        raise NotImplementedError("Subclasses must implement run(inputs)")