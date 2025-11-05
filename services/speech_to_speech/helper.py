import torch
import gc
from typing import Optional, Any


class ModelHelper:
    """
    Base helper class for managing pretrained models with device and memory management.

    Responsibilities:
    - Load pretrained model onto a specified device (GPU or CPU).
    - Provide properties to check device availability and target device.
    - Provide an abstract `run(inputs)` method to be implemented by subclasses.

    Attributes:
        model_id (str): Identifier or path for the pretrained model.
        model_class (type): Hugging Face model class to instantiate.
        torch_dtype (torch.dtype): Desired data type for model weights and tensors.
        use_gpu (bool): User's preference to use GPU if available.
        device (torch.device): The device the model is (or will be) loaded onto.
        model (torch.nn.Module, optional): The loaded model instance.
    """

    def __init__(
        self,
        model_id: str,
        model_class: type,
        torch_dtype: torch.dtype = torch.float32,
        use_gpu: bool = True
    ) -> None:
        """
        Initialize the ModelHelper.

        Args:
            model_id (str): Model identifier or local path for pretrained weights.
            model_class (type): Hugging Face model class to load.
            torch_dtype (torch.dtype, optional): Data type for model weights. Defaults to torch.float32.
            use_gpu (bool, optional): Whether to use GPU if available. Defaults to True.
        """
        self.model_id: str = model_id
        self.model_class: type = model_class
        self.torch_dtype: torch.dtype = torch_dtype
        self.use_gpu: bool = use_gpu
        self.device: torch.device = self._get_target_device()
        self.model: Optional[torch.nn.Module] = None

    @property
    def _is_gpu_available(self) -> bool:
        """Check if a CUDA-capable GPU is available on the system."""
        return torch.cuda.is_available()

    def _get_target_device(self) -> torch.device:
        """Determine the target device based on availability and user preference."""
        if self._is_gpu_available and self.use_gpu:
            return torch.device("cuda")
        return torch.device("cpu")

    def load_model(self) -> None:
        """
        Load the pretrained model directly onto the target device.

        Raises:
            Exception: Propagates exceptions raised during model loading.
        """
        try:
            if self._is_gpu_available and self.use_gpu:
                # For GPU, use device_map="auto" for transformers to handle
                # potential multi-GPU or offloading.
                self.model = self.model_class.from_pretrained(
                    self.model_id,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    device_map="auto" # "auto" will place on GPU
                )
            else:
                # For CPU, explicitly set device_map to "cpu"
                self.model = self.model_class.from_pretrained(
                    self.model_id,
                    torch_dtype=self.torch_dtype, # Use float32 for CPU
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    device_map="cpu"
                )
            
            # Ensure model is in eval mode
            self.model.eval()
            
            # device attribute might be set by device_map, let's confirm
            if not hasattr(self.model, 'device'):
                 self.model.to(self.device)
            else:
                 self.device = self.model.device # Get actual device from model

            print(f"Model loaded on {self.device} with dtype {self.torch_dtype}")

        except Exception as exc:
            print(f"Error loading model: {exc}")
            raise

    def run(self, inputs: Any) -> Any:
        """
        Abstract method for performing inference with the model.

        Subclasses must override this method with task-specific inference logic.

        Args:
            inputs (Any): Input data for the model inference.

        Raises:
            NotImplementedError: If called directly from this base class.
        """
        raise NotImplementedError("Subclasses must implement run(inputs)")