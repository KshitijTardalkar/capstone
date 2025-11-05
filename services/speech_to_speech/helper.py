import torch
import gc
from typing import Optional, Any


class ModelHelper:
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
        return torch.cuda.is_available()

    def _get_target_device(self) -> torch.device:
        if self._is_gpu_available and self.use_gpu:
            return torch.device("cuda")
        return torch.device("cpu")

    def load_model(self) -> None:
        model_kwargs = {
            "low_cpu_mem_usage": True,
            "use_safetensors": True
        }

        if self._is_gpu_available and self.use_gpu:
            model_kwargs["device_map"] = "auto"
            if self.use_flash_attn:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                model_kwargs["torch_dtype"] = torch.float16 
                self.torch_dtype = torch.float16
                print("Enabling Flash Attention 2 (forcing torch.float16)")
            else:
                model_kwargs["torch_dtype"] = self.torch_dtype
        else:
            model_kwargs["device_map"] = "cpu"
            model_kwargs["torch_dtype"] = torch.float32 
            self.torch_dtype = torch.float32

        try:
            self.model = self.model_class.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            
            self.model.eval()
            
            if hasattr(self.model, 'device'):
                 self.device = self.model.device
            
            print(f"Model {self.model_id} loaded on {self.device} with dtype {self.torch_dtype}")

        except Exception as exc:
            print(f"Error loading model {self.model_id}: {exc}")
            if "flash_attention_2" in str(exc):
                print("!!! Flash Attention 2 failed to load. The model may not support it, or your GPU/driver is incompatible. !!!")
            raise

    def run(self, inputs: Any, **kwargs) -> Any:
        raise NotImplementedError("Subclasses must implement run(inputs)")