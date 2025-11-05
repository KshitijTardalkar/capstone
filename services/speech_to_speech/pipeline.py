import torch
from typing import Optional, Union
import os
import gc
import io

from .stt import stt
from .llm import LLM

class SpeechToSpeechPipeline:
    def __init__(self) -> None:
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print("="*60)
        print("SPEECH-TO-COMMAND PIPELINE CONTAINER INITIALIZED")
        print("="*60)

        self.stt_service: Optional[stt] = None
        self.llm_service: Optional[LLM] = None

        self.current_stt_model_id: Optional[str] = None
        self.current_llm_model_id: Optional[str] = None
        self.stt_gpu_pref: Optional[bool] = None
        self.llm_gpu_pref: Optional[bool] = None
        self.llm_quantize_pref: Optional[str] = None
        self.stt_flash_attn_pref: Optional[bool] = None
        self.llm_flash_attn_pref: Optional[bool] = None

    def _clear_service(self, service_name: str):
        service = getattr(self, service_name)
        if service is not None:
            if hasattr(service, 'model') and service.model is not None:
                del service.model
            del service
            setattr(self, service_name, None)
            
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Cleaned up old {service_name} and cleared CUDA cache.")

    def load_models(
        self, 
        stt_model_id: str, 
        llm_model_id: str, 
        stt_use_gpu: bool,
        llm_use_gpu: bool,
        llm_quantize: str = 'none',
        stt_use_flash_attn: bool = False,
        llm_use_flash_attn: bool = False
    ) -> None:
        
        print("\n" + "="*60)
        print("CHECKING AND LOADING PIPELINE MODELS...")
        print("="*60)

        stt_changed = (self.current_stt_model_id != stt_model_id or 
                       self.stt_gpu_pref != stt_use_gpu or
                       self.stt_flash_attn_pref != stt_use_flash_attn or
                       self.llm_quantize_pref != llm_quantize)

        if self.stt_service is None or stt_changed:
            print(f"Step 1/2: Loading STT service with {stt_model_id} (GPU: {stt_use_gpu}, FA2: {stt_use_flash_attn}, Quant: {llm_quantize})...")
            self._clear_service('stt_service')
            self.stt_service = stt(
                stt_model_id, 
                torch_dtype=self.torch_dtype, 
                use_gpu=stt_use_gpu,
                use_flash_attn=stt_use_flash_attn,
                quantize=llm_quantize
            )
            self.stt_service.load_model()
            self.current_stt_model_id = stt_model_id
            self.stt_gpu_pref = stt_use_gpu
            self.stt_flash_attn_pref = stt_use_flash_attn
            print("✓ STT service ready")
        else:
            print("✓ STT service already loaded.")

        
        llm_changed = (self.current_llm_model_id != llm_model_id or
                       self.llm_gpu_pref != llm_use_gpu or
                       self.llm_quantize_pref != llm_quantize or
                       self.llm_flash_attn_pref != llm_use_flash_attn)

        if self.llm_service is None or llm_changed:
            print(f"\nStep 2/2: Loading LLM service with {llm_model_id} (GPU: {llm_use_gpu}, Quant: {llm_quantize}, FA2: {llm_use_flash_attn})...")
            self._clear_service('llm_service')
            self.llm_service = LLM(
                llm_model_id, 
                torch_dtype=self.torch_dtype, 
                use_gpu=llm_use_gpu, 
                quantize=llm_quantize,
                use_flash_attn=llm_use_flash_attn
            )
            self.llm_service.load_model()
            self.current_llm_model_id = llm_model_id
            self.llm_gpu_pref = llm_use_gpu
            self.llm_quantize_pref = llm_quantize
            self.llm_flash_attn_pref = llm_use_flash_attn
            print("✓ LLM service ready")
        else:
            if self.llm_service:
                self.llm_service.clear_memory()
                print("✓ LLM service already loaded. Cleared conversation memory.")
            else:
                print("✓ LLM service already loaded.")

        print("\n" + "="*60)
        print("PIPELINE MODEL LOADING COMPLETE")
        print("="*60)

    def models_loaded(self) -> bool:
        return all([
            self.stt_service and self.stt_service.model is not None,
            self.llm_service and self.llm_service.model is not None
        ])

    def clear_conversation_memory(self) -> None:
        if self.llm_service:
            self.llm_service.clear_memory()
            print("Conversation memory cleared.")
        else:
            print("LLM service not initialized. Cannot clear memory.")

    def get_system_info(self) -> dict:
        stt_device = str(self.stt_service.device) if self.stt_service else "N/A"
        if self.stt_service:
            stt_tags = []
            if self.llm_quantize_pref != 'none':
                stt_tags.append("8-bit")
            if self.stt_flash_attn_pref:
                stt_tags.append("FA2")
            if stt_tags:
                stt_device = f"{stt_device} ({', '.join(stt_tags)})"

        llm_device = str(self.llm_service.device) if self.llm_service else "N/A"
        if self.llm_service:
            llm_tags = []
            if self.llm_quantize_pref != 'none':
                llm_tags.append(self.llm_quantize_pref)
            if self.llm_flash_attn_pref:
                llm_tags.append("FA2")
            if llm_tags:
                llm_device = f"{llm_device} ({', '.join(llm_tags)})"

        return {
            "stt_model": self.stt_service.model_id if self.stt_service else "Not Loaded",
            "stt_device": stt_device,
            "llm_model": self.llm_service.model_id if self.llm_service else "Not Loaded", 
            "llm_device": llm_device,
            "torch_dtype": str(self.torch_dtype),
            "gpu_available": torch.cuda.is_available(),
            "models_loaded": self.models_loaded()
        }