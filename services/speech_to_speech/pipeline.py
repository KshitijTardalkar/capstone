"""
Main Speech-to-Speech Pipeline Orchestrator.

This file defines the `SpeechToSpeechPipeline` class, which is responsible
for initializing, loading, and managing the state of all AI models
(STT, LLM, TTS). It handles the logic for loading and unloading models
from the GPU to conserve VRAM and provides a central point of control.
"""

import torch
from typing import Optional, Union
import os
import gc
import io

from .stt import stt
from .llm import LLM
from .tts import TTS

class SpeechToSpeechPipeline:
    """
    Orchestrates the loading and management of STT, LLM, and TTS models.

    This class holds references to the individual model services and manages
    their state, including which models are loaded, their hardware preferences
    (GPU, quantization, etc.), and clearing services to free VRAM.
    """
    def __init__(self) -> None:
        """Initializes the pipeline container and default settings."""
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print("="*60)
        print("SPEECH-TO-SPEECH PIPELINE CONTAINER INITIALIZED")
        print("="*60)

        self.stt_service: Optional[stt] = None
        self.llm_service: Optional[LLM] = None
        self.tts_service: Optional[TTS] = None

        self.current_stt_model_id: Optional[str] = None
        self.current_llm_model_id: Optional[str] = None
        self.current_tts_model_id: Optional[str] = None
        self.stt_gpu_pref: Optional[bool] = None
        self.llm_gpu_pref: Optional[bool] = None
        self.tts_gpu_pref: Optional[bool] = None
        self.llm_quantize_pref: Optional[str] = None
        self.stt_flash_attn_pref: Optional[bool] = None
        self.llm_flash_attn_pref: Optional[bool] = None

    def _clear_service(self, service_name: str):
        """
        Safely deletes a model service and clears CUDA cache.

        Args:
            service_name (str): The attribute name of the service to clear
                                (e.g., 'stt_service').
        """
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
        tts_model_id: str,
        stt_use_gpu: bool,
        llm_use_gpu: bool,
        tts_use_gpu: bool,
        llm_quantize: str = 'none',
        stt_use_flash_attn: bool = False,
        llm_use_flash_attn: bool = False
    ) -> None:
        """
        Loads all models in the pipeline based on user configuration.

        This method intelligently checks if models or their configurations
        have changed before reloading them, saving time and resources.

        Args:
            stt_model_id (str): The Hugging Face ID for the STT model.
            llm_model_id (str): The Hugging Face ID for the LLM.
            tts_model_id (str): The Hugging Face ID for the TTS model.
            stt_use_gpu (bool): Whether the STT model should use the GPU.
            llm_use_gpu (bool): Whether the LLM should use the GPU.
            tts_use_gpu (bool): Whether the TTS model should use the GPU.
            llm_quantize (str): The quantization mode for the LLM (and STT).
            stt_use_flash_attn (bool): Whether the STT model should use Flash Attention 2.
            llm_use_flash_attn (bool): Whether the LLM should use Flash Attention 2.
        """
        
        print("\n" + "="*60)
        print("CHECKING AND LOADING PIPELINE MODELS...")
        print("="*60)

        stt_changed = (self.current_stt_model_id != stt_model_id or 
                       self.stt_gpu_pref != stt_use_gpu or
                       self.stt_flash_attn_pref != stt_use_flash_attn or
                       self.llm_quantize_pref != llm_quantize) # STT quant is tied to LLM quant

        if self.stt_service is None or stt_changed:
            print(f"Step 1/3: Loading STT service with {stt_model_id} (GPU: {stt_use_gpu}, FA2: {stt_use_flash_attn}, Quant: {llm_quantize})...")
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
            print(f"\nStep 2/3: Loading LLM service with {llm_model_id} (GPU: {llm_use_gpu}, Quant: {llm_quantize}, FA2: {llm_use_flash_attn})...")
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


        tts_changed = (self.current_tts_model_id != tts_model_id or
                       self.tts_gpu_pref != tts_use_gpu)
                       
        if self.tts_service is None or tts_changed:
            print(f"\nStep 3/3: Loading TTS service with {tts_model_id} (GPU: {tts_use_gpu})...")
            self._clear_service('tts_service')
            self.tts_service = TTS(
                tts_model_id, 
                torch_dtype=self.torch_dtype, 
                use_gpu=tts_use_gpu,
                use_flash_attn=False 
            )
            self.tts_service.load_model()
            self.current_tts_model_id = tts_model_id
            self.tts_gpu_pref = tts_use_gpu
            print("✓ TTS service ready")
        else:
            print("✓ TTS service already loaded.")

        print("\n" + "="*60)
        print("PIPELINE MODEL LOADING COMPLETE")
        print("="*60)

    def models_loaded(self) -> bool:
        """Returns True if all three model services are loaded, False otherwise."""
        return all([
            self.stt_service and self.stt_service.model is not None,
            self.llm_service and self.llm_service.model is not None,
            self.tts_service and self.tts_service.model is not None
        ])

    def clear_conversation_memory(self) -> None:
        """Clears the LLM's conversational history."""
        if self.llm_service:
            self.llm_service.clear_memory()
            print("Conversation memory cleared.")
        else:
            print("LLM service not initialized. Cannot clear memory.")

    def get_system_info(self) -> dict:
        """
        Gathers status information from all loaded models for the frontend.

        Returns:
            dict: A dictionary containing model names and device placements.
        """
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

        tts_device = str(self.tts_service.device) if self.tts_service else "N/A"
        if self.tts_service and self.tts_service.device.type == 'cuda':
             tts_device = f"{tts_device} (Compiled)"


        return {
            "stt_model": self.stt_service.model_id if self.stt_service else "Not Loaded",
            "stt_device": stt_device,
            "llm_model": self.llm_service.model_id if self.llm_service else "Not Loaded", 
            "llm_device": llm_device,
            "tts_model": self.tts_service.model_id if self.tts_service else "Not Loaded",
            "tts_device": tts_device,
            "torch_dtype": str(self.torch_dtype),
            "gpu_available": torch.cuda.is_available(),
            "models_loaded": self.models_loaded()
        }