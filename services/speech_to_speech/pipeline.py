import torch
from typing import Optional, Union
import os
import gc
import io

from .stt import stt
from .llm import LLM
from .tts import TTS

class SpeechToSpeechPipeline:
    def __init__(self) -> None:
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
        
        print("\n" + "="*60)
        print("CHECKING AND LOADING PIPELINE MODELS...")
        print("="*60)

        stt_changed = (self.current_stt_model_id != stt_model_id or 
                       self.stt_gpu_pref != stt_use_gpu or
                       self.stt_flash_attn_pref != stt_use_flash_attn)

        if self.stt_service is None or stt_changed:
            print(f"Step 1/3: Loading STT service with {stt_model_id} (GPU: {stt_use_gpu}, FA2: {stt_use_flash_attn})...")
            self._clear_service('stt_service')
            self.stt_service = stt(
                stt_model_id, 
                torch_dtype=self.torch_dtype, 
                use_gpu=stt_use_gpu,
                use_flash_attn=stt_use_flash_attn
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
        return all([
            self.stt_service and self.stt_service.model is not None,
            self.llm_service and self.llm_service.model is not None,
            self.tts_service and self.tts_service.model is not None
        ])

    def run(self, audio_file_path: Union[str, io.BytesIO], output_dir: str = "output") -> Optional[str]:
        if not self.models_loaded():
            print("❌ Error: Pipeline models are not loaded. Call load_models() first.")
            return None

        print("\n" + "="*50)
        print("STARTING SPEECH-TO-SPEECH PIPELINE")
        print("="*50)

        if isinstance(audio_file_path, str) and not os.path.exists(audio_file_path):
            print(f"❌ Error: Audio file not found at {audio_file_path}")
            return None

        os.makedirs(output_dir, exist_ok=True)

        try:
            print("Step 1/5: Transcribing audio input...")
            user_query = self.stt_service.run(audio_file_path)

            if not user_query or not user_query.strip():
                print("❌ STT failed to produce a valid transcript.")
                return None
            
            if isinstance(user_query, list):
                user_query = user_query[0] if user_query else ""
                
            user_query = user_query.strip()
            print(f"✓ Transcribed Text: '{user_query}'")

            print("\nStep 3/5: Formulating response prompt for LLM...")
            prompt = (
                f"User Input: {user_query}\n\n"
                " Identify the speech"
            )
            print("✓ Prompt formulated")

            print("\nStep 4/5: Generating response with LLM...")
            llm_response = self.llm_service.run(prompt, max_new_tokens=100)
            
            if not llm_response or not llm_response.strip():
                print("❌ LLM failed to generate a valid response.")
                return None
                
            llm_response = llm_response.strip()
            print(f"✓ LLM Response: '{llm_response}'")

            print("\nStep 5/5: Synthesizing audio response...")
            output_filename = f"response_{hash(user_query) % 10000}.wav"
            output_audio_path = os.path.join(output_dir, output_filename)
            
            tts_audio_bytes = self.tts_service.run(llm_response)
            
            if not tts_audio_bytes:
                print("❌ TTS failed to generate audio file.")
                return None
            
            with open(output_audio_path, 'wb') as f:
                f.write(tts_audio_bytes)
                
            print(f"✓ Response audio saved to: {output_audio_path}")

            print("\n" + "="*50)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*50)
            
            return output_audio_path

        except Exception as e:
            print(f"\n❌ Pipeline Error: {e}")
            print("="*50)
            return None

    def clear_conversation_memory(self) -> None:
        if self.llm_service:
            self.llm_service.clear_memory()
            print("Conversation memory cleared.")
        else:
            print("LLM service not initialized. Cannot clear memory.")

    def get_system_info(self) -> dict:
        stt_device = str(self.stt_service.device) if self.stt_service else "N/A"
        if self.stt_service and self.stt_flash_attn_pref:
             stt_device = f"{stt_device} (FA2)"

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
            "tts_model": self.tts_service.model_id if self.tts_service else "Not Loaded",
            "tts_device": str(self.tts_service.device) if self.tts_service else "N/A",
            "torch_dtype": str(self.torch_dtype),
            "gpu_available": torch.cuda.is_available(),
            "models_loaded": self.models_loaded()
        }

    def test_pipeline_components(self) -> dict:
        if not self.models_loaded():
            return {
                "stt": False, "llm": False, "tts": False,
                "error": "Models not loaded."
            }
            
        results = { "stt": False, "llm": False, "tts": False }

        try:
            if self.stt_service.model is not None:
                results["stt"] = True
        except Exception as e:
            print(f"STT test failed: {e}")

        try:
            if self.llm_service.model is not None:
                test_response = self.llm_service.run("Test", max_new_tokens=10)
                if test_response:
                    results["llm"] = True
        except Exception as e:
            print(f"LLM test failed: {e}")

        try:
            if self.tts_service.model is not None:
                results["tts"] = True
        except Exception as e:
            print(f"TTS test failed: {e}")

        return results