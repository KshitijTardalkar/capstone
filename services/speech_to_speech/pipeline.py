import torch
from typing import Optional
import os

from .stt import stt
from .llm import LLM
from .tts import TTS
# IoTDataManager will have our simulated data 
#from ..iot_simulator import IoTDataManager


class SpeechToSpeechPipeline:
    """
    Complete Speech-to-Speech pipeline for industrial voice assistance.

    Orchestrates the entire workflow from audio input to audio output:
    1. Speech-to-Text (STT) - Convert user's audio query to text
    2. Large Language Model (LLM) - Generate contextual response
    3. Text-to-Speech (TTS) - Convert response text back to audio

    This pipeline manages model loading and device placement (CPU/GPU)
    based on user preferences.

    Attributes:
        stt_service (stt): Speech-to-text service instance
        llm_service (LLM): Large language model service instance  
        tts_service (TTS): Text-to-speech service instance
        torch_dtype (torch.dtype): Tensor dtype based on GPU availability
    """

    def __init__(self) -> None:
        """
        Initialize the speech-to-speech pipeline container.

        Models are not loaded at construction. Call `load_models()` to load them.
        """
        # Set appropriate dtype based on hardware
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print("="*60)
        print("SPEECH-TO-SPEECH PIPELINE CONTAINER INITIALIZED")
        print("="*60)

        # Initialize services as None. They will be loaded by load_models()
        self.stt_service: Optional[stt] = None
        self.llm_service: Optional[LLM] = None
        self.tts_service: Optional[TTS] = None

        # Track currently loaded models and their settings
        self.current_stt_model_id: Optional[str] = None
        self.current_llm_model_id: Optional[str] = None
        self.current_tts_model_id: Optional[str] = None
        self.stt_gpu_pref: Optional[bool] = None
        self.llm_gpu_pref: Optional[bool] = None
        self.tts_gpu_pref: Optional[bool] = None

    def load_models(
        self, 
        stt_model_id: str, 
        llm_model_id: str, 
        tts_model_id: str,
        stt_use_gpu: bool,
        llm_use_gpu: bool,
        tts_use_gpu: bool
    ) -> None:
        """
        Load or reload models as needed.
        
        Only loads a model if its requested model_id or GPU preference
        is different from the currently loaded one.
        """
        
        print("\n" + "="*60)
        print("CHECKING AND LOADING PIPELINE MODELS...")
        print("="*60)

        # --- Check STT service ---
        stt_changed = (self.current_stt_model_id != stt_model_id or 
                       self.stt_gpu_pref != stt_use_gpu)

        if self.stt_service is None or stt_changed:
            print(f"Step 1/3: Loading STT service with {stt_model_id} (GPU: {stt_use_gpu})...")
            # (Optional: Add cleanup for old model if it exists)
            # if self.stt_service: del self.stt_service.model
            self.stt_service = stt(stt_model_id, torch_dtype=self.torch_dtype, use_gpu=stt_use_gpu)
            self.stt_service.load_model()
            self.current_stt_model_id = stt_model_id
            self.stt_gpu_pref = stt_use_gpu
            print("✓ STT service ready")
        else:
            print("✓ STT service already loaded.")

        
        # --- Check LLM service ---
        llm_changed = (self.current_llm_model_id != llm_model_id or
                       self.llm_gpu_pref != llm_use_gpu)

        if self.llm_service is None or llm_changed:
            print(f"\nStep 2/3: Loading LLM service with {llm_model_id} (GPU: {llm_use_gpu})...")
            # (Optional: Add cleanup for old model if it exists)
            # if self.llm_service: del self.llm_service.model
            self.llm_service = LLM(llm_model_id, torch_dtype=self.torch_dtype, use_gpu=llm_use_gpu)
            self.llm_service.load_model()
            self.current_llm_model_id = llm_model_id
            self.llm_gpu_pref = llm_use_gpu
            print("✓ LLM service ready")
        else:
            print("✓ LLM service already loaded.")

        # --- Check TTS service ---
        tts_changed = (self.current_tts_model_id != tts_model_id or
                       self.tts_gpu_pref != tts_use_gpu)
                       
        if self.tts_service is None or tts_changed:
            print(f"\nStep 3/3: Loading TTS service with {tts_model_id} (GPU: {tts_use_gpu})...")
            # (Optional: Add cleanup for old model if it exists)
            # if self.tts_service: del self.tts_service.model
            self.tts_service = TTS(tts_model_id, torch_dtype=self.torch_dtype, use_gpu=tts_use_gpu)
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
        """Check if all services are initialized and have models."""
        return all([
            self.stt_service and self.stt_service.model is not None,
            self.llm_service and self.llm_service.model is not None,
            self.tts_service and self.tts_service.model is not None
        ])

    def run(self, audio_file_path: str, output_dir: str = "output") -> Optional[str]:
        """
        Execute the complete speech-to-speech pipeline.
        (For standalone execution, like in main.py)
        """
        if not self.models_loaded():
            print("❌ Error: Pipeline models are not loaded. Call load_models() first.")
            return None

        print("\n" + "="*50)
        print("STARTING SPEECH-TO-SPEECH PIPELINE")
        print("="*50)

        if not os.path.exists(audio_file_path):
            print(f"❌ Error: Audio file not found at {audio_file_path}")
            return None

        os.makedirs(output_dir, exist_ok=True)

        try:
            # Step 1: Speech-to-Text
            print("Step 1/5: Transcribing audio input...")
            user_query = self.stt_service.run(audio_file_path) # Use .run()

            if not user_query or not user_query.strip():
                print("❌ STT failed to produce a valid transcript.")
                return None
            
            if isinstance(user_query, list):
                user_query = user_query[0] if user_query else ""
                
            user_query = user_query.strip()
            print(f"✓ Transcribed Text: '{user_query}'")

            # Step 2: IoT Data Retrieval (Commented out)
            # ...

            # Step 3: Prompt Formation
            print("\nStep 3/5: Formulating response prompt for LLM...")
            prompt = (
                f"User Input: {user_query}\n\n"
                " Identify the speech"
            )
            print("✓ Prompt formulated")

            # Step 4: LLM Response Generation
            print("\nStep 4/5: Generating response with LLM...")
            llm_response = self.llm_service.run(prompt, max_new_tokens=100) # Use .run()
            
            if not llm_response or not llm_response.strip():
                print("❌ LLM failed to generate a valid response.")
                return None
                
            llm_response = llm_response.strip()
            print(f"✓ LLM Response: '{llm_response}'")

            # Step 5: Text-to-Speech Synthesis
            print("\nStep 5/5: Synthesizing audio response...")
            output_filename = f"response_{hash(user_query) % 10000}.wav"
            output_audio_path = os.path.join(output_dir, output_filename)
            
            final_audio_path = self.tts_service.run( # Use .run()
                llm_response, 
                output_audio_path
            )
            
            if not final_audio_path or not os.path.exists(final_audio_path):
                print("❌ TTS failed to generate audio file.")
                return None
                
            print(f"✓ Response audio saved to: {final_audio_path}")

            print("\n" + "="*50)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*50)
            
            return final_audio_path

        except Exception as e:
            print(f"\n❌ Pipeline Error: {e}")
            print("="*50)
            return None

    def clear_conversation_memory(self) -> None:
        """
        Clear the LLM's conversation memory for fresh interactions.
        """
        if self.llm_service:
            self.llm_service.clear_memory()
            print("Conversation memory cleared.")
        else:
            print("LLM service not initialized. Cannot clear memory.")

    def get_system_info(self) -> dict:
        """
        Get information about the current pipeline configuration.
        """
        return {
            "stt_model": self.stt_service.model_id if self.stt_service else "Not Loaded",
            "stt_device": str(self.stt_service.device) if self.stt_service else "N/A",
            "llm_model": self.llm_service.model_id if self.llm_service else "Not Loaded", 
            "llm_device": str(self.llm_service.device) if self.llm_service else "N/A",
            "tts_model": self.tts_service.model_id if self.tts_service else "Not Loaded",
            "tts_device": str(self.tts_service.device) if self.tts_service else "N/A",
            "torch_dtype": str(self.torch_dtype),
            "gpu_available": torch.cuda.is_available(),
            "models_loaded": self.models_loaded()
        }

    def test_pipeline_components(self) -> dict:
        """
        Test each pipeline component individually for debugging.
        """
        if not self.models_loaded():
            return {
                "stt": False, "llm": False, "tts": False,
                "error": "Models not loaded."
            }
            
        results = {
            "stt": False,
            "llm": False, 
            "tts": False,
            # "data_manager": False # Disabled
        }

        # Test STT
        try:
            if self.stt_service.model is not None:
                results["stt"] = True
        except Exception as e:
            print(f"STT test failed: {e}")

        # Test LLM
        try:
            if self.llm_service.model is not None:
                test_response = self.llm_service.run("Test", max_new_tokens=10)
                if test_response:
                    results["llm"] = True
        except Exception as e:
            print(f"LLM test failed: {e}")

        # Test TTS
        try:
            if self.tts_service.model is not None:
                results["tts"] = True
        except Exception as e:
            print(f"TTS test failed: {e}")

        # Test Data Manager (Disabled)
        # ...

        return results