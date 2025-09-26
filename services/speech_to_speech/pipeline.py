import torch
from typing import Optional
import os

from .stt import stt  # Note: your class name is lowercase 'stt'
from .llm import LLM
from .tts import TTS
# IoTDataManager will have our simulated data 
#from ..iot_simulator import IoTDataManager


class SpeechToSpeechPipeline:
    """
    Complete Speech-to-Speech pipeline for industrial voice assistance.

    Orchestrates the entire workflow from audio input to audio output:
    1. Speech-to-Text (STT) - Convert user's audio query to text
    2. IoT Data Retrieval - Fetch relevant system data based on query
    3. Large Language Model (LLM) - Generate contextual response
    4. Text-to-Speech (TTS) - Convert response text back to audio

    This pipeline is designed for industrial control room environments where
    operators need hands-free access to system information and controls.

    Attributes:
        stt_service (stt): Speech-to-text service instance
        llm_service (LLM): Large language model service instance  
        tts_service (TTS): Text-to-speech service instance
        data_manager (IoTDataManager): IoT data retrieval service
        torch_dtype (torch.dtype): Tensor dtype based on GPU availability
    """

    def __init__(
        self, 
        stt_model_id: str, 
        llm_model_id: str, 
        tts_model_id: str, 
        # tts_vocoder_id: str,
        #data_manager: IoTDataManager
    ) -> None:
        """
        Initialize the complete speech-to-speech pipeline.

        Args:
            stt_model_id (str): Hugging Face model ID for speech-to-text
                               (e.g., "openai/whisper-base")
            llm_model_id (str): Hugging Face model ID for language model
                               (e.g., "microsoft/DialoGPT-medium")  
            tts_model_id (str): Hugging Face model ID for text-to-speech
                               (e.g., "microsoft/speecht5_tts")
            tts_vocoder_id (str): Hugging Face model ID for TTS vocoder
                                 (e.g., "microsoft/speecht5_hifigan")
            data_manager (IoTDataManager): Initialized IoT data management instance

        Notes:
            - Uses float16 precision on GPU for memory efficiency
            - Falls back to float32 on CPU for compatibility
            - All models are loaded during initialization for faster runtime
        """
        # Set appropriate dtype based on hardware
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print("="*60)
        print("INITIALIZING SPEECH-TO-SPEECH PIPELINE")
        print("="*60)

        # Initialize STT service
        print("Step 1/4: Initializing Speech-to-Text (STT) service...")
        self.stt_service = stt(stt_model_id, torch_dtype=self.torch_dtype)
        self.stt_service.load_model()
        print("✓ STT service ready")
        
        # Initialize LLM service
        print("\nStep 2/4: Initializing Large Language Model (LLM) service...")
        self.llm_service = LLM(llm_model_id, torch_dtype=self.torch_dtype)
        self.llm_service.load_model()
        print("✓ LLM service ready")

        # Initialize TTS service
        print("\nStep 3/4: Initializing Text-to-Speech (TTS) service...")
        self.tts_service = TTS(tts_model_id, torch_dtype=self.torch_dtype)
        self.tts_service.load_model()
        print("✓ TTS service ready")

        # Set up data manager
        # print("\nStep 4/4: Setting up IoT Data Manager...")
        # self.data_manager = data_manager
        # print("✓ Data manager ready")

        print("\n" + "="*60)
        print("PIPELINE INITIALIZATION COMPLETE")
        print("="*60)

    def run(self, audio_file_path: str, output_dir: str = "output") -> Optional[str]:
        """
        Execute the complete speech-to-speech pipeline.

        Args:
            audio_file_path (str): Path to input audio file containing user query
            output_dir (str): Directory to save output audio file. Defaults to "output".

        Returns:
            Optional[str]: Path to generated response audio file if successful,
                          None if any step fails.

        Raises:
            Exception: If any pipeline step fails critically.

        Workflow:
            1. Validate input audio file exists
            2. Transcribe audio to text using STT
            3. Retrieve relevant IoT data based on query
            4. Generate contextual response using LLM
            5. Synthesize response audio using TTS
            6. Return path to generated audio file
        """
        print("\n" + "="*50)
        print("STARTING SPEECH-TO-SPEECH PIPELINE")
        print("="*50)

        # Validate input file
        if not os.path.exists(audio_file_path):
            print(f"❌ Error: Audio file not found at {audio_file_path}")
            return None

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Step 1: Speech-to-Text
            print("Step 1/5: Transcribing audio input...")
            user_query = self.stt_service.run_and_cleanup(audio_file_path)

            if not user_query or not user_query.strip():
                print("❌ STT failed to produce a valid transcript.")
                return None
            
            # Handle list return from batch_decode
            if isinstance(user_query, list):
                user_query = user_query[0] if user_query else ""
                
            user_query = user_query.strip()
            print(f"✓ Transcribed Text: '{user_query}'")

            # Step 2: IoT Data Retrieval
            # print("\nStep 2/5: Retrieving IoT system data...")
            # iot_response = self.data_manager.get_data(user_query)

            # if not iot_response or 'data' not in iot_response:
            #     print("⚠️  Warning: No IoT data retrieved, proceeding with general response.")
            #     iot_data = "No specific system data available."
            # else:
            #     iot_data = iot_response['data']
                
            # print(f"✓ IoT Data Retrieved: {iot_data}")

            # Step 3: Prompt Formation
            print("\nStep 3/5: Formulating response prompt for LLM...")
            # prompt = (
            #     f"User Question: {user_query}\n\n"
            #     f"System Data: {iot_data}\n\n"
            #     "Please provide a clear, concise response for the industrial operator "
            #     "based on the user's question and the available system data. "
            #     "Focus on actionable information and safety considerations."
            # )
            prompt = (
                f"User Input: {user_query}\n\n"
                " Identify the speech"
            )
            print("✓ Prompt formulated")

            # Step 4: LLM Response Generation
            print("\nStep 4/5: Generating response with LLM...")
            llm_response = self.llm_service.run_and_cleanup(prompt, max_new_tokens=100)
            
            if not llm_response or not llm_response.strip():
                print("❌ LLM failed to generate a valid response.")
                return None
                
            llm_response = llm_response.strip()
            print(f"✓ LLM Response: '{llm_response}'")

            # Step 5: Text-to-Speech Synthesis
            print("\nStep 5/5: Synthesizing audio response...")
            output_filename = f"response_{hash(user_query) % 10000}.wav"
            output_audio_path = os.path.join(output_dir, output_filename)
            
            final_audio_path = self.tts_service.run_and_cleanup(
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
        
        Useful for resetting the conversation context or preventing
        memory buildup in long-running sessions.
        """
        self.llm_service.clear_memory()
        print("Conversation memory cleared.")

    def get_system_info(self) -> dict:
        """
        Get information about the current pipeline configuration.

        Returns:
            dict: Configuration details including model IDs, device info, and status.
        """
        return {
            "stt_model": self.stt_service.model_id,
            "llm_model": self.llm_service.model_id, 
            "tts_model": self.tts_service.model_id,
            "device_type": "GPU" if torch.cuda.is_available() else "CPU",
            "torch_dtype": str(self.torch_dtype),
            "gpu_available": torch.cuda.is_available(),
            "models_loaded": all([
                self.stt_service.model is not None,
                self.llm_service.model is not None,
                self.tts_service.model is not None
            ])
        }

    def test_pipeline_components(self) -> dict:
        """
        Test each pipeline component individually for debugging.

        Returns:
            dict: Test results for each component.
        """
        results = {
            "stt": False,
            "llm": False, 
            "tts": False,
            "data_manager": False
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

        # Test Data Manager
        try:
            test_data = self.data_manager.get_data("test query")
            if test_data:
                results["data_manager"] = True
        except Exception as e:
            print(f"Data Manager test failed: {e}")

        return results