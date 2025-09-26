import torch
from typing import Any

from .stt import STT
from .llm import LLM
from .tts import TTS
#IoTDataManger will have our simulated data 
from ..iot_simulator import IoTDataManager

class SpeechToSpeechPipeline:

    def __init__(self, stt_model_id: str, llm_model_id: str, tts_model_id: str, tts_model_id: str):
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print("Initializing STT service...")
        self.stt_service = STT(stt_model_id, torch_dtype=self.torch_dtype)
        self.stt_service.load_model()
        
        print("\nInitializing LLM service...")
        self.llm_service = LLM(llm_model_id, torch_dtype=self.torch_dtype)
        self.llm_service.load_model()

        print("\nInitializing TTS service...")
        self.tts_service = TTS(tts_model_id, tts_vocoder_id, torch_dtype=self.torch_dtype)
        self.tts_service.load_model()

        self.data_manager=data_manager

        print("\nPipeline initialized successfully.")

    def run(self, audio_file_path: str) -> str:
        print("-"*50)

        print("Step 1. Transcribing audio input...")
        user_query=self.stt_service.run_and_cleanup(audio_file_path)

        if not user_query:
            raise Exception("STT falied to produce a transcript.")
        print(f"  > Transcribed Text: `{user_query}`")

        print("Step 2. Retrieving IoT data...")

        iot_response=self.data_manager.get_data(user_query)

        print(f"   > IoT Data: {iot_response['data']}")

        print("Step 3. Formulating response prompt for LLM...")

        prompt = (
            f"Given the user's question: '{user_query}', "
            f"and the retrieved system data: '{iot_response['data']}'. "
            "Formulate a direct and concise response for the industrial operator."
        )

        print("Step 4. Generating text response...")
        llm_response = self.llm_service.run_and_cleanup(prompt)
        if not llm_response:
            raise Exception("LLM failed to generate a response.")
        print(f"   > LLM Response: '{llm_response}'")

        print("Step 5. Synthesizing audio response...")
        output_audio_path = self.tts_service.run_and_cleanup(llm_response, "response.wav")
        if not output_audio_path:
            raise Exception("TTS failed to generate audio.")
        print(f"   > Response audio saved to: {output_audio_path}")
        print("-" * 50)
        
        return output_audio_path