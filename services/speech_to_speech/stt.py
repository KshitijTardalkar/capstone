import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from typing import Optional

from services.speech_to_speech.helper import ModelHelper
from services.speech_to_speech.audio_utils import load_audio_file


class stt(ModelHelper):
    """
    Speech-to-Text (STT) wrapper around Hugging Face Whisper models.
    """

    def __init__(self, model_id: str, torch_dtype: torch.dtype = torch.float32) -> None:
        super().__init__(model_id, AutoModelForSpeechSeq2Seq, torch_dtype)
        self.processor: AutoProcessor = AutoProcessor.from_pretrained(model_id)

    def run(self, audio_file_path: str) -> str:
        if self.model is None:
            raise ValueError("Model not loaded yet; call load_model() before run().")

        # Load audio
        audio_data, _ = load_audio_file(
            audio_file_path, self.processor.feature_extractor.sampling_rate
        )

        # Preprocess and move tensors to device
        inputs = self.processor(
            audio_data,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors="pt",
        )
        # input_features = inputs.input_features.to(self.device)
        input_features = inputs.input_features.to(self.device, dtype=self.torch_dtype)


        # Inference
        predicted_ids = self.model.generate(
            input_features, max_length=self.model.config.max_length
        )

        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        return transcription

    def run_and_cleanup(self, audio_file_path: str) -> Optional[str]:
        if self.has_gpu:
            print("GPU detected. Moving model to GPU for inference.")
            self.to_gpu(dtype=self.torch_dtype)

        transcription = None
        try:
            transcription = self.run(audio_file_path)
        except Exception as e:
            print(f"An error occurred during transcription: {e}")
        finally:
            if self.device.type == "cuda":
                self.to_cpu(dtype=self.torch_dtype)
                print("Model moved back to CPU and GPU memory cleared.")

        return transcription
