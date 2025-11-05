import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from typing import Optional

from services.speech_to_speech.helper import ModelHelper
from services.speech_to_speech.audio_utils import load_audio_file


class stt(ModelHelper):
    """
    Speech-to-Text (STT) wrapper for Hugging Face Whisper models.
    """

    def __init__(
        self, 
        model_id: str, 
        torch_dtype: torch.dtype = torch.float32, 
        use_gpu: bool = True
    ) -> None:
        """
        Initialize the STT wrapper.
        """
        super().__init__(model_id, AutoModelForSpeechSeq2Seq, torch_dtype, use_gpu)
        self.processor: AutoProcessor = AutoProcessor.from_pretrained(model_id)

    def run(self, audio_file_path: str) -> str:
        """
        Perform transcription on a single audio file.
        """
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
        # Move *only the input tensor* to the model's device
        input_features = inputs.input_features.to(self.device, dtype=self.torch_dtype)

        # Inference (model is already on self.device)
        predicted_ids = self.model.generate(
            input_features, max_length=self.model.config.max_length
        )

        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        return transcription