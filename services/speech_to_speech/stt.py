import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from typing import Optional, Union
import io

from services.speech_to_speech.helper import ModelHelper
from services.speech_to_speech.audio_utils import load_audio_file


class stt(ModelHelper):
    def __init__(
        self, 
        model_id: str, 
        torch_dtype: torch.dtype = torch.float32, 
        use_gpu: bool = True,
        use_flash_attn: bool = False
    ) -> None:
        super().__init__(model_id, AutoModelForSpeechSeq2Seq, torch_dtype, use_gpu, use_flash_attn)
        self.processor: AutoProcessor = AutoProcessor.from_pretrained(model_id)

    def run(self, audio_input: Union[str, io.BytesIO]) -> str:
        if self.model is None: 
            raise ValueError("Model not loaded yet; call load_model() before run().")

        audio_data, _ = load_audio_file(
            audio_input, self.processor.feature_extractor.sampling_rate
        )

        inputs = self.processor(
            audio_data,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device, dtype=self.torch_dtype)

        predicted_ids = self.model.generate(
            input_features, max_length=self.model.config.max_length
        )

        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        return transcription