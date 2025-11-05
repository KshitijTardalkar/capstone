import torch
import soundfile as sf
import numpy as np
from transformers import VitsModel, VitsTokenizer
from typing import Optional
import io

from services.speech_to_speech.helper import ModelHelper

class TTS(ModelHelper):

    def __init__(
        self,
        model_id: str,
        torch_dtype: torch.dtype = torch.float32,
        use_gpu: bool = True,
        use_flash_attn: bool = False 
    ) -> None:
        super().__init__(model_id, VitsModel, torch_dtype, use_gpu, use_flash_attn=False)
        self.tokenizer = VitsTokenizer.from_pretrained(model_id)

    def run(self, text: str) -> bytes:
        if self.model is None:
            raise ValueError("Model not loaded yet; call load_model() before run().")

        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            waveform = self.model(input_ids).waveform

        audio_np = waveform.cpu().numpy().squeeze()
        audio_np = audio_np.astype(np.float32)

        sample_rate = self.model.config.sampling_rate
        
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)

        print(f"Audio synthesized in-memory")
        return buffer.read()