import torch
import soundfile as sf
import numpy as np
from transformers import VitsModel, VitsTokenizer
from typing import Optional

from services.speech_to_speech.helper import ModelHelper

class TTS(ModelHelper):

    def __init__(
        self,
        model_id: str,
        torch_dtype: torch.dtype = torch.float32,
        use_gpu: bool = True
    ) -> None:
        """
        Initialize the TTS wrapper.
        """
        super().__init__(model_id, VitsModel, torch_dtype, use_gpu)
        self.tokenizer = VitsTokenizer.from_pretrained(model_id)

    def run(self, text: str, output_path: str = "output.wav") -> str:
        """
        Synthesize speech from text.
        """
        if self.model is None:
            raise ValueError("Model not loaded yet; call load_model() before run().")

        inputs = self.tokenizer(text, return_tensors="pt")
        # Move *only the input tensor* to the model's device
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            # Model is already on self.device
            waveform = self.model(input_ids).waveform

        audio_np = waveform.cpu().numpy().squeeze()
        audio_np = audio_np.astype(np.float32)

        sample_rate = self.model.config.sampling_rate
        sf.write(output_path, audio_np, sample_rate)

        print(f"Audio saved to: {output_path}")
        return output_path