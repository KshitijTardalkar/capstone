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
        torch_dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__(model_id, VitsModel, torch_dtype)
        self.tokenizer = VitsTokenizer.from_pretrained(model_id)

    def run(self, text: str, output_path: str = "output.wav") -> str:
        if self.model is None:
            raise ValueError("Model not loaded yet; call load_model() before run().")

        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            waveform = self.model(input_ids).waveform

        audio_np = waveform.cpu().numpy().squeeze()
        audio_np = audio_np.astype(np.float32)

        sample_rate = self.model.config.sampling_rate
        sf.write(output_path, audio_np, sample_rate)

        print(f"Audio saved to: {output_path}")
        return output_path

    def run_and_cleanup(
        self,
        text: str,
        output_path: str = "output.wav"
    ) -> Optional[str]:
        if self.has_gpu:
            print("GPU detected. Moving TTS model to GPU for inference.")
            self.to_gpu(dtype=self.torch_dtype)

        output_file = None
        try:
            output_file = self.run(text, output_path)
        except Exception as e:
            print(f"An error occurred during TTS synthesis: {e}")
        finally:
            if self.device.type == "cuda":
                self.to_cpu(dtype=self.torch_dtype)
                print("TTS model moved back to CPU and GPU memory cleared.")

        return output_file
