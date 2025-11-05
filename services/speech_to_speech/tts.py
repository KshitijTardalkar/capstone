"""
Text-to-Speech (TTS) service module.

This file defines the `TTS` class, which can load and run
different types of TTS models. It currently supports:
- VitsModel (e.g., 'facebook/mms-tts-eng')
- SpeechT5ForTextToSpeech (e.g., 'microsoft/speecht5_tts')

It handles model-specific tokenizers, processors, and generation logic.
"""

import torch
import soundfile as sf
import numpy as np
from transformers import (
    VitsModel, VitsTokenizer, 
    SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan
)
from datasets import load_dataset
from typing import Optional
import io

from services.speech_to_speech.helper import ModelHelper

class TTS(ModelHelper):
    """
    Text-to-Speech service supporting multiple model architectures.
    
    Attributes:
        processor: The model's processor (for SpeechT5).
        tokenizer: The model's tokenizer (for VITS).
        vocoder: The vocoder (for SpeechT5).
        speaker_embeddings: Speaker data for SpeechT5.
        model_type (str): 'vits' or 'speecht5'.
    """

    def __init__(
        self,
        model_id: str,
        torch_dtype: torch.dtype = torch.float32,
        use_gpu: bool = True,
        use_flash_attn: bool = False 
    ) -> None:
        
        self.model_type = 'vits' if 'vits' in model_id.lower() or 'mms' in model_id.lower() else 'speecht5'
        
        model_class = VitsModel
        if self.model_type == 'speecht5':
            model_class = SpeechT5ForTextToSpeech
        
        super().__init__(model_id, model_class, torch_dtype, use_gpu, use_flash_attn=False)
        
        self.processor = None
        self.tokenizer = None
        self.vocoder = None
        self.speaker_embeddings = None

        try:
            if self.model_type == 'vits':
                self.tokenizer = VitsTokenizer.from_pretrained(model_id)
            else:
                self.processor = SpeechT5Processor.from_pretrained(model_id)
                self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
                embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
                self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error loading TTS dependencies: {e}")
            raise

    def load_model(self) -> None:
        """Loads the specified TTS model from Hugging Face."""
        
        model_kwargs = {
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
            "device_map": "auto" if self._is_gpu_available and self.use_gpu else "cpu",
            "torch_dtype": self.torch_dtype if self._is_gpu_available and self.use_gpu else torch.float32
        }

        if not (self._is_gpu_available and self.use_gpu):
            model_kwargs["torch_dtype"] = torch.float32
            self.torch_dtype = torch.float32

        try:
            self.model = self.model_class.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            self.model.eval()
            
            if hasattr(self.model, 'device'):
                 self.device = self.model.device
            
            print(f"TTS Model {self.model_id} ({self.model_type}) loaded on {self.device}")

            try:
                if torch.__version__ >= "2.0.0" and self.device.type == 'cuda':
                    self.model = torch.compile(self.model)
                    print(f"TTS Model {self.model_id} compiled with torch.compile()")
            except Exception as compile_exc:
                print(f"Warning: torch.compile() failed for TTS: {compile_exc}")

        except Exception as exc:
            print(f"Error loading TTS model {self.model_id}: {exc}")
            raise

    def run(self, text: str) -> bytes:
        """
        Synthesizes speech from the given text using the loaded model.
        """
        if self.model is None:
            raise ValueError("Model not loaded yet; call load_model() before run().")

        if not text.strip():
            return b""

        with torch.no_grad():
            if self.model_type == 'vits':
                inputs = self.tokenizer(text, return_tensors="pt")
                input_ids = inputs["input_ids"].to(self.device)
                waveform = self.model(input_ids).waveform
                sample_rate = self.model.config.sampling_rate
            else:
                inputs = self.processor(text=text, return_tensors="pt").to(self.device)
                speech = self.model.generate_speech(
                    inputs["input_ids"], 
                    self.speaker_embeddings, 
                    vocoder=self.vocoder
                )
                waveform = speech
                sample_rate = 16000 # SpeechT5 sample rate is 16k

        audio_np = waveform.cpu().numpy().squeeze()
        audio_np = audio_np.astype(np.float32)
        
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)

        return buffer.read()