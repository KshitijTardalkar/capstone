"""
Speech-to-Text (STT) service module.

This file defines the `stt` class, which encapsulates the functionality
for loading a Whisper model and performing audio transcription.
It inherits from `ModelHelper` and implements custom model loading
to support 8-bit quantization and `torch.compile()`.
"""

import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, BitsAndBytesConfig
from typing import Union
import io

from services.speech_to_speech.helper import ModelHelper
from services.speech_to_speech.audio_utils import load_audio_file


class stt(ModelHelper):
    """
    Speech-to-Text service using a Whisper-based model.

    Handles loading the STT model (with optional 8-bit quantization) and
    processing audio input (from a file path or in-memory bytes) to
    generate a text transcription.

    Attributes:
        processor (AutoProcessor): The processor for the STT model.
        quantize (str): The quantization mode (e.g., 'none', '8bit').
    """

    def __init__(
        self, 
        model_id: str, 
        torch_dtype: torch.dtype = torch.float32, 
        use_gpu: bool = True,
        use_flash_attn: bool = False,
        quantize: str = 'none'
    ) -> None:
        """
        Initializes the STT service.

        Args:
            model_id (str): The Hugging Face model identifier (e.g., 'openai/whisper-small').
            torch_dtype (torch.dtype): The desired data type for the model.
            use_gpu (bool): Whether to use the GPU if available.
            use_flash_attn (bool): Whether to use Flash Attention 2.
            quantize (str): The quantization mode. '4bit' or '8bit' will load in 8-bit.
        """
        super().__init__(model_id, AutoModelForSpeechSeq2Seq, torch_dtype, use_gpu, use_flash_attn)
        self.processor: AutoProcessor = AutoProcessor.from_pretrained(model_id)
        self.quantize = quantize

    def load_model(self) -> None:
        """
        Loads the STT model from Hugging Face.

        Applies 8-bit quantization if specified and available.
        Applies `torch.compile()` for optimization on GPU.
        """
        if not (self._is_gpu_available and self.use_gpu):
            self.quantize = 'none'
            self.use_flash_attn = False
        
        quantization_config = None
        
        if self.quantize in ['4bit', '8bit']:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            print(f"Applying 8-bit quantization for STT model {self.model_id}")
        
        model_kwargs = {
            "quantization_config": quantization_config,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
            "device_map": "auto" if self._is_gpu_available and self.use_gpu else "cpu"
        }

        if self._is_gpu_available and self.use_gpu and self.use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            model_kwargs["torch_dtype"] = torch.float16
            self.torch_dtype = torch.float16
        elif self.quantize == 'none':
            model_kwargs["torch_dtype"] = self.torch_dtype
        
        try:
            self.model = self.model_class.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            
            self.model.eval()
            
            if hasattr(self.model, 'device'):
                 self.device = self.model.device
            
            print(f"STT Model {self.model_id} loaded on {self.device}")
            if self.quantize != 'none':
                print("STT Quantization: 8-bit")

            try:
                if torch.__version__ >= "2.0.0" and self.device.type == 'cuda':
                    self.model = torch.compile(self.model)
                    print(f"STT Model {self.model_id} compiled with torch.compile()")
            except Exception as compile_exc:
                print(f"Warning: torch.compile() failed for STT: {compile_exc}")

        except Exception as exc:
            print(f"Error loading STT model {self.model_id}: {exc}")
            if "flash_attention_2" in str(exc):
                print("!!! STT: Flash Attention 2 failed to load. !!!")
            raise

    def run(self, audio_input: Union[str, io.BytesIO]) -> str:
        """
        Transcribes the given audio input into text.

        Args:
            audio_input (Union[str, io.BytesIO]): Path to the audio file or an
                in-memory BytesIO object.

        Returns:
            str: The generated text transcription.
        
        Raises:
            ValueError: If the model is not loaded before running.
        """
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

        return transcription.strip()