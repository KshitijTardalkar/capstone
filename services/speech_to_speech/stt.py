import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from typing import Optional

from services.speech_to_speech.helper import ModelHelper
from services.speech_to_speech.audio_utils import load_audio_file


class stt(ModelHelper):
    """
    Speech-to-Text (STT) wrapper for Hugging Face Whisper models.

    Extends `ModelHelper` to provide audio transcription capabilities using
    pretrained Whisper models from Hugging Face. Handles audio preprocessing,
    inference, and optional device management (CPU/GPU).

    Responsibilities:
    - Initialize and manage Whisper STT models via Hugging Face Transformers.
    - Preprocess audio files into model-compatible tensor representations.
    - Perform speech-to-text inference and return decoded transcriptions.
    - Provide GPU-aware inference execution with automatic cleanup.

    Attributes:
        processor (AutoProcessor): Hugging Face processor for audio feature extraction
                                   and token decoding.
        model (torch.nn.Module, optional): Whisper model instance loaded via ModelHelper.
    """

    def __init__(self, model_id: str, torch_dtype: torch.dtype = torch.float32) -> None:
        """
        Initialize the STT wrapper with a given model ID and dtype.

        Args:
            model_id (str): Identifier or local path of the Whisper model to load.
            torch_dtype (torch.dtype, optional): Desired tensor dtype for model weights
                                                 and inference. Defaults to torch.float32.

        Notes:
            - The model itself is not loaded at construction time. Call `load_model()`
              from the parent `ModelHelper` class before running inference.
            - `processor` is always initialized here for audio preprocessing.
        """
        super().__init__(model_id, AutoModelForSpeechSeq2Seq, torch_dtype)
        self.processor: AutoProcessor = AutoProcessor.from_pretrained(model_id)

    def run(self, audio_file_path: str) -> str:
        """
        Perform transcription on a single audio file.

        Args:
            audio_file_path (str): Path to a `.wav` (or compatible) audio file.

        Returns:
            str: Transcribed text from the audio.

        Raises:
            ValueError: If the model has not been loaded (call `load_model()` first).
            Exception: Propagates any errors during preprocessing or inference.

        Workflow:
            1. Load and resample audio using `load_audio_file`.
            2. Convert audio into feature tensors via Hugging Face processor.
            3. Move tensors to the appropriate device/dtype.
            4. Generate predicted token IDs with the model.
            5. Decode tokens into human-readable text.
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
        """
        Perform transcription with optional GPU acceleration and safe cleanup.

        Args:
            audio_file_path (str): Path to the audio file for transcription.

        Returns:
            Optional[str]: Transcribed text if successful, None otherwise.

        Workflow:
            - If a GPU is available, move the model to GPU before inference.
            - Run standard `run()` transcription pipeline.
            - On completion (or error), return model to CPU and clear GPU memory.
        
        Notes:
            - Designed for one-off transcriptions where GPU memory should be freed
              immediately after inference.
            - Safe to call repeatedly, as device state is reset at the end.
        """
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