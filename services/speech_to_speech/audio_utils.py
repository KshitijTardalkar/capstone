"""
Audio Processing Utilities.

This file provides a helper function `load_audio_file` to robustly
load audio from either a file path or an in-memory BytesIO object.
It handles resampling to a target sample rate, conversion to mono,
and trimming leading/trailing silence for optimization.
"""

import librosa
import numpy as np
import os
from typing import Tuple, Union
import io
import soundfile as sf

def load_audio_file(file_path: Union[str, io.BytesIO], target_sr: int) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file from path or bytes, resamples, and trims silence.

    Args:
        file_path (Union[str, io.BytesIO]): The path to the audio file or
            an in-memory BytesIO object.
        target_sr (int): The target sample rate to resample the audio to.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing the audio data as a
            NumPy array and the (target) sample rate.
    
    Raises:
        FileNotFoundError: If `file_path` is a string and the file does not exist.
        TypeError: If `file_path` is not a string or BytesIO object.
        Exception: If loading or processing the audio fails.
    """
    
    if isinstance(file_path, str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found at: {file_path}")
        try:
            audio_data, sr = librosa.load(file_path, sr=target_sr, mono=True)
        except Exception as e:
            raise Exception(f"Failed to load audio file from path: {e}")
    
    elif isinstance(file_path, io.BytesIO):
        try:
            file_path.seek(0)
            audio_data, sr = sf.read(file_path, dtype='float32')
            
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
                
            if sr != target_sr:
                audio_data = librosa.resample(y=audio_data, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
                
        except Exception as e:
            try:
                file_path.seek(0)
                audio_data, sr = librosa.load(file_path, sr=target_sr, mono=True)
            except Exception as librosa_e:
                raise Exception(f"Failed to load audio from BytesIO with soundfile ({e}) and librosa ({librosa_e})")
    
    else:
        raise TypeError(f"audio_input must be a file path (str) or io.BytesIO, not {type(file_path)}")

    try:
        audio_data, _ = librosa.effects.trim(audio_data, top_db=20)
    except Exception as e:
        print(f"Warning: librosa.effects.trim failed: {e}")

    return audio_data, sr