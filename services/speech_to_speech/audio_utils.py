import librosa
import numpy as np
import os
from typing import Tuple, Union
import io
import soundfile as sf

def load_audio_file(file_path: Union[str, io.BytesIO], target_sr: int) -> Tuple[np.ndarray, int]:
    
    if isinstance(file_path, str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found at: {file_path}")
        try:
            audio_data, sr = librosa.load(file_path, sr=target_sr, mono=True)
            return audio_data, sr
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
                
            return audio_data, sr
        except Exception as e:
            try:
                file_path.seek(0)
                audio_data, sr = librosa.load(file_path, sr=target_sr, mono=True)
                return audio_data, sr
            except Exception as librosa_e:
                raise Exception(f"Failed to load audio from BytesIO with soundfile ({e}) and librosa ({librosa_e})")
    
    else:
        raise TypeError(f"audio_input must be a file path (str) or io.BytesIO, not {type(file_path)}")

