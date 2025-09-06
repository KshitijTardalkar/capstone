import librosa
import numpy as np
import os
from typing import Tuple


def load_audio_file(file_path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file from a given path and resamples it to the target sample rate.

    This function uses the Librosa library to handle various audio file formats
    and resample them to a uniform sample rate required by the model.

    Args:
        file_path (str): The path to the audio file.
        target_sr (int): The target sample rate for the audio data.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing the audio data as a NumPy array
                                and the final sample rate.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        Exception: For any other errors during audio loading.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found at: {file_path}")

    # librosa.load reads the file, decodes it, and resamples it to the target_sr.
    try:
        audio_data, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio_data, sr
    except Exception as e:
        raise Exception(f"Failed to load audio file: {e}")


def get_microphone_input():
    """
    (Placeholder) Captures live audio from a microphone.

    This function is a placeholder for future implementation of real-time
    microphone input.
    """
    print("Microphone input functionality is not yet implemented.")
    # Future implementation would use libraries like PyAudio to capture live audio.
    pass
