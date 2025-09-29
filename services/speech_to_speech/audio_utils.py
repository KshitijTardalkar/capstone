import librosa
import numpy as np
import os
import sounddevice as sd
import soundfile as sf
from typing import Tuple, Optional
import tempfile


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


def get_microphone_input(
    duration: int = 5,
    sample_rate: int = 16000,
    save_path: Optional[str] = None
) -> str:
    """
    Captures live audio from the microphone and saves it to a file.

    Args:
        duration (int): Duration of recording in seconds. Defaults to 5.
        sample_rate (int): Sample rate for recording. Defaults to 16000 Hz.
        save_path (Optional[str]): Path to save the recorded audio. If None,
                                   saves to a temporary file.

    Returns:
        str: Path to the saved audio file.

    Raises:
        Exception: If there's an error during recording or saving.
    """
    print(f"\n🎤 Recording audio for {duration} seconds...")
    print("Please speak into your microphone...")
    
    try:
        # Record audio from microphone
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,  # Mono recording
            dtype='float32'
        )
        
        # Wait for recording to complete
        sd.wait()
        print("✓ Recording completed!")
        
        # Determine save path
        if save_path is None:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.wav',
                delete=False
            )
            save_path = temp_file.name
            temp_file.close()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # Save the recorded audio
        sf.write(save_path, audio_data, sample_rate)
        print(f"✓ Audio saved to: {save_path}")
        
        return save_path
        
    except Exception as e:
        raise Exception(f"Failed to record audio from microphone: {e}")


def record_until_silence(
    sample_rate: int = 16000,
    silence_threshold: float = 0.01,
    silence_duration: float = 2.0,
    max_duration: int = 30,
    save_path: Optional[str] = None
) -> str:
    """
    Records audio from microphone and stops when silence is detected.

    Args:
        sample_rate (int): Sample rate for recording. Defaults to 16000 Hz.
        silence_threshold (float): Amplitude threshold below which audio is considered silence.
        silence_duration (float): Duration of silence (in seconds) required to stop recording.
        max_duration (int): Maximum recording duration in seconds. Defaults to 30.
        save_path (Optional[str]): Path to save the recorded audio. If None,
                                   saves to a temporary file.

    Returns:
        str: Path to the saved audio file.

    Raises:
        Exception: If there's an error during recording or saving.
    """
    print("\n🎤 Recording audio (will stop automatically after silence)...")
    print("Please speak into your microphone...")
    
    try:
        recorded_chunks = []
        silence_counter = 0
        chunk_duration = 0.1  # 100ms chunks
        chunk_samples = int(sample_rate * chunk_duration)
        max_chunks = int(max_duration / chunk_duration)
        silence_chunks_needed = int(silence_duration / chunk_duration)
        
        for i in range(max_chunks):
            # Record a small chunk
            chunk = sd.rec(
                chunk_samples,
                samplerate=sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            
            recorded_chunks.append(chunk)
            
            # Check if chunk is silent
            if np.max(np.abs(chunk)) < silence_threshold:
                silence_counter += 1
            else:
                silence_counter = 0
                print("🔊 Speaking detected...", end='\r')
            
            # Stop if we've had enough silence
            if silence_counter >= silence_chunks_needed:
                print("\n✓ Silence detected. Stopping recording...")
                break
        
        print("✓ Recording completed!")
        
        # Concatenate all chunks
        audio_data = np.concatenate(recorded_chunks, axis=0)
        
        # Determine save path
        if save_path is None:
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.wav',
                delete=False
            )
            save_path = temp_file.name
            temp_file.close()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # Save the recorded audio
        sf.write(save_path, audio_data, sample_rate)
        print(f"✓ Audio saved to: {save_path}")
        
        return save_path
        
    except Exception as e:
        raise Exception(f"Failed to record audio from microphone: {e}")