from services.speech_to_speech.stt import stt
from typing import Final
import torch

# --- Configuration ---
# The model ID from Hugging Face Hub.
MODEL_ID: Final[str] = "openai/whisper-tiny"
# The path to the audio file you want to transcribe.
AUDIO_FILE_PATH: Final[str] = "/home/kshitij/Music/MLKDream.wav"
# Set the data type to float16 to reduce VRAM usage
DTYPE: Final[torch.dtype] = torch.float16

def main():
    """
    Main function to run the speech-to-text pipeline.
    """
    # 1. Create an instance of the STT class
    transcriber = stt(model_id=MODEL_ID, torch_dtype=DTYPE)

    # 2. Load the model for use.
    print(f"Loading model: {MODEL_ID}")
    transcriber.load_model()
    print("Model loaded successfully.")

    # 3. Run the transcription pipeline with cleanup
    # transcription = transcriber.run_and_cleanup(AUDIO_FILE_PATH)

    # # 4. Print the result
    # if transcription:
    #     print("\n--- Transcription Result ---")
    #     print(transcription)

if __name__ == "__main__":
    main()