from services.speech_to_speech.pipeline import SpeechToSpeechPipeline

# Initialize pipeline
print("Initializing pipeline...")
pipeline = SpeechToSpeechPipeline(
    stt_model_id="openai/whisper-tiny",
    llm_model_id="meta-llama/Llama-3.2-1B-Instruct", 
    tts_model_id="facebook/mms-tts-eng"
)

print("\nPipeline ready! Starting microphone test...")
print("You will be asked to speak for 5 seconds.\n")

# Test with microphone input
response_audio = pipeline.run(
    use_microphone=True,
    recording_duration=5,
    output_dir="output"
)

if response_audio:
    print(f"\nSuccess! Response saved to: {response_audio}")
else:
    print("\nFailed to process audio.")
