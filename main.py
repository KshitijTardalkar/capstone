from services.speech_to_speech.pipeline import SpeechToSpeechPipeline

# Initialize the pipeline
pipeline = SpeechToSpeechPipeline(
    stt_model_id="openai/whisper-tiny",
    llm_model_id="meta-llama/Llama-3.2-1B-Instruct", 
    tts_model_id="facebook/mms-tts-eng"
)

# ========================================
# USAGE EXAMPLES
# ========================================

# Example 1: Process audio from file (original functionality)
print("\n" + "="*60)
print("Example 1: Processing audio from file")
print("="*60)
response_audio = pipeline.run(
    audio_file_path="/home/kshitij/Music/MLKDream.wav"
)
if response_audio:
    print(f"Response saved to: {response_audio}")


# Example 2: Capture audio from microphone (fixed duration - 5 seconds)
print("\n" + "="*60)
print("Example 2: Recording from microphone (5 seconds)")
print("="*60)
response_audio = pipeline.run(
    use_microphone=True,
    recording_duration=5
)
if response_audio:
    print(f"Response saved to: {response_audio}")


# Example 3: Capture audio from microphone with auto-stop on silence
print("\n" + "="*60)
print("Example 3: Recording from microphone (auto-stop on silence)")
print("="*60)
response_audio = pipeline.run(
    use_microphone=True,
    auto_stop_on_silence=True
)
if response_audio:
    print(f"Response saved to: {response_audio}")


# Example 4: Interactive loop - keep asking for microphone input
print("\n" + "="*60)
print("Example 4: Interactive conversation mode")
print("="*60)

def interactive_mode():
    """Run an interactive conversation loop with microphone input."""
    print("\nStarting interactive mode...")
    print("Press Ctrl+C to exit at any time.\n")
    
    conversation_count = 0
    
    try:
        while True:
            conversation_count += 1
            print(f"\n--- Conversation {conversation_count} ---")
            
            # Get user input via microphone
            response_audio = pipeline.run(
                use_microphone=True,
                auto_stop_on_silence=True,  # Automatically stop when user stops speaking
                output_dir="output"
            )
            
            if response_audio:
                print(f"\n✓ Conversation {conversation_count} completed!")
                print(f"  Response audio: {response_audio}")
            else:
                print(f"\n❌ Conversation {conversation_count} failed!")
            
            # Ask if user wants to continue
            continue_choice = input("\nContinue? (y/n): ").strip().lower()
            if continue_choice != 'y':
                break
                
    except KeyboardInterrupt:
        print("\n\nExiting interactive mode...")
    
    print(f"\nTotal conversations: {conversation_count}")
    print("Goodbye!")


# Uncomment the line below to run interactive mode
# interactive_mode()