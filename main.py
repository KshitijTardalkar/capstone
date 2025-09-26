from services.speech_to_speech.pipeline import SpeechToSpeechPipeline
   
pipeline = SpeechToSpeechPipeline(
    stt_model_id="openai/whisper-tiny",
    llm_model_id="meta-llama/Llama-3.2-1B-Instruct", 
    tts_model_id="facebook/mms-tts-eng"
    # tts_model_id="microsoft/speecht5_tts",
    # tts_vocoder_id="microsoft/speecht5_hifigan"
)
   
# Process audio file
response_audio = pipeline.run("/home/kshitij/Music/MLKDream.wav")