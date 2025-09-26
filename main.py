import os
from services.speech_to_speech.pipeline import SpeechToSpeechPipeline
from services.iot_simulator import IoTDataManager

input_audio="path/to/your/input_audio.wav"



def main():
    # need to do model configuration
    stt_model=""
    llm_model=""
    tts_model=""

    if not os.path.exists(INPUT_AUDIO):
        print(f"Error: Input audio file not found at '{INPUT_AUDIO}'")
        return
    
    try:
        
        iot_manager = IoTDataManager()

        
        pipeline = SpeechToSpeechPipeline(
            stt_model_id=STT_MODEL,
            llm_model_id=LLM_MODEL,
            tts_model_id=TTS_MODEL,
            data_manager=iot_manager  
        )
        
        output_path = pipeline.run(INPUT_AUDIO)
        
        print(f"\nPipeline execution complete. Final audio is at: {output_path}")

    except Exception as e:
        print(f"An error occurred in the pipeline: {e}")


    



if __name__ == "__main__":
    main()
