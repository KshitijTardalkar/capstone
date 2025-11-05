AVAILABLE_MODELS = {
    'stt': [
        'openai/whisper-tiny',
        'openai/whisper-base',
        'openai/whisper-small',
        'openai/whisper-medium'
    ],
    'llm': [
        'meta-llama/Llama-3.2-1B-Instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
        'microsoft/phi-2',
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'google/gemma-2-2b-it',
    ],
    'tts': [
        'facebook/mms-tts-eng',
        'microsoft/speecht5_tts',
        # 'suno/bark-small' # Bark often requires more complex generation args
    ]
}

LLM_SYSTEM_PROMPT = (
    "You are a helpful assistant. ""Provide clear and concise responses based on the user's input."
)

LLM_USER_PROMPT_TEMPLATE = (
    "User Input: {stt_result}\n\n""Generate an appropriate response."
)