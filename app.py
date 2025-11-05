from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
import base64
import psutil
import GPUtil
from services.speech_to_speech.pipeline import SpeechToSpeechPipeline
import threading
import wave

app = Flask(__name__)
CORS(app)

# Global pipeline instance, created once
pipeline = SpeechToSpeechPipeline()
pipeline_lock = threading.Lock()

# Available models configuration
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
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    ],
    'tts': [
        'facebook/mms-tts-eng',
        'microsoft/speecht5_tts',
        'suno/bark-small'
    ]
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    """Return available models"""
    return jsonify(AVAILABLE_MODELS)

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize pipeline with selected models"""
    data = request.json
    stt_model = data.get('stt_model')
    llm_model = data.get('llm_model')
    tts_model = data.get('tts_model')
    
    stt_use_gpu = data.get('stt_use_gpu', False)
    llm_use_gpu = data.get('llm_use_gpu', False)
    tts_use_gpu = data.get('tts_use_gpu', False)

    with pipeline_lock:
        try:
            # Call the load_models method with GPU preferences
            pipeline.load_models(
                stt_model_id=stt_model,
                llm_model_id=llm_model,
                tts_model_id=tts_model,
                stt_use_gpu=stt_use_gpu,
                llm_use_gpu=llm_use_gpu,
                tts_use_gpu=tts_use_gpu
            )
            return jsonify({'success': True, 'message': 'Pipeline models loaded successfully'})
        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            return jsonify({'success': False, 'message': f'Failed to initialize pipeline: {e}'}), 500

@app.route('/api/process_audio', methods=['POST'])
def process_audio():
    """Process audio through the pipeline"""
    global pipeline
    
    # Check if models are loaded inside the pipeline object
    if not pipeline or not pipeline.models_loaded():
        return jsonify({'error': 'Pipeline not initialized'}), 400
    
    try:
        # Get audio data from request
        audio_data = request.json.get('audio')
        if not audio_data:
            return jsonify({'error': 'No audio data provided'}), 400
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_data.split(',')[1] if ',' in audio_data else audio_data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        
        # Create output directory
        output_dir = tempfile.mkdtemp()
        
        # Process through pipeline
        # Lock to prevent processing while models might be swapping
        with pipeline_lock:
            # Get STT result
            stt_result = pipeline.stt_service.run(temp_audio_path)
            
            # Get LLM result
            llm_result = pipeline.llm_service.run(
                f"User Input: {stt_result}\n\nRespond in a single sentence as it will be spoken by TTS.",
                max_new_tokens=100
            )
            
            # Generate TTS audio
            output_audio_path = os.path.join(output_dir, 'response.wav')
            tts_result = pipeline.tts_service.run(llm_result, output_audio_path)
        
        # Read audio file and convert to base64
        with open(tts_result, 'rb') as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
        
        # Cleanup
        os.unlink(temp_audio_path)
        os.unlink(tts_result)
        os.rmdir(output_dir)
        
        return jsonify({
            'stt_text': stt_result,
            'llm_response': llm_result,
            'audio': f'data:audio/wav;base64,{audio_base64}'
        })
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system_info', methods=['GET'])
def get_system_info():
    """Get system information"""
    try:
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_freq = psutil.cpu_freq()
        
        # Memory info
        memory = psutil.virtual_memory()
        
        # Disk info
        disk = psutil.disk_usage('/')
        
        # GPU info
        gpus = []
        try:
            gpu_list = GPUtil.getGPUs()
            for gpu in gpu_list:
                gpus.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': f'{gpu.load * 100:.1f}%',
                    'memory_used': f'{gpu.memoryUsed}MB',
                    'memory_total': f'{gpu.memoryTotal}MB',
                    'temperature': f'{gpu.temperature}°C'
                })
        except:
            gpus = [] # Send empty list if no GPU
        
        # Process info (top processes by CPU)
        processes = []
        for proc in sorted(psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']), 
                          key=lambda p: p.info['cpu_percent'] or 0, reverse=True)[:10]:
            try:
                processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cpu': f"{proc.info['cpu_percent']:.1f}%",
                    'memory': f"{proc.info['memory_percent']:.1f}%"
                })
            except:
                continue
        
        # Get model info from pipeline
        pipeline_info = pipeline.get_system_info()

        return jsonify({
            'cpu': {
                'percent': cpu_percent,
                'avg_percent': sum(cpu_percent) / len(cpu_percent),
                'frequency': f'{cpu_freq.current:.0f} MHz' if cpu_freq else 'N/A',
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True)
            },
            'memory': {
                'total': f'{memory.total / (1024**3):.2f} GB',
                'used': f'{memory.used / (1024**3):.2f} GB',
                'percent': memory.percent
            },
            'disk': {
                'total': f'{disk.total / (1024**3):.2f} GB',
                'used': f'{disk.used / (1024**3):.2f} GB',
                'percent': disk.percent
            },
            'gpu': gpus,
            'processes': processes,
            'pipeline': pipeline_info # Add pipeline info
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)