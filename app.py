from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64
import psutil
import GPUtil
from services.speech_to_speech.pipeline import SpeechToSpeechPipeline
import threading
import io
import time

from config import AVAILABLE_MODELS, LLM_USER_PROMPT_TEMPLATE

app = Flask(__name__)
CORS(app)

pipeline = SpeechToSpeechPipeline()
pipeline_lock = threading.Lock()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify(AVAILABLE_MODELS)

@app.route('/api/initialize', methods=['POST'])
def initialize():
    data = request.json
    stt_model = data.get('stt_model')
    llm_model = data.get('llm_model')
    tts_model = data.get('tts_model')
    
    stt_use_gpu = data.get('stt_use_gpu', False)
    llm_use_gpu = data.get('llm_use_gpu', False)
    tts_use_gpu = data.get('tts_use_gpu', False)
    llm_quantize = data.get('llm_quantize', 'none')
    stt_use_flash_attn = data.get('stt_use_flash_attn', False)
    llm_use_flash_attn = data.get('llm_use_flash_attn', False)

    with pipeline_lock:
        try:
            pipeline.load_models(
                stt_model_id=stt_model,
                llm_model_id=llm_model,
                tts_model_id=tts_model,
                stt_use_gpu=stt_use_gpu,
                llm_use_gpu=llm_use_gpu,
                tts_use_gpu=tts_use_gpu,
                llm_quantize=llm_quantize,
                stt_use_flash_attn=stt_use_flash_attn,
                llm_use_flash_attn=llm_use_flash_attn
            )
            return jsonify({'success': True, 'message': 'Pipeline models loaded successfully'})
        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            return jsonify({'success': False, 'message': f'Failed to initialize pipeline: {str(e)}'}), 500

@app.route('/api/process_audio', methods=['POST'])
def process_audio():
    global pipeline
    
    if not pipeline or not pipeline.models_loaded():
        return jsonify({'error': 'Pipeline not initialized'}), 400
    
    try:
        audio_data = request.json.get('audio')
        if not audio_data:
            return jsonify({'error': 'No audio data provided'}), 400
        
        audio_bytes = base64.b64decode(audio_data.split(',')[1] if ',' in audio_data else audio_data)
        
        audio_buffer = io.BytesIO(audio_bytes)
        
        with pipeline_lock:
            stt_result = pipeline.stt_service.run(audio_buffer)
            
            prompt = LLM_USER_PROMPT_TEMPLATE.format(stt_result=stt_result)
            
            llm_result = pipeline.llm_service.run(
                prompt,
                max_new_tokens=100
            )
            
            tts_audio_bytes = pipeline.tts_service.run(llm_result)
        
        audio_base64 = base64.b64encode(tts_audio_bytes).decode('utf-8')
        
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
    try:
        cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
        cpu_freq = psutil.cpu_freq()
        
        memory = psutil.virtual_memory()
        
        disk = psutil.disk_usage('/')
        
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
            gpus = []
        
        processes = []
        for proc in sorted(psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']), 
                          key=lambda p: p.info['cpu_percent'] or 0, reverse=True)[:10]:
            try:
                info = proc.info
                processes.append({
                    'pid': info['pid'],
                    'name': info['name'],
                    'cpu': f"{info['cpu_percent']:.1f}%",
                    'memory': f"{info['memory_percent']:.1f}%"
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
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
            'pipeline': pipeline_info
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)