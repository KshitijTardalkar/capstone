"""
Main Flask web application for the Speech-to-Speech Terminal.

This application serves the frontend, manages the AI model pipeline,
and handles API requests for audio processing and command execution.
It acts as the central orchestrator for the entire system.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64
import psutil
import GPUtil
from services.speech_to_speech.pipeline import SpeechToSpeechPipeline
from services.terminal_executor import TerminalExecutor
import threading
import io
import time
import json
import os
import re

from config import AVAILABLE_MODELS

app = Flask(__name__)
CORS(app)

pipeline = SpeechToSpeechPipeline()
executor = TerminalExecutor()
pipeline_lock = threading.Lock()

def split_into_sentences(text):
    """Splits text into sentences using regex."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return [s.strip() for s in sentences if s.strip()]

@app.route('/')
def index():
    """Serves the main HTML interface."""
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    """Provides the list of available models to the frontend."""
    return jsonify(AVAILABLE_MODELS)

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """
    Initializes and loads the AI models based on frontend selection.
    This is a long-running task that locks the pipeline.
    """
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
            if pipeline.llm_service:
                pipeline.llm_service.clear_memory()
                
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
    """
    The core speech-to-action endpoint.
    Receives audio, transcribes it (STT), reasons on it (LLM),
    and returns either a command proposal or synthesized speech (TTS).
    """
    global pipeline
    
    if not pipeline or not pipeline.models_loaded():
        return jsonify({'error': 'Pipeline not initialized'}), 400
    
    try:
        audio_data = request.json.get('audio')
        if not audio_data:
            return jsonify({'error': 'No audio data provided'}), 400
        
        audio_bytes = base64.b64decode(audio_data.split(',')[1] if ',' in audio_data else audio_data)
        
        audio_buffer = io.BytesIO(audio_bytes)
        
        llm_result = ""
        tts_audio_bytes = b""
        
        with pipeline_lock:
            stt_result = pipeline.stt_service.run(audio_buffer)
            
            if not stt_result:
                 return jsonify({'error': 'STT failed to transcribe audio'}), 400

            llm_response_str = pipeline.llm_service.run(
                user_input=stt_result,
                cwd=executor.cwd,
                max_new_tokens=256
            )
            
            print(f"LLM Raw Output: {llm_response_str}")

            try:
                start = llm_response_str.find('{')
                end = llm_response_str.rfind('}') + 1
                
                if start == -1 or end == -1:
                    raise json.JSONDecodeError("No JSON object found", llm_response_str, 0)
                
                json_str = llm_response_str[start:end]
                
                llm_json = json.loads(json_str)
                response_type = llm_json.get("type")

                if response_type == "command":
                    return jsonify({
                        "type": "command",
                        "command": llm_json.get("command", "echo 'error: no command provided'"),
                        "stt_text": stt_result
                    })

                elif response_type == "chat":
                    llm_result = llm_json.get("response", "I'm sorry, I couldn't process that.")
                
                else:
                    llm_result = f"Error: LLM returned unknown type: {response_type}"
                    print(f"Invalid JSON 'type': {response_type}")

            except json.JSONDecodeError as e:
                print(f"JSON DECODE ERROR ({e}) on: {llm_response_str}")
                match = re.search(r'["\']response["\']\s*:\s*["\'](.*?)["\']', llm_response_str, re.DOTALL)
                if match:
                    llm_result = match.group(1).strip()
                    print(f"Fallback: Extracted chat response: {llm_result}")
                else:
                    print("Fallback: Could not extract chat response. Returning raw text without audio.")
                    llm_result = llm_response_str 
                    
                    return jsonify({
                        'type': 'chat',
                        'stt_text': stt_result,
                        'llm_response': llm_result,
                        'audio': None 
                    })
            
            sentences = split_into_sentences(llm_result)
            all_audio_bytes = []
            
            for sentence in sentences:
                if sentence:
                    print(f"Synthesizing: {sentence}")
                    audio_chunk = pipeline.tts_service.run(sentence)
                    all_audio_bytes.append(audio_chunk)
            
            tts_audio_bytes = b"".join(all_audio_bytes)
        
        audio_base64 = base64.b64encode(tts_audio_bytes).decode('utf-8')
        
        return jsonify({
            'type': 'chat',
            'stt_text': stt_result,
            'llm_response': llm_result,
            'audio': f'data:audio/wav;base64,{audio_base64}'
        })
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/execute_command', methods=['POST'])
def execute_command():
    """
    Executes a terminal command provided by the frontend.
    Uses the stateful TerminalExecutor to manage CWD.
    """
    global executor
    data = request.json
    command = data.get('command')

    if not command:
        return jsonify({'error': 'No command provided', 'cwd': executor.cwd}), 400

    try:
        result = executor.execute(command)
        return jsonify(result)
    except Exception as e:
        print(f"Error executing command: {e}")
        return jsonify({'output': str(e), 'cwd': executor.cwd}), 500

@app.route('/api/system_info', methods=['GET'])
def get_system_info():
    """
    Provides detailed hardware (CPU, GPU, RAM) and pipeline status
    to the frontend monitoring panel.
    """
    global executor
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
        pipeline_info['current_working_directory'] = executor.cwd

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