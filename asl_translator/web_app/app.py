from flask import Flask, render_template, request, jsonify, Response
import os
import sys
import json
import time
import numpy as np
from threading import Thread
import logging

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import translator
from inference import RealTimeASLTranslator
from utils.speech_to_text import SpeechToText
from utils.english_to_asl import EnglishToASLConverter

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.logger.setLevel(logging.INFO)

# Ensure uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Global variables
translator = None
continuous_thread = None
is_running = False
latest_results = []

def initialize_translator(model_path, label_mappings_path):
    """Initialize the ASL translator."""
    global translator
    
    if translator is None:
        app.logger.info(f"Initializing translator with model: {model_path}")
        try:
            translator = RealTimeASLTranslator(
                model_path=model_path,
                label_mappings_path=label_mappings_path
            )
            app.logger.info("Translator initialized successfully.")
        except Exception as e:
            app.logger.error(f"Error initializing translator: {e}")
            return False
    
    return True

def process_continuous_speech():
    """Process continuous speech in a separate thread."""
    global translator, is_running, latest_results
    
    def callback(result):
        """Callback function for continuous speech recognition."""
        global latest_results
        latest_results.append(result)
        if len(latest_results) > 10:
            latest_results = latest_results[-10:]
    
    is_running = True
    translator.run_continuous(callback=callback)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    """
    Handle translation requests.
    Accepts text input or audio files.
    """
    global translator
    
    if translator is None:
        return jsonify({'error': 'Translator not initialized'}), 500
    
    if 'text' in request.form:
        # Process text input
        text = request.form['text']
        app.logger.info(f"Translating text: {text}")
        
        try:
            # Convert text to ASL gloss
            asl_gloss = translator.english_to_asl.convert(text)
            
            return jsonify({
                'english': text,
                'asl_gloss': asl_gloss,
                'success': True
            })
        except Exception as e:
            app.logger.error(f"Error translating text: {e}")
            return jsonify({'error': str(e), 'success': False}), 500
    
    elif 'audio' in request.files:
        # Process audio file
        audio_file = request.files['audio']
        
        # Save audio file
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"audio_{int(time.time())}.wav")
        audio_file.save(audio_path)
        
        app.logger.info(f"Processing audio file: {audio_path}")
        
        try:
            # Process speech
            english_text, asl_gloss = translator.process_speech(audio_path)
            
            return jsonify({
                'english': english_text,
                'asl_gloss': asl_gloss,
                'success': True
            })
        except Exception as e:
            app.logger.error(f"Error processing audio: {e}")
            return jsonify({'error': str(e), 'success': False}), 500
        finally:
            # Clean up audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
    
    else:
        return jsonify({'error': 'No text or audio provided', 'success': False}), 400

@app.route('/listen/start', methods=['POST'])
def start_listening():
    """Start continuous speech recognition."""
    global translator, continuous_thread, is_running
    
    if translator is None:
        return jsonify({'error': 'Translator not initialized', 'success': False}), 500
    
    if is_running:
        return jsonify({'message': 'Already listening', 'success': True})
    
    # Start continuous speech recognition in a separate thread
    continuous_thread = Thread(target=process_continuous_speech)
    continuous_thread.daemon = True
    continuous_thread.start()
    
    return jsonify({'message': 'Started listening', 'success': True})

@app.route('/listen/stop', methods=['POST'])
def stop_listening():
    """Stop continuous speech recognition."""
    global is_running
    
    is_running = False
    
    return jsonify({'message': 'Stopped listening', 'success': True})

@app.route('/listen/results')
def get_results():
    """Get latest translation results."""
    global latest_results
    
    return jsonify({
        'results': latest_results,
        'success': True
    })

@app.route('/load_model', methods=['POST'])
def load_model():
    """Load a model."""
    model_path = request.form.get('model_path')
    label_mappings_path = request.form.get('label_mappings_path')
    
    if not model_path or not label_mappings_path:
        return jsonify({'error': 'Model path and label mappings path required', 'success': False}), 400
    
    if not os.path.exists(model_path):
        return jsonify({'error': f'Model file not found: {model_path}', 'success': False}), 404
    
    if not os.path.exists(label_mappings_path):
        return jsonify({'error': f'Label mappings file not found: {label_mappings_path}', 'success': False}), 404
    
    success = initialize_translator(model_path, label_mappings_path)
    
    if success:
        return jsonify({'message': 'Model loaded successfully', 'success': True})
    else:
        return jsonify({'error': 'Failed to load model', 'success': False}), 500

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="ASL Translation Web App")
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--label_mappings_path", type=str, help="Path to the label mappings file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    if args.model_path and args.label_mappings_path:
        initialize_translator(args.model_path, args.label_mappings_path)
    
    app.run(host=args.host, port=args.port, debug=args.debug) 