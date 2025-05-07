import os
import json
import numpy as np
import pickle
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import custom modules
from inference import ASLInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Global ASL inference object
asl_inference = None

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'ok',
        'model_loaded': asl_inference is not None
    })

@app.route('/predict-sequence', methods=['POST'])
def predict_sequence():
    """
    Endpoint for predicting ASL sign from a .pkl sequence
    
    Expected request format:
    - Multipart form with 'file' field containing the .pkl file
    OR
    - JSON with 'features' field containing the feature sequence
    
    Returns:
        JSON object with predicted class and confidence
    """
    if not asl_inference:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    try:
        if 'file' in request.files:
            # Handle .pkl file upload
            file = request.files['file']
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
                file.save(temp_file.name)
                temp_filepath = temp_file.name
            
            # Make prediction
            try:
                predicted_class, class_name, confidence = asl_inference.predict_from_pkl(temp_filepath)
                
                # Delete temporary file
                os.unlink(temp_filepath)
                
                return jsonify({
                    'class': int(predicted_class),
                    'class_name': class_name,
                    'confidence': float(confidence)
                })
            except Exception as e:
                # Delete temporary file
                os.unlink(temp_filepath)
                raise e
        
        elif request.json and 'features' in request.json:
            # Handle feature sequence in JSON
            features = np.array(request.json['features'])
            
            # Make prediction
            predicted_class, class_name, confidence = asl_inference.predict_from_feature_sequence(features)
            
            return jsonify({
                'class': int(predicted_class),
                'class_name': class_name,
                'confidence': float(confidence)
            })
        
        else:
            return jsonify({
                'error': 'No file or feature sequence provided'
            }), 400
    
    except Exception as e:
        logger.error(f"Error in predict-sequence: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/speech-to-gloss', methods=['POST'])
def speech_to_gloss():
    """
    Endpoint for converting English speech to ASL gloss
    
    Expected request format:
    - JSON with 'text' field containing the English text
    
    Returns:
        JSON object with ASL gloss
    """
    if not OPENAI_API_KEY:
        return jsonify({
            'error': 'OPENAI_API_KEY not set in environment variables'
        }), 500
    
    try:
        # Get text from request
        if not request.json or 'text' not in request.json:
            return jsonify({
                'error': 'No text provided'
            }), 400
        
        english_text = request.json['text']
        
        # Call OpenAI API to convert English to ASL gloss
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'gpt-4-turbo',
            'messages': [
                {
                    'role': 'system', 
                    'content': (
                        'You are an expert in American Sign Language (ASL) gloss. '
                        'Convert the following English text to ASL gloss notation. '
                        'ASL gloss uses uppercase English words in the order they would be signed in ASL, '
                        'without English grammar elements that don\'t exist in ASL. '
                        'For example "I am going to the store" would be "I GO STORE".'
                    )
                },
                {
                    'role': 'user',
                    'content': english_text
                }
            ],
            'max_tokens': 150
        }
        
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
        response_data = response.json()
        
        if 'choices' in response_data and len(response_data['choices']) > 0:
            asl_gloss = response_data['choices'][0]['message']['content'].strip()
            
            return jsonify({
                'english_text': english_text,
                'asl_gloss': asl_gloss
            })
        else:
            logger.error(f"Unexpected response from OpenAI API: {response_data}")
            return jsonify({
                'error': 'Failed to convert text to ASL gloss'
            }), 500
    
    except Exception as e:
        logger.error(f"Error in speech-to-gloss: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

def init_app(model_path, label_map_path=None, seq_length=50):
    """
    Initialize the Flask app with the ASL inference model
    
    Args:
        model_path: Path to the saved model (.h5 file)
        label_map_path: Path to the label map file (JSON)
        seq_length: Length of the input sequence
        
    Returns:
        Flask app
    """
    global asl_inference
    
    # Initialize ASL inference
    try:
        asl_inference = ASLInference(
            model_path=model_path,
            seq_length=seq_length
        )
        
        # Load label map if provided
        if label_map_path:
            asl_inference.load_label_map(label_map_path)
        
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
    
    return app

if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ASL recognition API')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the saved model (.h5 file)')
    parser.add_argument('--label-map', type=str, default=None,
                        help='Path to the label map file (JSON)')
    parser.add_argument('--seq-length', type=int, default=50,
                        help='Length of the input sequence')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the API server on')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the API server on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Initialize app
    app = init_app(
        model_path=args.model_path,
        label_map_path=args.label_map,
        seq_length=args.seq_length
    )
    
    # Run app
    app.run(host=args.host, port=args.port, debug=args.debug) 