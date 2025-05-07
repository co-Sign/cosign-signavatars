import os
import numpy as np
import tensorflow as tf
import argparse
import pickle
import json

from model import ASLRecognitionModel
from preprocessing import DataLoader

class ASLInference:
    def __init__(self, model_path, seq_length=50, label_map=None):
        """
        Initialize ASL inference with pretrained model
        
        Args:
            model_path: Path to the saved model (.h5 file)
            seq_length: Length of the input sequence (must match training)
            label_map: Dictionary mapping class indices to class names
        """
        self.model = tf.keras.models.load_model(model_path)
        self.seq_length = seq_length
        self.label_map = label_map or {}
        self.data_loader = DataLoader(data_dir=None, seq_length=seq_length)
        
        # Get input shape from model
        self.feature_dim = self.model.input_shape[-1]
        
    def load_label_map(self, label_map_path):
        """
        Load label map from file
        
        Args:
            label_map_path: Path to the label map file (JSON)
        """
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
        
    def predict_from_pkl(self, pkl_path):
        """
        Make prediction from .pkl file
        
        Args:
            pkl_path: Path to the .pkl file containing motion sequence
            
        Returns:
            Predicted class index, class name, and confidence
        """
        # Load and preprocess .pkl file
        data = self.data_loader.load_pkl_file(pkl_path)
        if data is None:
            raise ValueError(f"Could not load data from {pkl_path}")
        
        features = self.data_loader.extract_features_from_pkl(data)
        if features is None:
            raise ValueError(f"Could not extract features from {pkl_path}")
        
        # Pad or truncate sequence
        features = self.data_loader.pad_or_truncate_sequence(features)
        
        # Add batch dimension
        features = np.expand_dims(features, axis=0)
        
        # Make prediction
        predictions = self.model.predict(features)
        
        # Get predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Get class name from label map if available
        class_name = self.label_map.get(str(predicted_class), f"Class_{predicted_class}")
        
        return predicted_class, class_name, confidence
    
    def predict_from_feature_sequence(self, feature_sequence):
        """
        Make prediction from feature sequence
        
        Args:
            feature_sequence: Numpy array of shape (num_frames, feature_dim)
            
        Returns:
            Predicted class index, class name, and confidence
        """
        # Ensure feature_sequence has the expected shape
        if feature_sequence.ndim == 2 and feature_sequence.shape[1] == self.feature_dim:
            # Pad or truncate sequence
            feature_sequence = self.data_loader.pad_or_truncate_sequence(feature_sequence)
            
            # Add batch dimension
            feature_sequence = np.expand_dims(feature_sequence, axis=0)
            
            # Make prediction
            predictions = self.model.predict(feature_sequence)
            
            # Get predicted class and confidence
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Get class name from label map if available
            class_name = self.label_map.get(str(predicted_class), f"Class_{predicted_class}")
            
            return predicted_class, class_name, confidence
        else:
            raise ValueError(
                f"Expected feature sequence of shape (num_frames, {self.feature_dim}), "
                f"got {feature_sequence.shape}"
            )

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='ASL recognition inference')
    
    # Model parameters
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the saved model (.h5 file)')
    parser.add_argument('--label-map', type=str, default=None,
                        help='Path to the label map file (JSON)')
    parser.add_argument('--seq-length', type=int, default=50,
                        help='Length of the input sequence (must match training)')
    
    # Input parameters
    parser.add_argument('--input-pkl', type=str, default=None,
                        help='Path to the input .pkl file')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Directory containing multiple .pkl files for batch inference')
    
    # Output parameters
    parser.add_argument('--output-file', type=str, default=None,
                        help='Path to save the prediction results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if args.input_pkl is None and args.input_dir is None:
        raise ValueError("Either --input-pkl or --input-dir must be provided")
    
    return args

def main():
    """
    Main function for ASL recognition inference
    """
    # Parse arguments
    args = parse_arguments()
    
    # Initialize ASL inference
    asl_inference = ASLInference(
        model_path=args.model_path,
        seq_length=args.seq_length
    )
    
    # Load label map if provided
    if args.label_map:
        asl_inference.load_label_map(args.label_map)
    
    # Perform inference
    results = {}
    
    if args.input_pkl:
        # Single file inference
        predicted_class, class_name, confidence = asl_inference.predict_from_pkl(args.input_pkl)
        
        print(f"Prediction for {os.path.basename(args.input_pkl)}:")
        print(f"  Class: {class_name} (index: {predicted_class})")
        print(f"  Confidence: {confidence:.4f}")
        
        results[os.path.basename(args.input_pkl)] = {
            'class': int(predicted_class),
            'class_name': class_name,
            'confidence': float(confidence)
        }
    
    elif args.input_dir:
        # Batch inference for multiple files
        pkl_files = [f for f in os.listdir(args.input_dir) if f.endswith('.pkl')]
        
        for pkl_file in pkl_files:
            pkl_path = os.path.join(args.input_dir, pkl_file)
            
            try:
                predicted_class, class_name, confidence = asl_inference.predict_from_pkl(pkl_path)
                
                print(f"Prediction for {pkl_file}:")
                print(f"  Class: {class_name} (index: {predicted_class})")
                print(f"  Confidence: {confidence:.4f}")
                
                results[pkl_file] = {
                    'class': int(predicted_class),
                    'class_name': class_name,
                    'confidence': float(confidence)
                }
            except Exception as e:
                print(f"Error processing {pkl_file}: {e}")
    
    # Save results if output file is provided
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")

if __name__ == '__main__':
    main() 