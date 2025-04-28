"""
Test inference with a trained ASL translation model.
"""

import os
import argparse
import torch
import numpy as np
import json
import glob
from models.lstm_model import ASLSequenceClassifier, load_model

def load_label_mappings(mappings_path):
    """
    Load label mappings from a JSON file.
    
    Args:
        mappings_path (str): Path to the label mappings JSON file
        
    Returns:
        tuple: (label_to_idx, idx_to_label) dictionaries
    """
    with open(mappings_path, 'r') as f:
        mappings = json.load(f)
    
    label_to_idx = mappings['label_to_idx']
    idx_to_label = {}
    for label, idx in label_to_idx.items():
        idx_to_label[str(idx)] = label
    
    return label_to_idx, idx_to_label

def load_and_predict(model_path, label_mappings_path, feature_file, device=None):
    """
    Load a model and make a prediction on a feature file.
    
    Args:
        model_path (str): Path to the trained model
        label_mappings_path (str): Path to the label mappings JSON file
        feature_file (str): Path to the feature file (NPY)
        device (torch.device, optional): Device to use for inference
        
    Returns:
        str: Predicted label
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load label mappings
    _, idx_to_label = load_label_mappings(label_mappings_path)
    
    # Load model parameters
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, "..", "config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        hidden_dim = config.get('hidden_dim', 256)
        num_layers = config.get('num_layers', 2)
    else:
        print("Config file not found, using default parameters")
        hidden_dim = 256
        num_layers = 2
    
    # Load features
    features = np.load(feature_file)
    
    # Ensure features have the right shape
    if len(features.shape) == 2:
        # Add batch dimension if needed
        features = features[np.newaxis, :, :]
    
    # Get input dimension from features
    input_dim = features.shape[2]
    
    # Convert to tensor
    features_tensor = torch.tensor(features).float().to(device)
    
    # Create a mask for valid sequence positions
    seq_length = features.shape[1]
    mask = torch.ones((1, seq_length)).to(device)
    
    # Load model
    model_params = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'num_classes': len(idx_to_label),
        'dropout': 0.2,
        'bidirectional': True
    }
    
    model = load_model(model_path, model_params, device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(features_tensor, mask)
        _, predicted = torch.max(outputs, 1)
        predicted_idx = predicted.item()
    
    # Get label from index
    predicted_label = idx_to_label.get(str(predicted_idx), f"Unknown label {predicted_idx}")
    
    return predicted_label

def test_inference(model_path, label_mappings_path, test_dir, num_samples=3):
    """
    Test inference on multiple files from a test directory.
    
    Args:
        model_path (str): Path to the trained model
        label_mappings_path (str): Path to the label mappings JSON file
        test_dir (str): Directory containing test files
        num_samples (int): Number of samples to test
    """
    # Get feature files
    feature_files = glob.glob(os.path.join(test_dir, "*.npy"))
    
    if not feature_files:
        print(f"No feature files found in {test_dir}")
        return
    
    # Select a subset of files for testing
    if num_samples < len(feature_files):
        feature_files = feature_files[:num_samples]
    
    # Test each file
    for feature_file in feature_files:
        file_name = os.path.basename(feature_file)
        base_name = os.path.splitext(file_name)[0]
        
        # Get ground truth label if available
        metadata_path = os.path.join(test_dir, f"{base_name}_metadata.json")
        ground_truth = None
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                ground_truth = metadata.get('label', None)
        
        # Make prediction
        predicted_label = load_and_predict(model_path, label_mappings_path, feature_file)
        
        # Print results
        if ground_truth:
            print(f"File: {file_name} - Predicted: {predicted_label} - Ground Truth: {ground_truth} - Correct: {predicted_label == ground_truth}")
        else:
            print(f"File: {file_name} - Predicted: {predicted_label}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test inference with a trained ASL translation model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--label_mappings_path", type=str, required=True, help="Path to the label mappings JSON file")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing test files")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to test")
    
    args = parser.parse_args()
    
    test_inference(args.model_path, args.label_mappings_path, args.test_dir, args.num_samples) 