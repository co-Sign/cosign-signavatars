#!/usr/bin/env python3
"""
Simple test script for loading and inspecting pkl files
"""
import os
import sys
import argparse
from load_pkl_helper import load_pkl_file
import torch
import numpy as np
import traceback

def parse_args():
    parser = argparse.ArgumentParser(description='Test loading of pkl files')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Directory containing .pkl files')
    parser.add_argument('--max-files', type=int, default=5,
                      help='Maximum number of files to process')
    parser.add_argument('--output-dir', type=str, default='data/cpu_samples',
                       help='Directory to save processed files')
    return parser.parse_args()

def extract_features(data):
    """
    Extract features from the loaded data structure
    """
    try:
        if not isinstance(data, dict):
            print(f"Warning: Data is not a dictionary, type: {type(data)}")
            
            # Try to handle some special cases
            if hasattr(data, 'smplx') and isinstance(data.smplx, torch.Tensor):
                print("Found 'smplx' attribute in the object")
                tensor = data.smplx
                if isinstance(tensor, torch.Tensor):
                    return tensor.cpu().numpy()
            return None
            
        # Extract SMPLX parameters from the data
        all_parameters = data.get('smplx', None)
        if all_parameters is None:
            print("No 'smplx' key found in data")
            # Try some other common keys
            for key in data.keys():
                print(f"Found key: {key}")
                if isinstance(data[key], (torch.Tensor, np.ndarray)):
                    print(f"  Shape: {data[key].shape}")
            return None
            
        # Convert to numpy if it's a torch tensor
        if isinstance(all_parameters, torch.Tensor):
            all_parameters = all_parameters.cpu().numpy()
            
        print(f"SMPLX tensor shape: {all_parameters.shape}")
        return all_parameters
    except Exception as e:
        print(f"Error extracting features: {e}")
        traceback.print_exc()
        return None

def main():
    args = parse_args()
    
    print(f"Testing loading pkl files from: {args.data_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all .pkl files in directory
    pkl_files = [f for f in os.listdir(args.data_dir) if f.endswith('.pkl')]
    print(f"Found {len(pkl_files)} .pkl files")
    
    # Process a subset of files
    successful = 0
    for i, pkl_file in enumerate(pkl_files[:args.max_files]):
        file_path = os.path.join(args.data_dir, pkl_file)
        print(f"\nProcessing file {i+1}/{min(args.max_files, len(pkl_files))}: {pkl_file}")
        
        try:
            data = load_pkl_file(file_path)
            
            if data is not None:
                successful += 1
                print("Successfully loaded data")
                
                # Print some info about the data
                if isinstance(data, dict):
                    print(f"Data keys: {list(data.keys())}")
                else:
                    print(f"Data type: {type(data)}")
                
                # Extract and inspect features
                features = extract_features(data)
                if features is not None:
                    # Save the processed data
                    sample_file = os.path.join(args.output_dir, f"{os.path.splitext(pkl_file)[0]}.npy")
                    np.save(sample_file, features)
                    print(f"Saved processed data to {sample_file}")
        except Exception as e:
            print(f"Unexpected error processing file {pkl_file}: {e}")
            traceback.print_exc()
    
    print(f"\nTest completed. Successfully loaded {successful}/{min(args.max_files, len(pkl_files))} files.")

if __name__ == "__main__":
    main() 