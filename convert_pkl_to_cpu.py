#!/usr/bin/env python3
"""
Script to convert CUDA pickle files to NumPy format for CPU compatibility
"""
import os
import sys
import argparse
import numpy as np
import torch
import pickle
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Convert CUDA pickle files to NumPy arrays')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Directory containing .pkl files')
    parser.add_argument('--output-dir', type=str, default='data/cpu_samples',
                       help='Directory to save processed files')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process (None for all)')
    return parser.parse_args()

def safe_cuda_load(file_path):
    """
    Attempt to load a CUDA pickle file safely, with multiple fallback methods
    """
    try:
        # Try using pickle module with a relaxed security setting
        with open(file_path, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
        return pickle_data
    except Exception as e1:
        print(f"  Regular pickle load failed: {e1}")
        
        try:
            # Try torch.load with weights_only=False
            data = torch.load(file_path, map_location='cpu', weights_only=False)
            return data
        except Exception as e2:
            print(f"  Torch load failed: {e2}")
            
            try:
                # Try with mmap_mode='r' for large files
                with open(file_path, 'rb') as f:
                    pickle_data = pickle.load(f)
                return pickle_data
            except Exception as e3:
                print(f"  All methods failed: {e3}")
                return None

def extract_tensor_data(data):
    """
    Extract tensor data from various data structures and convert to NumPy
    """
    if isinstance(data, dict):
        # Try to find smplx key which contains the motion data
        if 'smplx' in data:
            tensor = data['smplx']
            if isinstance(tensor, torch.Tensor):
                return tensor.detach().cpu().numpy()
        
        # If no specific key is found, save all tensor values
        result = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.detach().cpu().numpy()
            elif isinstance(value, np.ndarray):
                result[key] = value
        return result
    
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    
    elif isinstance(data, np.ndarray):
        return data
    
    else:
        print(f"  Unsupported data type: {type(data)}")
        return None

def main():
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all .pkl files
    data_dir = Path(args.data_dir)
    pkl_files = list(data_dir.glob('**/*.pkl'))
    
    if args.max_files:
        pkl_files = pkl_files[:args.max_files]
    
    print(f"Found {len(pkl_files)} .pkl files, processing...")
    
    # Process files
    successful = 0
    failed = 0
    conversion_map = {}
    
    for i, pkl_path in enumerate(pkl_files):
        print(f"[{i+1}/{len(pkl_files)}] Processing {pkl_path}")
        
        # Load data
        data = safe_cuda_load(str(pkl_path))
        
        if data is None:
            print(f"  Failed to load file")
            failed += 1
            continue
        
        # Extract numerical data
        tensor_data = extract_tensor_data(data)
        
        if tensor_data is None:
            print(f"  Failed to extract tensors")
            failed += 1
            continue
        
        # Save as NumPy file
        output_filename = f"{pkl_path.stem}.npy"
        output_path = Path(args.output_dir) / output_filename
        
        try:
            np.save(str(output_path), tensor_data)
            print(f"  Saved to {output_path}")
            successful += 1
            
            # Record mapping
            conversion_map[str(pkl_path.name)] = str(output_path.name)
        except Exception as e:
            print(f"  Failed to save: {e}")
            failed += 1
    
    # Save conversion map
    map_path = Path(args.output_dir) / "conversion_map.json"
    with open(map_path, 'w') as f:
        json.dump(conversion_map, f, indent=2)
    
    print(f"\nConversion complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Conversion map saved to {map_path}")

if __name__ == "__main__":
    main() 