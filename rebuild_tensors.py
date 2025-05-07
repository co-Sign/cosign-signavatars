#!/usr/bin/env python3
"""
Script to patch torch for loading CUDA tensors on CPU
"""
import os
import sys
import argparse
import torch
import pickle
import numpy as np
from pathlib import Path
import io
import re

def parse_args():
    parser = argparse.ArgumentParser(description='Convert CUDA-serialized files to CPU-compatible format')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing .pkl files')
    parser.add_argument('--output-dir', type=str, default='data/cpu_samples',
                       help='Directory to save processed files')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process (None for all)')
    return parser.parse_args()

def patch_tensor_deserialization():
    """
    Patch torch's tensor deserialization to handle CUDA tensors on CPU
    """
    # Save original functions that we'll replace
    original_validate_device = torch.serialization._validate_device
    
    # Create patched version of _validate_device
    def patched_validate_device(location, backend_name):
        if location.startswith('cuda'):
            # Replace CUDA device with CPU
            location = 'cpu'
        return original_validate_device(location, backend_name)
    
    # Apply the patch
    torch.serialization._validate_device = patched_validate_device
    
    # Return a function to restore the original behavior
    def restore_original():
        torch.serialization._validate_device = original_validate_device
    
    return restore_original

def binary_cuda_to_cpu(file_path):
    """
    Modify the pickle file's binary content, replacing CUDA references with CPU
    """
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # Replace CUDA references with CPU
    modified = content.replace(b'cuda:', b'cpu:')
    modified = modified.replace(b'torch.cuda', b'torch.cpu')
    
    # Create an in-memory file
    buffer = io.BytesIO(modified)
    
    try:
        # Try to load with torch.load
        data = torch.load(buffer, map_location='cpu', weights_only=False)
        return data
    except Exception as e:
        print(f"  Binary modification failed: {e}")
        return None

def safe_load_tensor(file_path):
    """
    Try different methods to load a tensor from a CUDA-serialized file
    """
    # Method 1: Apply the patch to torch.serialization._validate_device
    try:
        restore_func = patch_tensor_deserialization()
        try:
            data = torch.load(file_path, map_location='cpu', weights_only=False)
            print("  Successfully loaded with patched torch.serialization")
            return data
        finally:
            restore_func()
    except Exception as e:
        print(f"  Method 1 failed: {str(e)}")
    
    # Method 2: Try binary content modification
    try:
        data = binary_cuda_to_cpu(file_path)
        if data is not None:
            print("  Successfully loaded with binary modification")
            return data
    except Exception as e:
        print(f"  Method 2 failed: {str(e)}")
    
    # Method 3: Try with pickle directly and handle exceptions
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  Method 3 failed: {str(e)}")
    
    print("  All methods failed")
    return None

def extract_tensor_data(data):
    """
    Extract the main tensor data from the loaded object
    """
    if data is None:
        return None
    
    if isinstance(data, dict):
        # Look for 'smplx' key which contains the motion data
        if 'smplx' in data:
            tensor = data['smplx']
            if isinstance(tensor, torch.Tensor):
                return tensor.detach().cpu().numpy()
        
        # If no 'smplx' key, try to find any tensor with good shape
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                if len(value.shape) == 2 and value.shape[1] > 100:
                    # This might be the motion data
                    print(f"  Using tensor with key '{key}', shape {value.shape}")
                    return value.detach().cpu().numpy()
    
    # If data is already a tensor
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    
    # If data is already a numpy array
    if isinstance(data, np.ndarray):
        return data
    
    return None

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all pickle files
    data_dir = Path(args.data_dir)
    pkl_files = list(data_dir.glob('**/*.pkl'))
    
    if args.max_files:
        pkl_files = pkl_files[:args.max_files]
    
    print(f"Processing {len(pkl_files)} pickle files...")
    
    # Track results
    successful = 0
    failed = 0
    conversion_map = {}
    
    # Process each file
    for i, file_path in enumerate(pkl_files):
        print(f"[{i+1}/{len(pkl_files)}] Processing {file_path}")
        
        # Load the file
        data = safe_load_tensor(file_path)
        
        if data is None:
            failed += 1
            continue
        
        # Extract the tensor data
        tensor_data = extract_tensor_data(data)
        
        if tensor_data is None:
            print("  Failed to extract tensor data")
            failed += 1
            continue
        
        # Save as numpy file
        output_path = Path(args.output_dir) / f"{file_path.stem}.npy"
        np.save(str(output_path), tensor_data)
        print(f"  Saved to {output_path}")
        
        # Record in conversion map
        conversion_map[file_path.name] = output_path.name
        successful += 1
    
    # Save conversion map
    import json
    map_path = Path(args.output_dir) / "conversion_map.json"
    with open(map_path, 'w') as f:
        json.dump(conversion_map, f, indent=2)
    
    print(f"\nConversion complete:")
    print(f"  Successfully converted: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Conversion map saved to {map_path}")

if __name__ == "__main__":
    main() 