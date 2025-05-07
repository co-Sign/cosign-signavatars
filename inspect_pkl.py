#!/usr/bin/env python3
"""
Script to inspect pickle files in more detail
"""
import os
import sys
import argparse
import torch
import pickle
import numpy as np
import traceback

def parse_args():
    parser = argparse.ArgumentParser(description='Inspect pickle files in detail')
    parser.add_argument('--file', type=str, required=True,
                       help='Path to the pickle file')
    return parser.parse_args()

def load_with_cpu_map_no_weights(file_path):
    """Try to load with weights_only=False"""
    return torch.load(file_path, map_location='cpu', weights_only=False)

def load_pickle_with_patch():
    """Monkey patch _rebuild_tensor_v2 to handle CUDA tensors"""
    import torch.cuda
    
    # Save the original _rebuild_tensor_v2
    original_rebuild = torch.cuda._rebuild_tensor_v2
    
    # Define a function to handle rebuilding from CPU
    def _cpu_rebuild_tensor_v2(*args, **kwargs):
        try:
            return original_rebuild(*args, **kwargs)
        except RuntimeError:
            # Modify the 3rd argument (device) to 'cpu'
            args_list = list(args)
            if len(args_list) > 2:
                if args_list[2].startswith('cuda'):
                    args_list[2] = 'cpu'
            return torch._utils._rebuild_tensor_v2(*args_list)
    
    # Monkey patch the function
    torch.cuda._rebuild_tensor_v2 = _cpu_rebuild_tensor_v2
    
    # Return the restore function
    def restore_original():
        torch.cuda._rebuild_tensor_v2 = original_rebuild
    
    return restore_original

def main():
    args = parse_args()
    file_path = args.file
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist")
        return
    
    print(f"Inspecting pickle file: {file_path}")
    
    # Try different loading methods
    methods = [
        ("Method 1: Read file as binary and analyze", lambda: open(file_path, 'rb').read()[:100]),
        ("Method 2: Try with weights_only=False", lambda: load_with_cpu_map_no_weights(file_path)),
        ("Method 3: Try with pickle module", lambda: pickle.load(open(file_path, 'rb'))),
        ("Method 4: Try with latin1 encoding", lambda: pickle.load(open(file_path, 'rb'), encoding='latin1')),
        ("Method 5: Monkey patch torch.cuda._rebuild_tensor_v2", lambda: load_with_monkey_patch(file_path)),
    ]
    
    def load_with_monkey_patch(file_path):
        restore_func = load_pickle_with_patch()
        try:
            data = torch.load(file_path, map_location='cpu')
            return data
        finally:
            restore_func()
    
    # Try each method
    for name, method in methods:
        print(f"\n{name}:")
        try:
            result = method()
            
            if name == "Method 1: Read file as binary and analyze":
                print(f"First 100 bytes (hex): {result.hex()}")
                print(f"First 100 bytes (ASCII): {repr(result)}")
            else:
                print(f"Type: {type(result)}")
                if isinstance(result, dict):
                    print(f"Keys: {list(result.keys())}")
                    for key, value in result.items():
                        print(f"  - {key}: {type(value)}")
                        if hasattr(value, 'shape'):
                            print(f"    Shape: {value.shape}")
        except Exception as e:
            print(f"Failed: {e}")
            traceback.print_exc()
    
    print("\nInspection complete")

if __name__ == "__main__":
    main() 