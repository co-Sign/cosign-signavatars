import os
import sys
import torch
import numpy as np
import argparse
from load_pkl_helper import load_pkl_file, visualize_data

def main(file_path):
    """
    Load a single PKL file and display its contents.
    
    Args:
        file_path: Path to the PKL file
    """
    data = load_pkl_file(file_path)
    
    if data is None:
        print("Failed to load the file")
        return False
    
    # Display data structure
    visualize_data(data)
    
    # Check for SMPL-X parameters
    if isinstance(data, dict) and 'smplx' in data:
        print("\nSMPL-X Information:")
        smplx = data['smplx']
        
        if isinstance(smplx, torch.Tensor):
            print(f"SMPLX is a tensor with shape: {smplx.shape}")
            print(f"SMPLX device: {smplx.device}")
        elif isinstance(smplx, np.ndarray):
            print(f"SMPLX is a NumPy array with shape: {smplx.shape}")
        else:
            print(f"SMPLX is type: {type(smplx)}")
        
        # Print a small sample of the data
        if hasattr(smplx, 'shape') and len(smplx.shape) > 0:
            print("\nSample of SMPLX data (first 5 elements):")
            if isinstance(smplx, torch.Tensor):
                print(smplx[0, :5] if smplx.shape[0] > 0 else "Empty tensor")
            elif isinstance(smplx, np.ndarray):
                print(smplx[0, :5] if smplx.shape[0] > 0 else "Empty array")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test loading a single PKL file")
    parser.add_argument("file_path", help="Path to the PKL file")
    args = parser.parse_args()
    
    main(args.file_path) 