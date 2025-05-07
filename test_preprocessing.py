#!/usr/bin/env python3
"""
Test script for preprocessing data with CUDA-pickled files
"""
import os
import sys
import argparse
from src.preprocessing import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Test preprocessing of pkl files')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing .pkl files')
    parser.add_argument('--seq-length', type=int, default=50,
                        help='Sequence length for padding/truncating')
    parser.add_argument('--labels-file', type=str, default=None,
                        help='Optional JSON file with labels for sequences')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Testing preprocessing with data from: {args.data_dir}")
    
    # Create data loader
    data_loader = DataLoader(
        data_dir=args.data_dir,
        seq_length=args.seq_length
    )
    
    # Load dataset
    print("Loading dataset...")
    features, labels, file_names = data_loader.load_dataset(labels_file=args.labels_file)
    
    # Print stats
    if features is not None and len(features) > 0:
        print(f"Successfully loaded {len(features)} sequences")
        print(f"Feature shape: {features.shape}")
        
        if labels is not None:
            print(f"Found {len(set(labels))} unique labels")
        
        print("\nFirst 5 files processed:")
        for i in range(min(5, len(file_names))):
            print(f"  - {file_names[i]}")
    else:
        print("No valid sequences were loaded!")
        
    # Save a sample of the processed data
    import numpy as np
    output_dir = "data/cpu_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    if features is not None and len(features) > 0:
        # Save the first sequence as a sample
        sample_file = os.path.join(output_dir, "sample_sequence.npy")
        np.save(sample_file, features[0])
        print(f"\nSaved sample sequence to {sample_file}")
    
    print("\nPreprocessing test completed.")

if __name__ == "__main__":
    main() 