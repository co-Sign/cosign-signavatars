#!/usr/bin/env python3
"""
Script to create synthetic sample data for training
"""
import os
import numpy as np
import json
import random
from pathlib import Path

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Create synthetic sample data')
    parser.add_argument('--output-dir', type=str, default='data/cpu_samples',
                      help='Directory to save sample data')
    parser.add_argument('--num-samples', type=int, default=100,
                      help='Number of sample sequences to generate')
    parser.add_argument('--seq-length', type=int, default=50,
                      help='Length of each sequence')
    parser.add_argument('--feature-dim', type=int, default=182,
                      help='Feature dimension (default matches SMPLX dimension)')
    parser.add_argument('--num-classes', type=int, default=10,
                      help='Number of classes for labels')
    return parser.parse_args()

def create_sample_sequence(seq_length, feature_dim, class_id):
    """
    Create a synthetic motion sequence with some class-specific patterns
    """
    # Base random sequence
    sequence = np.random.randn(seq_length, feature_dim) * 0.1
    
    # Add class-specific patterns to make classification possible
    for i in range(seq_length):
        # Add sinusoidal pattern with frequency based on class_id
        freq = (class_id + 1) / 10.0
        sequence[i, :3] += np.sin(i * freq) * 0.5
        sequence[i, 3:6] += np.cos(i * freq) * 0.5
        
        # Add class-specific offsets to certain parts
        if class_id % 2 == 0:  # Even classes
            sequence[i, 66:111] += 0.2  # Emphasize left hand
        else:  # Odd classes
            sequence[i, 111:156] += 0.2  # Emphasize right hand
    
    return sequence

def create_labels_file(samples, output_dir, num_classes):
    """
    Create a labels file mapping sample filenames to class labels
    """
    labels = {}
    for sample in samples:
        # Extract number from filename (sample_001.npy -> 1)
        sample_id = int(sample.stem.split('_')[1])
        
        # Assign class based on sample_id
        class_id = sample_id % num_classes
        
        # Store with .pkl extension for consistency with original workflow
        pkl_name = f"{sample.stem}.pkl"
        labels[sample.stem] = f"class_{class_id:02d}"
    
    # Save labels file
    labels_file = os.path.join(output_dir, "labels.json")
    with open(labels_file, 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"Created labels file at {labels_file}")
    return labels_file

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate sample sequences
    print(f"Generating {args.num_samples} sample sequences...")
    sample_files = []
    
    for i in range(args.num_samples):
        # Assign a class to this sample
        class_id = i % args.num_classes
        
        # Create a synthetic sequence
        sequence = create_sample_sequence(args.seq_length, args.feature_dim, class_id)
        
        # Save as .npy file
        filename = f"sample_{i:03d}.npy"
        output_path = os.path.join(args.output_dir, filename)
        np.save(output_path, sequence)
        
        sample_files.append(Path(output_path))
        
        if (i+1) % 10 == 0:
            print(f"  Created {i+1}/{args.num_samples} samples")
    
    # Create labels file
    labels_file = create_labels_file(sample_files, args.output_dir, args.num_classes)
    
    # Create a metadata file
    metadata = {
        "num_samples": args.num_samples,
        "seq_length": args.seq_length,
        "feature_dim": args.feature_dim,
        "num_classes": args.num_classes,
        "labels_file": labels_file
    }
    
    metadata_file = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSample data creation complete:")
    print(f"  - {args.num_samples} sequences")
    print(f"  - {args.num_classes} classes")
    print(f"  - Sequences shape: ({args.seq_length}, {args.feature_dim})")
    print(f"  - Data directory: {args.output_dir}")
    print(f"  - Metadata: {metadata_file}")
    print(f"  - Labels: {labels_file}")

if __name__ == "__main__":
    main() 