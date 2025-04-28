"""
Create synthetic test data for the ASL translation pipeline.
This allows testing the pipeline without requiring the actual PKL files.
"""

import os
import numpy as np
import json
import argparse

def create_synthetic_smplx_data(num_frames=100, num_samples=10):
    """
    Create synthetic SMPL-X data for testing.
    
    Args:
        num_frames (int): Number of frames per sequence
        num_samples (int): Number of sequences to generate
        
    Returns:
        list: List of dictionaries containing synthetic data
    """
    # Expected structure of SMPL-X data
    # root_pose: 0:3 (3 dimensions)
    # body_pose: 3:66 (63 dimensions)
    # left_hand_pose: 66:111 (45 dimensions)
    # right_hand_pose: 111:156 (45 dimensions)
    # jaw_pose: 156:159 (3 dimensions)
    # shape: 159:169 (10 dimensions)
    # expression: 169:179 (10 dimensions)
    # cam_trans: 179:182 (3 dimensions)
    
    samples = []
    
    for i in range(num_samples):
        # Generate random motion data with realistic patterns
        # Body pose varies smoothly over time
        body_pose = np.sin(np.linspace(0, 4*np.pi, num_frames).reshape(-1, 1) * 
                          np.random.rand(1, 63)) * 0.5
        
        # Hand poses with more rapid movements
        left_hand_pose = np.sin(np.linspace(0, 8*np.pi, num_frames).reshape(-1, 1) * 
                               np.random.rand(1, 45)) * 0.8
        right_hand_pose = np.sin(np.linspace(0, 8*np.pi, num_frames).reshape(-1, 1) * 
                                np.random.rand(1, 45)) * 0.8
        
        # Root pose and jaw pose with minimal movement
        root_pose = np.random.randn(num_frames, 3) * 0.1
        jaw_pose = np.random.randn(num_frames, 3) * 0.1
        
        # Static parameters
        shape = np.random.randn(num_frames, 10) * 0.1
        shape = np.tile(shape[0:1], (num_frames, 1))  # Same shape for all frames
        
        expression = np.random.randn(num_frames, 10) * 0.1
        cam_trans = np.zeros((num_frames, 3))  # No camera translation
        
        # Combine all parameters
        smplx_params = np.concatenate([
            root_pose,       # 0:3
            body_pose,       # 3:66
            left_hand_pose,  # 66:111
            right_hand_pose, # 111:156
            jaw_pose,        # 156:159
            shape,           # 159:169
            expression,      # 169:179
            cam_trans        # 179:182
        ], axis=1)
        
        # Create data sample with proper structure
        sample = {
            'width': 1280,
            'height': 720,
            'focal': np.array([[5000.0, 5000.0]] * num_frames),
            'princpt': np.array([[640.0, 360.0]] * num_frames),
            'smplx': smplx_params,
            'unsmooth_smplx': smplx_params[:, :169],  # Excluding expression and cam_trans
            'total_valid_index': np.arange(num_frames),
            'left_valid': np.ones(num_frames),
            'right_valid': np.ones(num_frames),
            'bb2img_trans': np.zeros((num_frames, 2, 3))
        }
        
        samples.append(sample)
    
    return samples

def create_metadata(num_samples=10):
    """
    Create metadata for the synthetic data.
    
    Args:
        num_samples (int): Number of samples
        
    Returns:
        dict: Metadata dictionary
    """
    # Create fake gloss labels
    glosses = ["HELLO", "THANK_YOU", "PLEASE", "SORRY", "YES", "NO", "HELP", "WANT", "NAME", "SIGN"]
    metadata = {}
    
    for i in range(num_samples):
        sample_id = f"sample_{i+1:03d}"
        gloss = glosses[i % len(glosses)]
        metadata[sample_id] = {
            "hamnosys_text": f"hamnosys_{i+1}",
            "gloss": gloss
        }
    
    return metadata

def save_synthetic_data(output_dir, num_frames=100, num_samples=10):
    """
    Generate and save synthetic data for testing.
    
    Args:
        output_dir (str): Directory to save the synthetic data
        num_frames (int): Number of frames per sequence
        num_samples (int): Number of samples to generate
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate synthetic data
    samples = create_synthetic_smplx_data(num_frames, num_samples)
    metadata = create_metadata(num_samples)
    
    # Save data samples as NPY files directly
    for i, sample in enumerate(samples):
        sample_id = f"sample_{i+1:03d}"
        
        # Extract just the SMPL-X parameters and save as NPY
        smplx_params = sample['smplx']
        output_path = os.path.join(output_dir, f"{sample_id}.npy")
        np.save(output_path, smplx_params)
        
        # Save metadata for each sample
        metadata_path = os.path.join(output_dir, f"{sample_id}_metadata.json")
        sample_metadata = {
            'file_name': sample_id,
            'label': metadata[sample_id]['gloss'],
            'seq_length': num_frames
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(sample_metadata, f, indent=2)
    
    # Save global metadata
    with open(os.path.join(output_dir, "data.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created {num_samples} synthetic data samples in {output_dir}")
    print(f"Each sample has {num_frames} frames")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create synthetic test data for ASL translation")
    parser.add_argument("--output_dir", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--num_frames", type=int, default=100, help="Number of frames per sequence")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    
    args = parser.parse_args()
    
    save_synthetic_data(args.output_dir, args.num_frames, args.num_samples) 