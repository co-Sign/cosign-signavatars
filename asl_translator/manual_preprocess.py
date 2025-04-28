"""
Simple script to copy and preprocess synthetic data.
"""

import os
import numpy as np
import json
import shutil
import argparse

def preprocess_data(input_dir, output_dir):
    """
    Copy and preprocess data from input_dir to output_dir.
    
    Args:
        input_dir (str): Directory containing synthetic data
        output_dir (str): Directory to save preprocessed data
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all NPY files
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    
    if not npy_files:
        print(f"No NPY files found in {input_dir}")
        return
    
    # Create a mapping from file name to label
    label_to_idx = {}
    idx_to_label = {}
    labels = []
    
    # Process each file
    for file_name in npy_files:
        base_name = os.path.splitext(file_name)[0]
        
        # Load the data
        data_path = os.path.join(input_dir, file_name)
        data = np.load(data_path)
        
        # Extract features
        # SMPL-X parameters breakdown:
        # root_pose: 0:3 (3 dimensions)
        # body_pose: 3:66 (63 dimensions)
        # left_hand_pose: 66:111 (45 dimensions)
        # right_hand_pose: 111:156 (45 dimensions)
        # jaw_pose: 156:159 (3 dimensions)
        # shape: 159:169 (10 dimensions)
        # expression: 169:179 (10 dimensions)
        # cam_trans: 179:182 (3 dimensions)
        
        # Extract relevant features (body, hands, and jaw poses)
        body_pose = data[:, 3:66]  # 63 dims
        left_hand_pose = data[:, 66:111]  # 45 dims
        right_hand_pose = data[:, 111:156]  # 45 dims
        jaw_pose = data[:, 156:159]  # 3 dims
        
        # Concatenate features
        features = np.concatenate([body_pose, left_hand_pose, right_hand_pose, jaw_pose], axis=1)  # 156 dims
        
        # Save features
        output_path = os.path.join(output_dir, f"{base_name}.npy")
        np.save(output_path, features)
        
        # Get metadata
        metadata_path = os.path.join(input_dir, f"{base_name}_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Save metadata
            output_metadata_path = os.path.join(output_dir, f"{base_name}_metadata.json")
            with open(output_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Add label to list
            label = metadata.get('label', base_name)
            if label not in label_to_idx:
                label_idx = len(label_to_idx)
                label_to_idx[label] = label_idx
                idx_to_label[label_idx] = label
            
            labels.append(label)
    
    # Save label mappings
    label_mappings = {
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label
    }
    
    with open(os.path.join(output_dir, "label_mappings.json"), 'w') as f:
        json.dump(label_mappings, f, indent=2)
    
    # Copy data.json if it exists
    data_json_path = os.path.join(input_dir, "data.json")
    if os.path.exists(data_json_path):
        shutil.copy(data_json_path, os.path.join(output_dir, "data.json"))
    
    print(f"Preprocessed {len(npy_files)} files")
    print(f"Created {len(label_to_idx)} labels: {', '.join(label_to_idx.keys())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess synthetic data")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing synthetic data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save preprocessed data")
    
    args = parser.parse_args()
    
    preprocess_data(args.input_dir, args.output_dir) 