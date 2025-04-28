import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.data_processor import preprocess_dataset, extract_features
import pickle
import time
import torch
import json
from utils.pickle_loader import safe_load_pickle, extract_smplx_params, convert_tensor_to_numpy

def test_preprocessing(raw_data_dir, output_dir, max_seq_length=128):
    """
    Test the preprocessing pipeline.
    
    Args:
        raw_data_dir (str): Directory containing the raw data files (PKL or NPY)
        output_dir (str): Directory to save the preprocessed features
        max_seq_length (int): Maximum sequence length
    """
    print(f"Testing preprocessing pipeline with data from {raw_data_dir}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # First look for PKL files, then NPY files
    pkl_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.pkl')]
    npy_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.npy')]
    
    if not pkl_files and not npy_files:
        print(f"No PKL or NPY files found in {raw_data_dir}")
        return
    
    # Choose a sample file
    if pkl_files:
        sample_file = pkl_files[0]
        is_pkl = True
    else:
        sample_file = npy_files[0]
        is_pkl = False
    
    sample_path = os.path.join(raw_data_dir, sample_file)
    print(f"Processing sample file: {sample_file}")
    
    # Load the data
    try:
        if is_pkl:
            data = safe_load_pickle(sample_path)
            smplx_params = extract_smplx_params(data)
        else:
            # Load NPY file directly
            smplx_params = np.load(sample_path)
            
            # Try to load corresponding metadata if it exists
            metadata_path = os.path.join(raw_data_dir, f"{os.path.splitext(sample_file)[0]}_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"Loaded metadata: {metadata}")
        
        print(f"SMPL-X params shape: {smplx_params.shape}")
        
        # Extract features
        features = extract_features(smplx_params)
        print(f"Extracted features shape: {features.shape}")
        
        # Save a sample plot
        plt.figure(figsize=(10, 6))
        
        # Plot a subset of body pose parameters
        body_pose = smplx_params[:, 3:66]
        plt.subplot(2, 2, 1)
        plt.plot(body_pose[:, :10])
        plt.title('Body Pose Parameters (First 10)')
        plt.xlabel('Frame')
        plt.ylabel('Parameter Value')
        
        # Plot left hand pose parameters
        left_hand_pose = smplx_params[:, 66:111]
        plt.subplot(2, 2, 2)
        plt.plot(left_hand_pose[:, :10])
        plt.title('Left Hand Pose Parameters (First 10)')
        plt.xlabel('Frame')
        plt.ylabel('Parameter Value')
        
        # Plot right hand pose parameters
        right_hand_pose = smplx_params[:, 111:156]
        plt.subplot(2, 2, 3)
        plt.plot(right_hand_pose[:, :10])
        plt.title('Right Hand Pose Parameters (First 10)')
        plt.xlabel('Frame')
        plt.ylabel('Parameter Value')
        
        # Plot jaw pose parameters
        jaw_pose = smplx_params[:, 156:159]
        plt.subplot(2, 2, 4)
        plt.plot(jaw_pose)
        plt.title('Jaw Pose Parameters')
        plt.xlabel('Frame')
        plt.ylabel('Parameter Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{os.path.splitext(sample_file)[0]}_plot.png"))
        plt.close()
        
        # Save the features
        output_path = os.path.join(output_dir, f"{os.path.splitext(sample_file)[0]}_features.npy")
        np.save(output_path, features)
        print(f"Saved features to {output_path}")
        
        # Test our manual preprocessing
        print("Testing manual preprocessing...")
        for input_file in (pkl_files if is_pkl else npy_files):
            input_path = os.path.join(raw_data_dir, input_file)
            file_name = os.path.splitext(input_file)[0]
            
            try:
                if is_pkl:
                    data = safe_load_pickle(input_path)
                    params = extract_smplx_params(data)
                else:
                    params = np.load(input_path)
                
                # Extract features
                feats = extract_features(params)
                
                # Save features
                out_path = os.path.join(output_dir, f"{file_name}_features.npy")
                np.save(out_path, feats)
                
                # Create metadata
                metadata_path = os.path.join(raw_data_dir, f"{file_name}_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        file_metadata = json.load(f)
                    
                    # Save metadata
                    out_metadata_path = os.path.join(output_dir, f"{file_name}_metadata.json")
                    with open(out_metadata_path, 'w') as f:
                        json.dump(file_metadata, f, indent=2)
            except Exception as e:
                print(f"Error processing {input_file}: {e}")
        
        print(f"Preprocessed {len(pkl_files if is_pkl else npy_files)} files")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the preprocessing pipeline")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Directory containing the raw data files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the preprocessed features")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    test_preprocessing(args.raw_data_dir, args.output_dir, args.max_seq_length)