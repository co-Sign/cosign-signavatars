import os
import argparse
import json
import shutil
import time
import random
import numpy as np
import glob
import pickle
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
import traceback
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pickle_loader import safe_load_pickle

def extract_features_from_smplx(smplx_params):
    """
    Extract relevant features from SMPL-X parameters.
    
    Args:
        smplx_params (numpy.ndarray): SMPL-X parameters
        
    Returns:
        numpy.ndarray: Extracted features
    """
    # SMPL-X parameters breakdown:
    # root_pose: 0:3
    # body_pose: 3:66
    # left_hand_pose: 66:111
    # right_hand_pose: 111:156
    # jaw_pose: 156:159
    # shape: 159:169
    # expression: 169:179
    # cam_trans: 179:182
    
    try:
        # Convert to numpy if it's a torch tensor
        if isinstance(smplx_params, torch.Tensor):
            smplx_params = smplx_params.cpu().numpy()
        
        # Handle different input formats
        if isinstance(smplx_params, dict):
            # If it's a dictionary, extract the relevant parts
            features = []
            # Body pose
            if 'body_pose' in smplx_params:
                body_pose = smplx_params['body_pose']
                if isinstance(body_pose, torch.Tensor):
                    body_pose = body_pose.cpu().numpy()
                features.append(body_pose.reshape(body_pose.shape[0], -1))
            
            # Hand poses
            if 'left_hand_pose' in smplx_params:
                left_hand = smplx_params['left_hand_pose']
                if isinstance(left_hand, torch.Tensor):
                    left_hand = left_hand.cpu().numpy()
                features.append(left_hand.reshape(left_hand.shape[0], -1))
            
            if 'right_hand_pose' in smplx_params:
                right_hand = smplx_params['right_hand_pose']
                if isinstance(right_hand, torch.Tensor):
                    right_hand = right_hand.cpu().numpy()
                features.append(right_hand.reshape(right_hand.shape[0], -1))
            
            # Global orientation
            if 'global_orient' in smplx_params:
                global_orient = smplx_params['global_orient']
                if isinstance(global_orient, torch.Tensor):
                    global_orient = global_orient.cpu().numpy()
                features.append(global_orient.reshape(global_orient.shape[0], -1))
            
            # Concatenate all features
            if features:
                return np.concatenate(features, axis=1)
            else:
                raise ValueError("No valid features found in SMPL-X dictionary")
        else:
            # If it's a numpy array with the standard format
            # Extract relevant features (body, hands, and jaw poses)
            body_pose = smplx_params[:, 3:66]  # 63 dims
            left_hand_pose = smplx_params[:, 66:111]  # 45 dims
            right_hand_pose = smplx_params[:, 111:156]  # 45 dims
            jaw_pose = smplx_params[:, 156:159]  # 3 dims
            
            # Concatenate features
            features = np.concatenate([body_pose, left_hand_pose, right_hand_pose, jaw_pose], axis=1)  # 156 dims
            
            return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        traceback.print_exc()
        return None

def process_file(file_path, output_dir, label_mapping=None):
    """
    Process a single file and save the extracted features.
    
    Args:
        file_path (str): Path to the input file
        output_dir (str): Directory to save the processed file
        label_mapping (dict): Optional mapping from file names to labels
        
    Returns:
        tuple: (file_path, success, error_message)
    """
    try:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, f"{file_name}.npy")
        metadata_path = os.path.join(output_dir, f"{file_name}_metadata.json")
        
        # Skip if already processed
        if os.path.exists(output_path) and os.path.exists(metadata_path):
            return file_path, True, "Already processed"
        
        # Determine file type and load
        if file_path.endswith('.pkl'):
            try:
                # Try with custom pickle loader that doesn't require 'device' parameter
                data = None
                try:
                    # First try without device parameter
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f, encoding='latin1')
                except:
                    # Fall back to our safe loader
                    try:
                        data = safe_load_pickle(file_path)
                    except TypeError:
                        # If we get a TypeError, maybe it's the 'device' parameter issue
                        import inspect
                        sig = inspect.signature(safe_load_pickle)
                        if 'device' in sig.parameters:
                            # If the function actually wants 'device' parameter, use CPU
                            from utils.pickle_loader import safe_load_pickle as safe_load_pickle_with_device
                            data = safe_load_pickle_with_device(file_path, device='cpu')
                
                # Check if data is loaded properly
                if data is None:
                    return file_path, False, "Failed to load pickle data"
                
                # Extract SMPL-X parameters
                if isinstance(data, dict):
                    if 'smplx' in data:
                        smplx_params = data['smplx']
                    elif 'body_pose' in data:
                        # It's already SMPL-X parameters
                        smplx_params = data
                    else:
                        # Try to find any tensor or array with the right shape
                        for key, value in data.items():
                            if isinstance(value, (torch.Tensor, np.ndarray)) and len(value.shape) == 2:
                                smplx_params = value
                                break
                        else:
                            return file_path, False, "Could not find SMPL-X parameters in data"
                else:
                    smplx_params = data  # Assume the data itself is SMPL-X params
                
            except Exception as e:
                return file_path, False, f"Error loading pickle: {e}"
        
        elif file_path.endswith('.npy'):
            try:
                smplx_params = np.load(file_path)
            except Exception as e:
                return file_path, False, f"Error loading numpy file: {e}"
        else:
            return file_path, False, f"Unsupported file format: {os.path.splitext(file_path)[1]}"
        
        # Extract features
        features = extract_features_from_smplx(smplx_params)
        
        if features is None:
            return file_path, False, "Failed to extract features"
        
        # Get sequence length
        seq_length = features.shape[0]
        
        # Get label
        if label_mapping and file_name in label_mapping:
            label = label_mapping[file_name]
        else:
            # Try to extract label from file name or path
            parts = file_name.split('_')
            if len(parts) > 1:
                label = parts[0]  # Use first part as label
            else:
                # Use directory name as label
                label = os.path.basename(os.path.dirname(file_path))
        
        # Save features
        np.save(output_path, features)
        
        # Save metadata
        metadata = {
            'file_name': file_name,
            'label': label,
            'seq_length': seq_length,
            'original_file': file_path,
            'feature_dim': features.shape[1],
            'processed_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        return file_path, True, None
        
    except Exception as e:
        error_message = f"Error processing file: {e}"
        traceback.print_exc()
        return file_path, False, error_message

def load_label_mapping(mapping_file):
    """
    Load label mapping from a file.
    
    Args:
        mapping_file (str): Path to the mapping file (JSON)
        
    Returns:
        dict: Label mapping
    """
    if not os.path.exists(mapping_file):
        return {}
    
    try:
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        return mapping
    except Exception as e:
        print(f"Error loading label mapping: {e}")
        return {}

def preprocess_large_dataset(input_dir, output_dir, num_workers=4, limit=None, label_mapping_file=None):
    """
    Preprocess a large dataset in parallel.
    
    Args:
        input_dir (str): Directory containing input files
        output_dir (str): Directory to save processed files
        num_workers (int): Number of parallel workers
        limit (int): Limit the number of files to process (for testing)
        label_mapping_file (str): Path to a JSON file mapping file names to labels
        
    Returns:
        dict: Processing statistics
    """
    start_time = time.time()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all input files
    print(f"Searching for files in {input_dir}...")
    pkl_files = glob.glob(os.path.join(input_dir, '**/*.pkl'), recursive=True)
    npy_files = glob.glob(os.path.join(input_dir, '**/*.npy'), recursive=True)
    npy_files = [f for f in npy_files if not f.endswith('_metadata.npy')]
    
    all_files = pkl_files + npy_files
    
    if not all_files:
        print(f"No PKL or NPY files found in {input_dir}")
        return {
            "status": "failed",
            "reason": "No input files found",
            "processing_time": f"{time.time() - start_time:.2f} seconds"
        }
    
    # Shuffle files to distribute workload better
    random.shuffle(all_files)
    
    # Limit number of files if specified
    if limit and limit > 0:
        all_files = all_files[:limit]
    
    print(f"Found {len(all_files)} files to process")
    
    # Load label mapping if provided
    label_mapping = load_label_mapping(label_mapping_file) if label_mapping_file else {}
    
    # Process files in parallel
    results = []
    successful = 0
    failed = 0
    skipped = 0
    errors = {}
    
    # Adjust number of workers based on system
    if os.name == 'nt':  # Windows
        num_workers = min(num_workers, 8)  # Windows might have issues with too many processes
    
    # Try a single file first to check for issues
    print("Testing processing with a single file...")
    if all_files:
        test_file = all_files[0]
        print(f"Testing with file: {test_file}")
        file_path, success, error_message = process_file(test_file, output_dir, label_mapping)
        if success:
            print("Test processing successful!")
        else:
            print(f"Test processing failed: {error_message}")
            # Continue anyway, but log the error
    
    # Process files with progress bar
    with tqdm(total=len(all_files), desc="Processing files") as pbar:
        # For small datasets or if num_workers is 1, use sequential processing
        if num_workers <= 1 or len(all_files) < 10:
            for file_path in all_files:
                file_path, success, error_message = process_file(file_path, output_dir, label_mapping)
                
                if success:
                    if error_message == "Already processed":
                        skipped += 1
                    else:
                        successful += 1
                else:
                    failed += 1
                    file_name = os.path.basename(file_path)
                    errors[file_name] = error_message
                
                pbar.update(1)
                pbar.set_postfix({
                    'success': successful, 
                    'failed': failed, 
                    'skipped': skipped
                })
        else:
            # Use parallel processing
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit tasks
                future_to_file = {
                    executor.submit(process_file, file_path, output_dir, label_mapping): file_path
                    for file_path in all_files
                }
                
                # Process results as they complete
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        file_path, success, error_message = future.result()
                        
                        if success:
                            if error_message == "Already processed":
                                skipped += 1
                            else:
                                successful += 1
                        else:
                            failed += 1
                            file_name = os.path.basename(file_path)
                            errors[file_name] = error_message
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            'success': successful, 
                            'failed': failed, 
                            'skipped': skipped
                        })
                        
                    except Exception as exc:
                        failed += 1
                        file_name = os.path.basename(file_path)
                        errors[file_name] = str(exc)
                        pbar.update(1)
    
    # Save preprocessing report
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_dir": input_dir,
        "output_dir": output_dir,
        "total_files": len(all_files),
        "successful_files": successful,
        "failed_files": failed,
        "skipped_files": skipped,
        "errors": errors,
        "processing_time": f"{time.time() - start_time:.2f} seconds"
    }
    
    report_path = os.path.join(output_dir, "preprocessing_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create a label mapping file from processed data
    create_label_mapping(output_dir)
    
    print("\nPreprocessing Summary:")
    print(f"Processed {len(all_files)} files with {successful} successful, {failed} failed, and {skipped} skipped")
    print(f"Processing time: {report['processing_time']}")
    print(f"Report saved to {report_path}")
    
    return report

def create_label_mapping(processed_dir):
    """
    Create a label mapping file from processed data.
    
    Args:
        processed_dir (str): Directory containing processed files
    """
    metadata_files = glob.glob(os.path.join(processed_dir, '*_metadata.json'))
    
    label_mapping = {}
    label_counts = {}
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            file_name = metadata['file_name']
            label = metadata['label']
            
            label_mapping[file_name] = label
            
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
                
        except Exception as e:
            print(f"Error processing metadata file {metadata_file}: {e}")
    
    # Save label mapping
    mapping_path = os.path.join(processed_dir, "label_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    # Save label statistics
    stats_path = os.path.join(processed_dir, "label_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump({
            "total_samples": len(label_mapping),
            "unique_labels": len(label_counts),
            "label_counts": label_counts
        }, f, indent=2)
    
    print(f"Created label mapping with {len(label_mapping)} samples and {len(label_counts)} unique labels")
    print(f"Label mapping saved to {mapping_path}")
    print(f"Label statistics saved to {stats_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess large SMPL-X dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw data files")
    parser.add_argument("--output_dir", type=str, default="../data/preprocessed_large", help="Directory to save processed data")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, help="Limit the number of files to process (for testing)")
    parser.add_argument("--label_mapping", type=str, help="Path to a JSON file mapping file names to labels")
    
    args = parser.parse_args()
    
    preprocess_large_dataset(
        args.input_dir, 
        args.output_dir, 
        args.num_workers, 
        args.limit, 
        args.label_mapping
    ) 