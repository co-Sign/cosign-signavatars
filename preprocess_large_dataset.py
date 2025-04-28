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
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from datetime import datetime
import traceback
import sys

# Add parent directory to path to be able to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our safe_load_pickle function
from utils.pickle_loader import safe_load_pickle

def extract_features_from_smplx(data, file_path=""):
    """
    Extract relevant features from SMPLX parameters
    
    Args:
        data: Dictionary or numpy array containing SMPLX parameters
        file_path: Path to the file for error reporting
        
    Returns:
        Extracted features as numpy array
    """
    try:
        # Print debug info about the data structure
        if isinstance(data, dict):
            print(f"Data is a dictionary with keys: {list(data.keys())}")
            # Try to analyze the data structure to find SMPL-X parameters
            
            # Common key patterns for SMPL-X
            possible_keys = ['smplx_params', 'body_pose', 'joints', 'poses', 'motions', 'pose_params']
            
            # Extract relevant features from the dict
            features = []
            
            # Try to find any key that might contain poses or joints
            for key in data.keys():
                if key in possible_keys or 'pose' in key.lower() or 'joint' in key.lower():
                    value = data[key]
                    if isinstance(value, torch.Tensor):
                        # Handle CUDA memory errors
                        if value.is_cuda:
                            try:
                                value = value.cpu().numpy()
                            except RuntimeError as e:
                                if "CUDA out of memory" in str(e):
                                    print(f"CUDA OOM error for {file_path}, skipping key {key}")
                                    continue
                                else:
                                    raise e
                        else:
                            value = value.numpy()
                    
                    # Make sure it's a numerical array
                    if isinstance(value, np.ndarray) and value.dtype.kind in 'fcib':
                        print(f"Found useful feature: {key} with shape {value.shape}")
                        # Only use 2D arrays (sequence, features)
                        if len(value.shape) == 2:
                            features.append(value)
                        elif len(value.shape) > 2:
                            # Reshape to 2D by flattening all but the first dimension
                            reshaped = value.reshape(value.shape[0], -1)
                            features.append(reshaped)
            
            # If we have found features, concatenate them
            if features:
                result = np.concatenate(features, axis=1)
                print(f"Final feature shape: {result.shape}")
                return result
            
            # If we haven't found anything useful yet, try some known patterns
            # Example: Extract joint positions if available
            if 'joints' in data:
                joints = data['joints']
                try:
                    if isinstance(joints, torch.Tensor):
                        # Handle CUDA memory errors
                        if joints.is_cuda:
                            try:
                                joints = joints.cpu().numpy()
                            except RuntimeError as e:
                                if "CUDA out of memory" in str(e):
                                    print(f"CUDA OOM error for {file_path}, skipping joints")
                                    # Skip this key by returning from the block
                                    pass  # Skip to next feature
                                else:
                                    raise e
                        else:
                            joints = joints.numpy()
                    
                    # Only add to features if we have valid data
                    if isinstance(joints, np.ndarray):
                        features.append(joints.reshape(joints.shape[0], -1))
                except Exception as e:
                    print(f"Error processing joints in {file_path}: {str(e)}")
                    if isinstance(joints, torch.Tensor):
                        print(f"Tensor device: {joints.device}, shape: {joints.shape}")
                    # Continue rather than raising an error
                
            # Example: Extract body pose if available
            if 'body_pose' in data:
                body_pose = data['body_pose']
                try:
                    if isinstance(body_pose, torch.Tensor):
                        # Handle CUDA memory errors
                        if body_pose.is_cuda:
                            try:
                                body_pose = body_pose.cpu().numpy()
                            except RuntimeError as e:
                                if "CUDA out of memory" in str(e):
                                    print(f"CUDA OOM error for {file_path}, skipping body_pose")
                                    # Skip this key by returning from the block
                                    pass  # Skip to next feature
                                else:
                                    raise e
                        else:
                            body_pose = body_pose.numpy()
                    
                    # Only add to features if we have valid data
                    if isinstance(body_pose, np.ndarray):
                        features.append(body_pose.reshape(body_pose.shape[0], -1))
                except Exception as e:
                    print(f"Error processing body_pose in {file_path}: {str(e)}")
                    if isinstance(body_pose, torch.Tensor):
                        print(f"Tensor device: {body_pose.device}, shape: {body_pose.shape}")
                    # Continue rather than raising an error
            
            # Combine all features
            if features:
                return np.concatenate(features, axis=1)
            else:
                raise ValueError(f"No usable features found in the data dictionary for {file_path}")
                
        elif isinstance(data, np.ndarray):
            # If data is already a numpy array, reshape if needed
            return data.reshape(data.shape[0], -1)
            
        elif isinstance(data, torch.Tensor):
            # Handle torch tensors
            print(f"Data is a torch tensor with shape: {data.shape}")
            numpy_data = data.cpu().numpy()
            if len(numpy_data.shape) >= 2:
                return numpy_data.reshape(numpy_data.shape[0], -1)
            else:
                raise ValueError(f"Data shape is not suitable for features: {numpy_data.shape}")
                
        elif isinstance(data, list):
            # Try to convert list to numpy array
            print(f"Data is a list with length: {len(data)}")
            try:
                numpy_data = np.array(data)
                if len(numpy_data.shape) >= 2:
                    return numpy_data.reshape(numpy_data.shape[0], -1)
                else:
                    raise ValueError(f"List data shape is not suitable for features: {numpy_data.shape}")
            except Exception as e:
                raise ValueError(f"Failed to convert list to numpy array: {str(e)}")
            
        else:
            raise TypeError(f"Unsupported data type: {type(data)} for {file_path}")
            
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"CUDA out of memory error in {file_path}. Clearing cache...")
            # Just clear the cache and raise the error for the caller to handle
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e
        else:
            print(f"Runtime error extracting features from {file_path}: {str(e)}")
            traceback.print_exc()
            raise e
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        traceback.print_exc()
        raise e

def process_file(args):
    """
    Process a single file, extracting features and saving to output directory
    
    Args:
        args: Tuple containing (file_path, output_dir, label_mapping)
        
    Returns:
        Dictionary with processing results
    """
    file_path, output_dir, label_mapping = args
    
    try:
        # Get filename without extension and label from it
        filename = os.path.basename(file_path)
        file_base = os.path.splitext(filename)[0]
        
        # Check if this file has already been processed
        output_file = os.path.join(output_dir, f"{file_base}.npy")
        if os.path.exists(output_file):
            return {
                "file": file_path,
                "status": "skipped",
                "reason": "Already processed"
            }
        
        # Extract label from filename (customize based on your naming convention)
        # For example, if filename format is "word_123.pkl", extract "word"
        label = file_base.split('_')[0]
        
        # Load data based on file extension
        if file_path.endswith('.pkl'):
            try:
                # Use our safe pickle loader that handles CUDA tensors on CPU
                data = safe_load_pickle(file_path)
            except Exception as e:
                return {
                    "file": file_path,
                    "status": "failed",
                    "reason": f"Failed to load pickle data: {str(e)}"
                }
        elif file_path.endswith('.npy'):
            try:
                data = np.load(file_path, allow_pickle=True)
            except Exception as e:
                return {
                    "file": file_path,
                    "status": "failed",
                    "reason": f"Failed to load numpy data: {str(e)}"
                }
        else:
            return {
                "file": file_path,
                "status": "failed",
                "reason": f"Unsupported file format: {os.path.splitext(file_path)[1]}"
            }
            
        # Extract features
        try:
            features = extract_features_from_smplx(data, file_path)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                # If CUDA OOM occurs during feature extraction, try one more time after clearing cache
                print(f"CUDA OOM during feature extraction for {file_path}, attempting to recover")
                try:
                    # Force garbage collection and clear CUDA cache
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Try reloading the data directly to CPU
                    data = safe_load_pickle(file_path)
                    features = extract_features_from_smplx(data, file_path)
                except Exception as e2:
                    return {
                        "file": file_path,
                        "status": "failed",
                        "reason": f"Feature extraction failed after recovery attempt: {str(e2)}"
                    }
            else:
                return {
                    "file": file_path,
                    "status": "failed",
                    "reason": f"Feature extraction failed with runtime error: {str(e)}"
                }
        except Exception as e:
            return {
                "file": file_path,
                "status": "failed",
                "reason": f"Feature extraction failed: {str(e)}"
            }
            
        # Save features
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_file, features)
        
        # Create metadata
        metadata = {
            "original_file": file_path,
            "label": label,
            "sequence_length": features.shape[0],
            "feature_dim": features.shape[1],
            "processed_date": datetime.now().isoformat()
        }
        
        # Save metadata
        with open(os.path.join(output_dir, f"{file_base}.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Update label mapping
        if label in label_mapping:
            label_mapping[label].append(file_base)
        else:
            label_mapping[label] = [file_base]
            
        return {
            "file": file_path,
            "status": "success",
            "label": label,
            "sequence_length": features.shape[0],
            "feature_dim": features.shape[1]
        }
        
    except Exception as e:
        return {
            "file": file_path,
            "status": "failed",
            "reason": f"Unexpected error: {str(e)}"
        }

def load_label_mapping(mapping_file):
    """
    Load label mapping from a JSON file
    
    Args:
        mapping_file: Path to the JSON file containing label mapping
        
    Returns:
        Dictionary mapping labels to file IDs
    """
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading label mapping: {str(e)}")
    return {}

def preprocess_large_dataset(input_dir, output_dir, num_workers=None, limit=None, mapping_file=None, batch_size=10):
    """
    Preprocess a large dataset in parallel
    
    Args:
        input_dir: Directory containing input files
        output_dir: Directory to save processed files
        num_workers: Number of processes to use (default: CPU count)
        limit: Maximum number of files to process (default: None)
        mapping_file: Path to save/load label mapping
        batch_size: Number of files to process in each batch to prevent memory issues
        
    Returns:
        Dictionary with preprocessing results
    """
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all files in the input directory
    print(f"Searching for files in {input_dir}...")
    files = []
    for ext in ['*.pkl', '*.npy']:
        pattern = os.path.join(input_dir, '**', ext)
        files.extend(glob.glob(pattern, recursive=True))
    
    if not files:
        print(f"No files found in {input_dir}")
        return {
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "total": 0,
            "time": 0
        }
    
    # Shuffle files for better load balancing
    random.shuffle(files)
    
    # Limit the number of files if specified
    if limit and limit > 0:
        files = files[:limit]
    
    print(f"Found {len(files)} files to process")
    
    # Load existing label mapping if available
    label_mapping = {}
    if mapping_file and os.path.exists(mapping_file):
        label_mapping = load_label_mapping(mapping_file)
    
    # Test processing with a single file first
    print("Testing processing with a single file...")
    if files:
        test_file = files[0]
        print(f"Testing with file: {test_file}")
        result = process_file((test_file, output_dir, {}))
        if result["status"] == "failed":
            print(f"Test processing failed: {result['reason']}")
        else:
            print(f"Test processing successful: {result['label']}")
    
    # Determine number of workers
    if num_workers is None:
        # Use fewer workers than CPU count to avoid memory issues
        num_workers = max(1, min(cpu_count() - 1, 4))
    
    print(f"Using {num_workers} workers for parallel processing")
    
    # Process files in batches to prevent memory issues
    results = []
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    # Process in batches
    total_files = len(files)
    num_batches = (total_files + batch_size - 1) // batch_size
    
    with tqdm(total=total_files, desc="Processing files") as pbar:
        for batch_idx in range(num_batches):
            # Clear CUDA cache at the start of each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Get batch of files
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_files)
            batch_files = files[start_idx:end_idx]
            
            # Prepare arguments for batch
            args = [(file_path, output_dir, label_mapping) for file_path in batch_files]
            
            # Process batch in parallel
            with Pool(processes=num_workers) as pool:
                batch_results = list(pool.imap_unordered(process_file, args))
                
                # Update counts and results
                for result in batch_results:
                    results.append(result)
                    
                    if result["status"] == "success":
                        success_count += 1
                    elif result["status"] == "failed":
                        failed_count += 1
                    elif result["status"] == "skipped":
                        skipped_count += 1
                    
                    pbar.set_postfix(success=success_count, failed=failed_count, skipped=skipped_count)
                
                # Update progress bar
                pbar.update(len(batch_files))
            
            # Force garbage collection after each batch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Save intermediate results periodically
            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                # Create intermediate report
                intermediate_report = {
                    "summary": {
                        "success": success_count,
                        "failed": failed_count,
                        "skipped": skipped_count,
                        "total": total_files,
                        "completed": success_count + failed_count + skipped_count,
                        "time_elapsed": time.time() - start_time
                    },
                    "results": results
                }
                
                # Save intermediate report
                with open(os.path.join(output_dir, f"preprocessing_report_batch_{batch_idx+1}.json"), 'w') as f:
                    json.dump(intermediate_report, f, indent=2)
    
    # Create label mapping and statistics
    create_label_mapping(output_dir, results)
    
    elapsed_time = time.time() - start_time
    
    # Create report
    report = {
        "success": success_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "total": len(files),
        "time": elapsed_time
    }
    
    # Save report
    with open(os.path.join(output_dir, "preprocessing_report.json"), 'w') as f:
        json.dump({
            "summary": report,
            "results": results
        }, f, indent=2)
    
    print(f"\nPreprocessing Summary:")
    print(f"Processed {len(files)} files with {success_count} successful, {failed_count} failed, and {skipped_count} skipped")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Report saved to {os.path.join(output_dir, 'preprocessing_report.json')}")
    
    return report

def create_label_mapping(output_dir, results):
    """
    Create label mapping and statistics from processed data
    
    Args:
        output_dir: Directory where processed files are saved
        results: List of processing results
    """
    # Create label mapping
    label_mapping = {}
    label_counts = {}
    
    for result in results:
        if result["status"] == "success":
            label = result["label"]
            file_base = os.path.splitext(os.path.basename(result["file"]))[0]
            
            if label in label_mapping:
                label_mapping[label].append(file_base)
                label_counts[label] += 1
            else:
                label_mapping[label] = [file_base]
                label_counts[label] = 1
    
    # Save label mapping
    mapping_file = os.path.join(output_dir, "label_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    # Save label statistics
    stats_file = os.path.join(output_dir, "label_statistics.json")
    with open(stats_file, 'w') as f:
        json.dump({
            "total_samples": sum(label_counts.values()),
            "unique_labels": len(label_counts),
            "label_counts": label_counts
        }, f, indent=2)
    
    print(f"Created label mapping with {sum(label_counts.values())} samples and {len(label_counts)} unique labels")
    print(f"Label mapping saved to {mapping_file}")
    print(f"Label statistics saved to {stats_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a large dataset for ASL translation")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed files")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of processes to use")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--mapping_file", type=str, default=None, help="Path to save/load label mapping")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of files to process in each batch")
    
    args = parser.parse_args()
    
    preprocess_large_dataset(
        args.input_dir,
        args.output_dir,
        args.num_workers,
        args.limit,
        args.mapping_file,
        args.batch_size
    ) 