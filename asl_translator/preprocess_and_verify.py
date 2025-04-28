"""
Preprocess and verify pipeline.
This script runs data preprocessing and verification in a single pipeline.
"""

import os
# Set OpenMP environment variable to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import shutil
import time
from datetime import datetime
import numpy as np
from preprocessing.data_processor import preprocess_dataset, extract_features
from utils.visualization import plot_feature_comparison, plot_smplx_params
from utils.pickle_loader import safe_load_pickle, extract_smplx_params as extract_params
from verify_preprocessing import verify_features, visualize_comparison

def create_verification_sample(raw_data_file, output_dir):
    """
    Create a sample for verification purposes.
    
    Args:
        raw_data_file (str): Path to raw data file
        output_dir (str): Directory to save verification sample
        
    Returns:
        tuple: (raw_features, processed_features, features_path)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract base name
    base_name = os.path.splitext(os.path.basename(raw_data_file))[0]
    
    # Load raw data
    try:
        # Try to load as pickle first
        if raw_data_file.endswith('.pkl'):
            raw_data = safe_load_pickle(raw_data_file)
            raw_features = extract_params(raw_data)
        else:
            # Load as numpy array
            raw_features = np.load(raw_data_file)
        
        # Extract features
        processed_features = extract_features(raw_features)
        
        # Save features
        features_path = os.path.join(output_dir, f"{base_name}_features.npy")
        np.save(features_path, processed_features)
        
        # Create a visualization
        plot_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        plot_feature_comparison(raw_features, processed_features, 
                              title=f"Feature Comparison - {base_name}",
                              output_file=plot_path)
        
        # If raw data is a dictionary, visualize params
        if isinstance(raw_features, dict):
            params_plot_path = os.path.join(output_dir, f"{base_name}_params.png")
            plot_smplx_params(raw_features, output_file=params_plot_path)
            
        return raw_features, processed_features, features_path
    except Exception as e:
        print(f"Error creating verification sample: {e}")
        return None, None, None

def run_preprocessing(raw_dir, processed_dir, verification_dir, num_samples=5):
    """
    Run preprocessing and verification.
    
    Args:
        raw_dir (str): Directory containing raw data
        processed_dir (str): Directory to save processed data
        verification_dir (str): Directory to save verification results
        num_samples (int): Number of samples for verification
        
    Returns:
        dict: Report of preprocessing results
    """
    start_time = time.time()
    
    # Create directories
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(verification_dir, exist_ok=True)
    
    # Find raw data files
    raw_files = []
    for ext in ['.pkl', '.npy']:
        raw_files.extend([os.path.join(raw_dir, f) for f in os.listdir(raw_dir) 
                        if f.endswith(ext) and not f.endswith('_metadata.npy')])
    
    if not raw_files:
        print(f"No data files found in {raw_dir}")
        return {"status": "failed", "reason": "No data files found"}
    
    print(f"Found {len(raw_files)} raw data files")
    
    # Preprocess data
    print(f"Preprocessing data from {raw_dir} to {processed_dir}")
    try:
        preprocess_dataset(raw_dir, processed_dir)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return {"status": "failed", "reason": str(e)}
    
    # Verify a sample of the preprocessed data
    verification_samples = min(num_samples, len(raw_files))
    verification_results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "verification_dir": verification_dir,
        "samples_checked": verification_samples,
        "results": []
    }
    
    print(f"\nVerifying {verification_samples} samples")
    for i, raw_file in enumerate(raw_files[:verification_samples]):
        file_id = os.path.splitext(os.path.basename(raw_file))[0]
        print(f"Verifying sample {i+1}/{verification_samples}: {file_id}")
        
        sample_verify_dir = os.path.join(verification_dir, file_id)
        os.makedirs(sample_verify_dir, exist_ok=True)
        
        # Process this sample specifically for verification
        _, _, features_path = create_verification_sample(raw_file, sample_verify_dir)
        
        if features_path is None:
            verification_results["results"].append({
                "sample": file_id,
                "status": "failed",
                "reason": "Error creating verification sample"
            })
            continue
        
        # Find the corresponding processed file from the main preprocessing
        processed_file = os.path.join(processed_dir, f"{file_id}.npy")
        if not os.path.exists(processed_file):
            verification_results["results"].append({
                "sample": file_id,
                "status": "failed",
                "reason": "Processed file not found"
            })
            continue
        
        # Verify the features
        success = verify_features(raw_file, processed_file)
        
        if success:
            verification_results["results"].append({
                "sample": file_id,
                "status": "passed"
            })
            
            # Create detailed visualization
            visualize_comparison(raw_file, processed_file, sample_verify_dir)
        else:
            verification_results["results"].append({
                "sample": file_id,
                "status": "failed",
                "reason": "Feature mismatch"
            })
    
    # Calculate summary
    passed = sum(1 for r in verification_results["results"] if r["status"] == "passed")
    total = len(verification_results["results"])
    
    verification_results["summary"] = {
        "passed": passed,
        "failed": total - passed,
        "total": total,
        "pass_rate": f"{(passed / total * 100):.2f}%" if total > 0 else "0%"
    }
    
    # Calculate processing time
    verification_results["processing_time"] = f"{time.time() - start_time:.2f} seconds"
    
    # Save verification report
    report_path = os.path.join(verification_dir, "verification_report.json")
    with open(report_path, 'w') as f:
        json.dump(verification_results, f, indent=2)
    
    print("\nPreprocessing and Verification Summary:")
    print(f"Processed {len(raw_files)} files to {processed_dir}")
    print(f"Verified {verification_samples} samples with {passed} passed and {total - passed} failed")
    print(f"Pass rate: {verification_results['summary']['pass_rate']}")
    print(f"Processing time: {verification_results['processing_time']}")
    print(f"Verification report saved to {report_path}")
    
    return verification_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and verify ASL data")
    parser.add_argument("--raw_dir", type=str, required=True, help="Directory containing raw data files")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Directory to save processed data")
    parser.add_argument("--verification_dir", type=str, default="verification_results", help="Directory to save verification results")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples for verification")
    
    args = parser.parse_args()
    
    run_preprocessing(args.raw_dir, args.processed_dir, args.verification_dir, args.num_samples) 