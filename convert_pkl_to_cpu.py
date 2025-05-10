#!/usr/bin/env python3
"""
Script to convert CUDA pickle files to CPU format for compatibility
"""
import os
import sys
import argparse
import numpy as np
import torch
import pickle
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
from datetime import datetime

# Import helpers from load_pkl_helper.py
from load_pkl_helper import load_pkl_file, convert_tensor_to_cpu, save_as_cpu_tensor

def parse_args():
    parser = argparse.ArgumentParser(description='Convert CUDA pickle files to CPU-compatible format')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Directory containing .pkl files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save processed files (default: data_dir + "_cpu")')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process (None for all)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes (default: number of CPU cores - 1)')
    parser.add_argument('--recursive', action='store_true',
                       help='Recursively search subdirectories for PKL files')
    parser.add_argument('--verify', action='store_true',
                       help='Verify each converted file can be loaded after conversion')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip files that already exist in the output directory')
    return parser.parse_args()

def simple_save(data, output_path):
    """
    Simple and reliable save function that works across platforms
    
    Args:
        data: Data to save
        output_path: Path to save the data
        
    Returns:
        bool: Success status
    """
    try:
        # Convert tensors to CPU and numpy arrays
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        # Recursively convert dictionaries and lists
        if isinstance(data, dict):
            cpu_data = {}
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    cpu_data[k] = v.detach().cpu().numpy()
                else:
                    cpu_data[k] = simple_save(v, None) if v is not None else v
            data = cpu_data
        elif isinstance(data, (list, tuple)):
            data = [simple_save(item, None) if item is not None else None for item in data]
        
        # Create parent directory if needed
        if output_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save using protocol 2 (widely compatible)
            with open(output_path, 'wb') as f:
                pickle.dump(data, f, protocol=2)
            print(f"Successfully saved CPU-compatible data to {output_path}")
            
        return data
    except Exception as e:
        if output_path is not None:
            print(f"Failed to save data: {e}")
        return data if output_path is None else False

def process_file(args):
    """Process a single file for parallel execution"""
    input_path, output_path, verify = args
    
    try:
        # Load PKL file
        data = load_pkl_file(input_path, verbose=False)
        
        if data is None:
            print(f"Failed to load {input_path}")
            return False, input_path
        
        # First try regular CPU conversion
        try:
            success = save_as_cpu_tensor(data, output_path)
            if not success:
                # If failed, try simple save
                success = simple_save(data, output_path)
        except Exception as e:
            print(f"Regular save failed, trying simple_save: {e}")
            success = simple_save(data, output_path)
        
        # Verify if requested
        if success and verify:
            try:
                # Try to load the saved file
                verification_data = None
                
                # Try multiple methods to verify
                try:
                    verification_data = torch.load(output_path, map_location='cpu')
                except:
                    try:
                        with open(output_path, 'rb') as f:
                            verification_data = pickle.load(f)
                    except:
                        try:
                            verification_data = np.load(output_path, allow_pickle=True)
                        except:
                            pass
                
                if verification_data is None:
                    print(f"Verification failed: Could not load from {output_path}")
                    return False, input_path
            except Exception as e:
                print(f"Verification failed for {output_path}: {e}")
                return False, input_path
        
        return success, input_path
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False, input_path

def main():
    args = parse_args()
    
    # Set up input and output directories
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return
    
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(data_dir) + "_cpu"
    output_dir = Path(output_dir)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PKL files
    if args.recursive:
        pkl_files = list(data_dir.glob('**/*.pkl'))
    else:
        pkl_files = list(data_dir.glob('*.pkl'))
    
    # Limit number of files if specified
    if args.max_files and len(pkl_files) > args.max_files:
        pkl_files = pkl_files[:args.max_files]
    
    print(f"Found {len(pkl_files)} .pkl files in {data_dir}")
    
    # Prepare processing jobs
    jobs = []
    for input_path in pkl_files:
        # Calculate relative path for output
        rel_path = input_path.relative_to(data_dir)
        output_path = output_dir / rel_path
        
        # Skip if output exists and skip_existing flag is set
        if args.skip_existing and output_path.exists():
            print(f"Skipping existing file: {output_path}")
            continue
        
        # Create parent directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Add to job list
        jobs.append((str(input_path), str(output_path), args.verify))
    
    # Set up worker pool
    num_workers = args.workers or max(1, multiprocessing.cpu_count() - 1)
    
    # Process files
    success_count = 0
    failed_files = []
    
    if num_workers > 1 and len(jobs) > 1:
        # Parallel processing for multiple files
        print(f"Processing {len(jobs)} files using {num_workers} workers...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_file, job) for job in jobs]
            
            # Process results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
                success, filepath = future.result()
                if success:
                    success_count += 1
                else:
                    failed_files.append(filepath)
    else:
        # Sequential processing for single file or worker
        print(f"Processing {len(jobs)} files sequentially...")
        for job in tqdm(jobs, desc="Converting"):
            success, filepath = process_file(job)
            if success:
                success_count += 1
            else:
                failed_files.append(filepath)
    
    # Save conversion stats
    stats = {
        "total_files": len(pkl_files),
        "successful": success_count,
        "failed": len(failed_files),
        "failed_files": [str(f) for f in failed_files],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    stats_path = output_dir / "conversion_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print(f"\nConversion complete:")
    print(f"  Total files:  {len(pkl_files)}")
    print(f"  Successful:   {success_count}")
    print(f"  Failed:       {len(failed_files)}")
    print(f"  Success rate: {success_count/max(1,len(pkl_files))*100:.1f}%")
    print(f"  Stats saved to {stats_path}")
    
    if failed_files:
        print("\nFailed files:")
        for f in failed_files[:10]:  # Show first 10 failed files
            print(f"  - {f}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more (see stats file for complete list)")

if __name__ == "__main__":
    main() 