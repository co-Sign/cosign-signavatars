#!/usr/bin/env python3
"""
Script to convert PyTorch pickle files using direct tensor methods
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
from tqdm import tqdm
import io
from datetime import datetime

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Convert PyTorch tensor storage files")
    parser.add_argument("--data-dir", required=True, help="Directory containing .pkl files")
    parser.add_argument("--output-dir", default=None, help="Output directory for converted files (default: {data_dir}_cpu)")
    parser.add_argument("--num-workers", type=int, default=None, 
                        help="Number of worker processes (default: CPU count)")
    parser.add_argument("--max-files", type=int, default=None, 
                        help="Maximum number of files to process (for testing)")
    parser.add_argument("--verify", action="store_true", 
                        help="Verify converted files by loading them")
    parser.add_argument("--skip-existing", action="store_true", 
                        help="Skip files that already exist in the output directory")
    
    return parser.parse_args()

def direct_torch_load(file_path, verbose=True):
    """
    Load a PyTorch pickle file using multiple approaches
    
    Args:
        file_path: Path to the pickle file
        verbose: Whether to print progress messages
        
    Returns:
        data: The loaded data or None if all methods fail
    """
    if verbose:
        print(f"Attempting to load: {file_path}")
    
    # Method 1: Try direct load with different map_location settings and options
    methods = [
        {"name": "torch.load CPU map", "fn": lambda: torch.load(file_path, map_location='cpu')},
        {"name": "torch.load CPU device", "fn": lambda: torch.load(file_path, map_location=torch.device('cpu'))},
        {"name": "torch.load weights_only", "fn": lambda: torch.load(file_path, map_location='cpu', weights_only=True)},
        {"name": "torch.jit.load", "fn": lambda: torch.jit.load(file_path, map_location='cpu')},
    ]
    
    for method in methods:
        try:
            data = method["fn"]()
            if verbose:
                print(f"Successfully loaded using {method['name']}")
            return data
        except Exception as e:
            if verbose:
                print(f"Failed with {method['name']}: {str(e)}")
    
    # Method 2: Read raw bytes and try to interpret with torch storage functions
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        
        # Try to deserialize directly using storage
        for storage_type in [torch.FloatStorage, torch.DoubleStorage, torch.HalfStorage, 
                            torch.IntStorage, torch.LongStorage, torch.ByteStorage]:
            try:
                buffer = io.BytesIO(raw_data)
                storage = storage_type._load_from_bytes(raw_data)
                tensor = torch.tensor(storage)
                if verbose:
                    print(f"Successfully loaded using {storage_type.__name__}")
                return tensor
            except Exception as e:
                if verbose:
                    print(f"Failed with {storage_type.__name__}: {str(e)}")
    except Exception as e:
        if verbose:
            print(f"Failed to read raw bytes: {str(e)}")
    
    # Method 3: Try Python pickle with special unpickler
    class CUDAToCPUUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.cuda':
                module = 'torch'
            return super().find_class(module, name)
    
    try:
        with open(file_path, 'rb') as f:
            unpickler = CUDAToCPUUnpickler(f)
            data = unpickler.load()
        if verbose:
            print("Successfully loaded using custom unpickler")
        return data
    except Exception as e:
        if verbose:
            print(f"Failed with custom unpickler: {str(e)}")
    
    if verbose:
        print("All loading methods failed")
    return None

def save_tensor_data(data, output_path, verbose=True):
    """
    Save tensor data in multiple formats for maximum compatibility
    
    Args:
        data: The tensor data to save
        output_path: Path to save the data
        verbose: Whether to print progress messages
        
    Returns:
        success: Whether the save was successful
    """
    # Create parent directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Convert tensor to CPU if needed
    if isinstance(data, torch.Tensor):
        try:
            # Move to CPU and detach from any computation graph
            cpu_data = data.detach().cpu()
            
            # Save as NPY (most compatible)
            npy_path = output_path.replace('.pkl', '.npy')
            np.save(npy_path, cpu_data.numpy())
            
            # Also save as PyTorch PT file for PyTorch users
            pt_path = output_path.replace('.pkl', '.pt')
            torch.save(cpu_data, pt_path)
            
            # And save as pickle with protocol 2 for widest compatibility
            with open(output_path, 'wb') as f:
                pickle.dump(cpu_data.numpy(), f, protocol=2)
            
            if verbose:
                print(f"Successfully saved tensor data to {output_path}, {npy_path}, and {pt_path}")
            return True
        except Exception as e:
            if verbose:
                print(f"Failed to save tensor: {e}")
            return False
    
    # Handle dictionaries with tensors
    elif isinstance(data, dict):
        try:
            cpu_dict = {}
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    cpu_dict[key] = value.detach().cpu().numpy()
                else:
                    cpu_dict[key] = value
            
            # Save as pickle with protocol 2
            with open(output_path, 'wb') as f:
                pickle.dump(cpu_dict, f, protocol=2)
            
            if verbose:
                print(f"Successfully saved dictionary data to {output_path}")
            return True
        except Exception as e:
            if verbose:
                print(f"Failed to save dictionary: {e}")
            return False
    
    # Try for other data types
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(data, f, protocol=2)
        
        if verbose:
            print(f"Successfully saved data to {output_path}")
        return True
    except Exception as e:
        if verbose:
            print(f"Failed to save data: {e}")
        return False

def process_file(args):
    """Process a single file for parallel execution"""
    input_path, output_path, verify = args
    
    # Load pickle file
    data = direct_torch_load(input_path, verbose=False)
    
    if data is None:
        print(f"Failed to load {input_path}")
        return False, input_path
    
    # Save converted data
    success = save_tensor_data(data, output_path, verbose=False)
    
    # Verify if requested
    if success and verify:
        try:
            # Try to load the saved file
            if output_path.endswith('.pkl'):
                with open(output_path, 'rb') as f:
                    verification_data = pickle.load(f)
            elif output_path.endswith('.npy'):
                verification_data = np.load(output_path)
            elif output_path.endswith('.pt'):
                verification_data = torch.load(output_path, map_location='cpu')
            
            if verification_data is None:
                print(f"Verification failed: Loaded None from {output_path}")
                return False, input_path
        except Exception as e:
            print(f"Verification failed for {output_path}: {e}")
            return False, input_path
    
    return success, input_path

def main():
    """Main function"""
    args = parse_arguments()
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = f"{args.data_dir}_cpu_torch"
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all .pkl files in the data directory
    data_dir_path = Path(args.data_dir)
    pkl_files = list(data_dir_path.glob("**/*.pkl"))
    
    # Limit number of files if requested
    if args.max_files is not None:
        pkl_files = pkl_files[:args.max_files]
    
    print(f"Found {len(pkl_files)} .pkl files in {args.data_dir}")
    
    # Set up process pool
    num_workers = args.num_workers or max(1, os.cpu_count() - 1)
    print(f"Processing {len(pkl_files)} files using {num_workers} workers...")
    
    # Create list of (input_path, output_path, verify) tuples
    jobs = []
    for file_path in pkl_files:
        # Get relative path from data directory
        rel_path = file_path.relative_to(data_dir_path)
        
        # Construct output path
        output_path = Path(args.output_dir) / rel_path
        
        # Skip if output file exists and --skip-existing is specified
        if args.skip_existing and output_path.exists():
            continue
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Add job
        jobs.append((str(file_path), str(output_path), args.verify))
    
    # Process files in parallel
    success_count = 0
    failed_files = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs
        future_to_file = {executor.submit(process_file, job): job for job in jobs}
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_file), total=len(jobs), desc="Converting"):
            success, file_path = future.result()
            if success:
                success_count += 1
            else:
                failed_files.append(file_path)
    
    # Print summary
    print("\nConversion complete:")
    print(f"  Total files:  {len(jobs)}")
    print(f"  Successful:   {success_count}")
    print(f"  Failed:       {len(failed_files)}")
    print(f"  Success rate: {success_count / max(1, len(jobs)):.1%}")
    
    # Save stats
    stats = {
        "total_files": len(jobs),
        "successful": success_count,
        "failed": len(failed_files),
        "failed_files": failed_files,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    stats_path = Path(args.output_dir) / "conversion_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"  Stats saved to {stats_path}")
    
    # Print first few failed files
    if failed_files:
        print("\nFailed files:")
        for file in failed_files[:10]:
            print(f"  - {file}")
        
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more (see stats file for complete list)")

if __name__ == "__main__":
    main() 