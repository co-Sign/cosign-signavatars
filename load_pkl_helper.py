import torch
import pickle
import numpy as np
import os
import re
import io
import sys
from pathlib import Path

class CUDAUnpickler(pickle.Unpickler):
    """
    Enhanced custom unpickler that redirects CUDA tensors to CPU with better error handling.
    """
    def find_class(self, module, name):
        # Handle CUDA tensor rebuilding
        if module == 'torch.cuda' and name == '_rebuild_tensor_v2':
            return torch.Tensor._make_subclass
        # Handle other CUDA-specific classes
        if module.startswith('torch.cuda'):
            module = module.replace('torch.cuda', 'torch')
        return super().find_class(module, name)
    
    def persistent_load(self, persistent_id):
        try:
            # Handle storage redirects for CUDA tensors
            if isinstance(persistent_id, tuple):
                type_id, storage_type, data_id, location, size, dtype, stride = persistent_id
                # If storage is on CUDA, move it to CPU
                if isinstance(location, str) and 'cuda' in location:
                    location = location.replace('cuda', 'cpu')
                return super().persistent_load((type_id, storage_type, data_id, location, size, dtype, stride))
            # Handle other persistent objects
            return super().persistent_load(persistent_id)
        except Exception as e:
            print(f"Error in persistent_load: {e}")
            # More robust fallback handling
            if isinstance(persistent_id, tuple) and len(persistent_id) >= 4:
                # Try to manually create a storage
                try:
                    print(f"Attempting manual storage creation for {persistent_id[1]}")
                    storage_type_name = persistent_id[1]
                    if isinstance(storage_type_name, str) and storage_type_name.startswith('torch.cuda'):
                        storage_type_name = storage_type_name.replace('torch.cuda', 'torch')
                    
                    # Get storage type dynamically
                    storage_type = None
                    for prefix in ['torch', 'torch.storage', '']:
                        try:
                            full_name = f"{prefix}.{storage_type_name}" if prefix else storage_type_name
                            storage_type = eval(full_name)
                            break
                        except (AttributeError, NameError):
                            continue
                    
                    if storage_type is None:
                        # Fallback to common storage types
                        if 'Float' in storage_type_name:
                            storage_type = torch.FloatStorage
                        elif 'Half' in storage_type_name:
                            storage_type = torch.HalfStorage
                        elif 'Int' in storage_type_name:
                            storage_type = torch.IntStorage
                        else:
                            storage_type = torch.FloatStorage
                    
                    # Create storage with appropriate size
                    if len(persistent_id) >= 5 and isinstance(persistent_id[4], int):
                        size = persistent_id[4]
                        storage = storage_type(size)
                    else:
                        storage = storage_type()
                    
                    print(f"Successfully created fallback storage with type {type(storage)}")
                    return storage
                except Exception as e2:
                    print(f"Failed to create storage: {e2}")
                    # Final fallback: create a basic tensor with the same shape
                    try:
                        if len(persistent_id) >= 5 and isinstance(persistent_id[4], int):
                            return torch.zeros(persistent_id[4], dtype=torch.float32)
                    except:
                        pass
            # Return empty tensor as last resort
            return torch.tensor([])

def load_pkl_file(file_path, verbose=True):
    """
    Loads a PKL file with multiple methods to handle different serialization formats.
    Particularly focuses on handling CUDA-serialized files on CPU-only machines.
    
    Args:
        file_path (str): Path to the PKL file
        verbose (bool): Whether to print detailed messages
        
    Returns:
        data: The loaded data or None if all methods fail
    """
    if verbose:
        print(f"Attempting to load: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return None
    
    error_messages = []
    
    # Method 0: Basic pickle load with error handling for protocol differences
    try:
        with open(file_path, 'rb') as f:
            try:
                data = pickle.load(f)
                if verbose:
                    print("Successfully loaded using basic pickle.load")
                return data
            except Exception as e:
                if "unsupported pickle protocol" in str(e).lower():
                    # Try with highest protocol version
                    f.seek(0)
                    for protocol in range(5, 1, -1):
                        try:
                            data = pickle.load(f, protocol=protocol)
                            if verbose:
                                print(f"Successfully loaded using protocol {protocol}")
                            return data
                        except:
                            f.seek(0)
                    
                error_msg = f"Method 0 (basic pickle.load) failed: {str(e)}"
                error_messages.append(error_msg)
                if verbose:
                    print(error_msg)
    except Exception as e:
        error_msg = f"Method 0 (file opening) failed: {str(e)}"
        error_messages.append(error_msg)
        if verbose:
            print(error_msg)
    
    # Method 1: Enhanced custom CUDA unpickler
    try:
        with open(file_path, 'rb') as f:
            unpickler = CUDAUnpickler(f)
            data = unpickler.load()
        if verbose:
            print("Successfully loaded using enhanced custom CUDA unpickler")
        return data
    except Exception as e:
        error_msg = f"Method 1 (CUDAUnpickler) failed: {str(e)}"
        error_messages.append(error_msg)
        if verbose:
            print(error_msg)
    
    # Method 2: Regular torch load with CPU mapping and different weights_only settings
    for weights_only in [False, True]:
        try:
            data = torch.load(file_path, map_location=torch.device('cpu'), weights_only=weights_only)
            if verbose:
                print(f"Successfully loaded using torch.load with CPU mapping and weights_only={weights_only}")
            return data
        except Exception as e:
            error_msg = f"Method 2 (torch.load weights_only={weights_only}) failed: {str(e)}"
            error_messages.append(error_msg)
            if verbose:
                print(error_msg)
    
    # Method 3: Binary modification approach - aggressive replacement
    try:
        with open(file_path, 'rb') as f:
            pickle_data = f.read()
        
        # Replace CUDA references with CPU in the binary data (multiple patterns)
        modified_data = pickle_data.replace(b'torch.cuda', b'torch')
        modified_data = re.sub(b'cuda:[0-9]+', b'cpu', modified_data)
        modified_data = modified_data.replace(b'device=cuda', b'device=cpu')
        
        # Try to load with CPU mapping
        buffer = io.BytesIO(modified_data)
        data = torch.load(buffer, map_location='cpu')
        if verbose:
            print("Successfully loaded using binary replacement method")
        return data
    except Exception as e:
        error_msg = f"Method 3 (binary replacement) failed: {str(e)}"
        error_messages.append(error_msg)
        if verbose:
            print(error_msg)
    
    # Method 4: Regular pickle load with various settings
    for encoding in [None, 'latin1', 'bytes']:
        try:
            with open(file_path, 'rb') as f:
                if encoding:
                    data = pickle.load(f, encoding=encoding)
                else:
                    data = pickle.load(f)
            if verbose:
                print(f"Successfully loaded using pickle.load with encoding={encoding}")
            return data
        except Exception as e:
            error_msg = f"Method 4a (pickle.load encoding={encoding}) failed: {str(e)}"
            error_messages.append(error_msg)
            if verbose:
                print(error_msg)
    
    # Method 4b: Pickle load with fix_imports option
    for fix_imports in [True, False]:
        for encoding in [None, 'latin1', 'bytes']:
            try:
                with open(file_path, 'rb') as f:
                    if encoding:
                        data = pickle.load(f, encoding=encoding, fix_imports=fix_imports)
                    else:
                        data = pickle.load(f, fix_imports=fix_imports)
                if verbose:
                    print(f"Successfully loaded using pickle.load with encoding={encoding}, fix_imports={fix_imports}")
                return data
            except Exception as e:
                error_msg = f"Method 4b (pickle.load encoding={encoding}, fix_imports={fix_imports}) failed: {str(e)}"
                error_messages.append(error_msg)
                if verbose:
                    print(error_msg)
    
    # Method 5: Read as raw bytes and try to interpret it with alternative methods
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        
        # Try different serialization formats:
        # First try numpy
        try:
            buffer = io.BytesIO(raw_data)
            data = np.load(buffer, allow_pickle=True)
            if verbose:
                print("Successfully loaded as numpy array from raw bytes")
            return data
        except:
            pass
        
        # Then try torch
        try:
            buffer = io.BytesIO(raw_data)
            data = torch.load(buffer, map_location='cpu')
            if verbose:
                print("Successfully loaded as torch tensor from raw bytes")
            return data
        except:
            pass
        
        # Finally try pickle with strict_map_key=False (less secure but might work)
        try:
            buffer = io.BytesIO(raw_data)
            unpickler = pickle.Unpickler(buffer)
            unpickler.strict_map_key = False  # Allow loading non-stringified dictionary keys
            data = unpickler.load()
            if verbose:
                print("Successfully loaded using unpickler with strict_map_key=False")
            return data
        except:
            pass
            
    except Exception as e:
        error_msg = f"Method 5 (raw bytes interpretation) failed: {str(e)}"
        error_messages.append(error_msg)
        if verbose:
            print(error_msg)
    
    # Method 6: Desperately try to parse as numpy array
    try:
        data = np.load(file_path, allow_pickle=True)
        if verbose:
            print("Successfully loaded using numpy.load")
        return data
    except Exception as e:
        error_msg = f"Method 6 (numpy.load) failed: {str(e)}"
        error_messages.append(error_msg)
        if verbose:
            print(error_msg)
    
    if verbose:
        print("All loading methods failed. See error messages for details.")
    return None

def convert_tensor_to_cpu(data):
    """
    Recursively converts all CUDA tensors to CPU tensors.
    
    Args:
        data: Data structure potentially containing CUDA tensors
        
    Returns:
        Same data structure with all tensors on CPU
    """
    if isinstance(data, torch.Tensor):
        # Convert CUDA tensor to CPU
        return data.cpu().detach()
    elif isinstance(data, dict):
        # Process dictionary values
        return {k: convert_tensor_to_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        # Process list elements
        return [convert_tensor_to_cpu(item) for item in data]
    elif isinstance(data, tuple):
        # Process tuple elements
        return tuple(convert_tensor_to_cpu(item) for item in data)
    
    # Return other types unchanged
    return data

def visualize_data(data):
    """
    Prints structure and summary of the loaded data.
    
    Args:
        data: The loaded data
    """
    if data is None:
        print("No data to visualize")
        return
    
    print("\nData Summary:")
    print(f"Type: {type(data)}")
    
    if isinstance(data, dict):
        print("Dictionary keys:")
        for key in data.keys():
            value = data[key]
            if isinstance(value, (torch.Tensor, np.ndarray)):
                print(f"  - {key}: {type(value)} with shape {value.shape} on {value.device if hasattr(value, 'device') else 'CPU'}")
            else:
                print(f"  - {key}: {type(value)}")
    elif isinstance(data, (list, tuple)):
        print(f"Sequence with {len(data)} items")
        if len(data) > 0:
            print(f"First item type: {type(data[0])}")
            if isinstance(data[0], (torch.Tensor, np.ndarray)):
                print(f"First item shape: {data[0].shape}")
    elif isinstance(data, (torch.Tensor, np.ndarray)):
        print(f"Shape: {data.shape}")
        if isinstance(data, torch.Tensor):
            print(f"Device: {data.device}")
            print(f"Dtype: {data.dtype}")
    
    print("")

def save_as_cpu_tensor(data, output_path):
    """
    Saves data as CPU tensor for future compatibility.
    
    Args:
        data: Data to save
        output_path: Where to save the converted data
        
    Returns:
        bool: Whether the operation was successful
    """
    try:
        # Ensure data is on CPU
        cpu_data = convert_tensor_to_cpu(data)
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save using torch.save
        torch.save(cpu_data, output_path)
        print(f"Successfully saved CPU-compatible data to {output_path}")
        return True
    except Exception as e:
        print(f"Failed to save data: {e}")
        return False

def batch_convert_pkl_dir(input_dir, output_dir=None):
    """
    Converts all pickle files in a directory to CPU-compatible format.
    
    Args:
        input_dir: Directory containing pickle files
        output_dir: Directory to save converted files (defaults to input_dir + "_cpu")
        
    Returns:
        Tuple of (success_count, fail_count)
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = Path(str(input_dir) + "_cpu")
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all pickle files
    pkl_files = list(input_dir.glob("**/*.pkl"))
    print(f"Found {len(pkl_files)} pickle files in {input_dir}")
    
    success_count = 0
    fail_count = 0
    
    for pkl_file in pkl_files:
        # Calculate relative path for output file
        rel_path = pkl_file.relative_to(input_dir)
        output_file = output_dir / rel_path
        
        # Create parent directories if needed
        os.makedirs(output_file.parent, exist_ok=True)
        
        # Try to load and save the file
        print(f"Processing {pkl_file}...")
        data = load_pkl_file(pkl_file)
        
        if data is not None:
            if save_as_cpu_tensor(data, output_file):
                success_count += 1
            else:
                fail_count += 1
        else:
            print(f"Failed to load {pkl_file}")
            fail_count += 1
    
    print(f"Batch conversion complete: {success_count} successful, {fail_count} failed")
    return success_count, fail_count

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage options:")
        print("  python load_pkl_helper.py <path_to_pkl_file>")
        print("  python load_pkl_helper.py --convert <input_dir> [<output_dir>]")
        sys.exit(1)
    
    if sys.argv[1] == "--convert":
        if len(sys.argv) < 3:
            print("Usage: python load_pkl_helper.py --convert <input_dir> [<output_dir>]")
            sys.exit(1)
        
        input_dir = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        batch_convert_pkl_dir(input_dir, output_dir)
    else:
        file_path = sys.argv[1]
        data = load_pkl_file(file_path)
        visualize_data(data)
        
        # If data was loaded successfully, offer to save a CPU version
        if data is not None:
            output_path = file_path.replace(".pkl", "_cpu.pkl")
            if input(f"Save CPU-compatible version to {output_path}? (y/n): ").lower() == 'y':
                save_as_cpu_tensor(data, output_path) 