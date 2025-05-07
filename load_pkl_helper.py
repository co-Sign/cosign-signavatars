import torch
import pickle
import numpy as np
import os
import re
import io

class CUDAUnpickler(pickle.Unpickler):
    """
    Custom unpickler that redirects CUDA tensors to CPU.
    """
    def find_class(self, module, name):
        if module == 'torch.cuda' and name == '_rebuild_tensor_v2':
            return torch.Tensor._make_subclass
        return super().find_class(module, name)
    
    def persistent_load(self, persistent_id):
        try:
            if isinstance(persistent_id, tuple):
                type_id, storage_type, data_id, location, size, dtype, stride = persistent_id
                if 'cuda' in location:
                    location = location.replace('cuda', 'cpu')
                return super().persistent_load((type_id, storage_type, data_id, location, size, dtype, stride))
            return super().persistent_load(persistent_id)
        except Exception as e:
            print(f"Error in persistent_load: {e}")
            if isinstance(persistent_id, tuple) and len(persistent_id) >= 4:
                # Try to manually create a storage
                try:
                    storage_type = getattr(torch, persistent_id[1])
                    storage = storage_type(persistent_id[4])
                    return storage
                except Exception as e2:
                    print(f"Failed to create storage: {e2}")
            return None

def load_pkl_file(file_path):
    """
    Loads a PKL file with multiple methods to handle different serialization formats.
    Particularly focuses on handling CUDA-serialized files on CPU-only machines.
    
    Args:
        file_path (str): Path to the PKL file
        
    Returns:
        data: The loaded data or None if all methods fail
    """
    print(f"Attempting to load: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return None
    
    # Method 1: Custom CUDA unpickler
    try:
        with open(file_path, 'rb') as f:
            unpickler = CUDAUnpickler(f)
            data = unpickler.load()
        print("Successfully loaded using custom CUDA unpickler")
        return data
    except Exception as e:
        print(f"Method 1 failed: {str(e)}")
    
    # Method 2: Regular torch load with CPU mapping and weights_only=False
    try:
        data = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
        print("Successfully loaded using torch.load with CPU mapping and weights_only=False")
        return data
    except Exception as e:
        print(f"Method 2 failed: {str(e)}")
    
    # Method 3: Binary modification approach - more aggressive replacement
    try:
        with open(file_path, 'rb') as f:
            pickle_data = f.read()
        
        # Replace CUDA references with CPU in the binary data
        modified_data = pickle_data.replace(b'torch.cuda', b'torch.cpu')
        modified_data = re.sub(b'cuda:[0-9]+', b'cpu', modified_data)
        
        # Try to load with CPU mapping and weights_only=False
        buffer = io.BytesIO(modified_data)
        data = torch.load(buffer, map_location='cpu', weights_only=False)
        print("Successfully loaded using binary replacement method")
        return data
    except Exception as e:
        print(f"Method 3 failed: {str(e)}")
    
    # Method 4: Regular pickle load
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print("Successfully loaded using pickle.load")
        return data
    except Exception as e:
        print(f"Method 4 failed: {str(e)}")
        
    # Method 5: Load pickle with encoding='latin1'
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        print("Successfully loaded using pickle.load with latin1 encoding")
        return data
    except Exception as e:
        print(f"Method 5 failed: {str(e)}")
    
    print("All methods failed")
    return None

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
                print(f"  - {key}: {type(value)} with shape {value.shape}")
            else:
                print(f"  - {key}: {type(value)}")
    elif isinstance(data, (list, tuple)):
        print(f"Sequence with {len(data)} items")
        if len(data) > 0:
            print(f"First item type: {type(data[0])}")
    elif isinstance(data, (torch.Tensor, np.ndarray)):
        print(f"Shape: {data.shape}")
    
    print("")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_pkl_helper.py <path_to_pkl_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    data = load_pkl_file(file_path)
    visualize_data(data) 