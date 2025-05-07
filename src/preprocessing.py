import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import sys
import torch
import traceback
import json
from pathlib import Path

# Import the custom CUDAUnpickler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_pkl_helper import CUDAUnpickler, load_pkl_file as helper_load_pkl_file

class DataLoader:
    def __init__(self, data_dir, seq_length=50, feature_dim=1629, cpu_data_dir='data/cpu_samples'):
        """
        Data loader for ASL recognition from .pkl files
        
        Args:
            data_dir: Directory containing .pkl files with ASL sequences
            seq_length: Number of frames to pad/truncate sequences to
            feature_dim: Dimension of features in each frame
            cpu_data_dir: Directory containing CPU-compatible numpy files
        """
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.cpu_data_dir = cpu_data_dir
        self.scaler = StandardScaler()
        
    def load_pkl_file(self, file_path):
        """Load motion data from a .pkl file.
        
        Args:
            file_path: Path to the .pkl file to load.
            
        Returns:
            The loaded content of the .pkl file, or None if loading fails.
        """
        print(f"Attempting to load: {file_path}")
        
        # Check if we have a CPU-compatible version of this file
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        cpu_file = os.path.join(self.cpu_data_dir, f"{base_name}.npy")
        
        if os.path.exists(cpu_file):
            try:
                print(f"Found CPU-compatible file: {cpu_file}")
                data = np.load(cpu_file, allow_pickle=True)
                
                # Convert to dictionary format if it's just an array
                if isinstance(data, np.ndarray) and not isinstance(data, dict):
                    return {'smplx': data}
                return data
            except Exception as e:
                print(f"Failed to load CPU file: {e}")
        
        # Use the helper function from load_pkl_helper.py
        data = helper_load_pkl_file(file_path)
        
        if data is None:
            print(f"All methods failed to load {file_path}, skipping this file")
        
        return data
    
    def extract_features_from_pkl(self, data):
        """
        Extract relevant motion features from the pkl data
        
        Args:
            data: Dictionary from .pkl file
            
        Returns:
            Flattened feature array for sequence
        """
        # Handle the case where data is already a numpy array
        if isinstance(data, np.ndarray):
            if len(data.shape) == 2:  # Assuming this is already the features
                return data
        
        # Extract SMPLX parameters from the data
        all_parameters = None
        
        if isinstance(data, dict):
            all_parameters = data.get('smplx', None)
        
        if all_parameters is None:
            print("No 'smplx' key found in data")
            return None
            
        # Convert to numpy if it's a torch tensor
        if isinstance(all_parameters, torch.Tensor):
            all_parameters = all_parameters.cpu().numpy()
            
        # Extract the relevant pose parameters
        try:
            root_pose = all_parameters[:, :3]  # (num_frames, 3)
            body_pose = all_parameters[:, 3:66]  # (num_frames, 63)
            left_hand_pose = all_parameters[:, 66:111]  # (num_frames, 45)
            right_hand_pose = all_parameters[:, 111:156]  # (num_frames, 45)
            jaw_pose = all_parameters[:, 156:159]  # (num_frames, 3)
            shape = all_parameters[:, 159:169]  # (num_frames, 10)
            expression = all_parameters[:, 169:179]  # (num_frames, 10)
            cam_trans = all_parameters[:, 179:182]  # (num_frames, 3)
            
            # Concatenate all features
            features = np.concatenate([
                root_pose, body_pose, left_hand_pose, right_hand_pose,
                jaw_pose, shape, expression, cam_trans
            ], axis=1)
            
            return features
        except Exception as e:
            print(f"Error extracting parameters: {e}")
            # If extraction fails, just return the raw data if it's the right shape
            if len(all_parameters.shape) == 2:
                return all_parameters
            return None
    
    def pad_or_truncate_sequence(self, sequence):
        """
        Pad or truncate sequence to fixed length
        
        Args:
            sequence: Numpy array of shape (num_frames, feature_dim)
            
        Returns:
            Padded/truncated sequence of shape (seq_length, feature_dim)
        """
        if sequence.shape[0] > self.seq_length:
            # Truncate
            return sequence[:self.seq_length]
        elif sequence.shape[0] < self.seq_length:
            # Pad with zeros
            padding = np.zeros((self.seq_length - sequence.shape[0], sequence.shape[1]))
            return np.concatenate([sequence, padding], axis=0)
        else:
            return sequence
    
    def load_dataset(self, labels_file=None):
        """
        Load all .pkl files from data_dir and extract features
        
        Args:
            labels_file: Optional JSON file with labels for each sequence
            
        Returns:
            features: List of feature sequences
            labels: List of corresponding labels (if labels_file provided)
            file_names: List of file names for each sequence
        """
        features = []
        file_names = []
        
        # First try to load from CPU samples directory if it exists
        if os.path.exists(self.cpu_data_dir):
            npy_files = [f for f in os.listdir(self.cpu_data_dir) if f.endswith('.npy') and f != 'conversion_map.npy']
            
            if npy_files:
                print(f"Found {len(npy_files)} CPU-compatible .npy files, loading these instead")
                
                for npy_file in npy_files:
                    file_path = os.path.join(self.cpu_data_dir, npy_file)
                    try:
                        data = np.load(file_path, allow_pickle=True)
                        
                        # If data is already in the right format, use it directly
                        if isinstance(data, np.ndarray) and len(data.shape) == 2:
                            feature_seq = data
                        else:
                            # Otherwise try to extract features
                            feature_seq = self.extract_features_from_pkl({'smplx': data})
                        
                        if feature_seq is not None:
                            # Pad or truncate sequence
                            feature_seq = self.pad_or_truncate_sequence(feature_seq)
                            features.append(feature_seq)
                            file_names.append(os.path.splitext(npy_file)[0] + '.pkl')  # Convert back to .pkl for labels
                    except Exception as e:
                        print(f"Error loading {npy_file}: {e}")
        
        # If no CPU samples found, or we want to load additional files
        if not features:
            print("Loading from original .pkl files...")
            # Get all .pkl files in directory
            pkl_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pkl')]
            
            for pkl_file in pkl_files:
                file_path = os.path.join(self.data_dir, pkl_file)
                data = self.load_pkl_file(file_path)
                
                if data is not None:
                    feature_seq = self.extract_features_from_pkl(data)
                    
                    if feature_seq is not None:
                        # Pad or truncate sequence
                        feature_seq = self.pad_or_truncate_sequence(feature_seq)
                        features.append(feature_seq)
                        file_names.append(pkl_file)
        
        # Convert to numpy arrays
        features = np.array(features)
        
        # Extract labels if labels_dict is provided
        labels = None
        if labels_file:
            with open(labels_file, 'r') as f:
                import json
                labels_dict = json.load(f)
            
            labels = []
            for file_name in file_names:
                # Extract identifier from file name to match with labels_dict
                identifier = os.path.splitext(file_name)[0]
                label = labels_dict.get(identifier, None)
                labels.append(label)
            
            # Convert labels to categorical
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            labels = le.fit_transform(labels)
            self.label_encoder = le
        
        return features, labels, file_names
    
    def normalize_features(self, features, fit=True):
        """
        Normalize features across the dataset
        
        Args:
            features: Numpy array of shape (num_samples, seq_length, feature_dim)
            fit: Whether to fit the scaler on this data
            
        Returns:
            Normalized features
        """
        # Reshape to 2D for normalization
        orig_shape = features.shape
        features_2d = features.reshape(-1, features.shape[-1])
        
        if fit:
            # Fit and transform
            features_2d = self.scaler.fit_transform(features_2d)
        else:
            # Transform only
            features_2d = self.scaler.transform(features_2d)
        
        # Reshape back to original shape
        features = features_2d.reshape(orig_shape)
        
        return features
    
    def split_data(self, features, labels, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into train, validation, and test sets
        
        Args:
            features: Numpy array of features
            labels: Numpy array of labels
            test_size: Fraction of data for test set
            val_size: Fraction of train data for validation set
            random_state: Random seed for reproducibility
            
        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        # Store the number of unique classes
        self.num_classes = len(np.unique(labels))
        print(f"Total number of unique classes in dataset: {self.num_classes}")
        
        try:
            # First split into train+val and test with stratification
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                features, labels, test_size=test_size, random_state=random_state, stratify=labels
            )
            
            # Then split train+val into train and val with stratification
            val_ratio = val_size / (1 - test_size)  # Adjust val_size for the remaining data
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_ratio, 
                random_state=random_state, stratify=y_train_val
            )
        except ValueError as e:
            # If stratification fails due to too few samples per class, split without stratification
            print(f"Stratification failed: {e}")
            print("Falling back to random splitting without stratification")
            
            # Split without stratification
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                features, labels, test_size=test_size, random_state=random_state
            )
            
            # Then split train+val into train and val
            val_ratio = val_size / (1 - test_size)  # Adjust val_size for the remaining data
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_ratio, 
                random_state=random_state
            )
        
        # Report class distribution in each split
        print(f"Train set class distribution: {np.bincount(y_train)}")
        print(f"Validation set class distribution: {np.bincount(y_val)}")
        print(f"Test set class distribution: {np.bincount(y_test)}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def create_tf_dataset(self, features, labels=None, batch_size=32, shuffle=True, drop_remainder=False):
        """
        Create a TensorFlow dataset
        
        Args:
            features: Numpy array of features
            labels: Numpy array of labels (optional)
            batch_size: Batch size
            shuffle: Whether to shuffle the dataset
            drop_remainder: Whether to drop the last batch if it's smaller than batch_size
            
        Returns:
            TensorFlow dataset
        """
        if labels is not None:
            # Use the number of classes from the dataset
            if hasattr(self, 'num_classes'):
                num_classes = self.num_classes
            else:
                # Fallback if num_classes is not set
                num_classes = max(10, int(np.max(labels) + 1))
                
            print(f"Converting labels to one-hot with {num_classes} classes")
            
            # Convert labels to one-hot encoding
            labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
            
            # Create dataset with features and labels
            dataset = tf.data.Dataset.from_tensor_slices((features, labels_one_hot))
        else:
            # Create dataset with features only (for inference)
            dataset = tf.data.Dataset.from_tensor_slices(features)
        
        # Shuffle if needed
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(features))
        
        # Batch
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        
        return dataset 