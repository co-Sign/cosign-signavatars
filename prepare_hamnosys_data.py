#!/usr/bin/env python3
"""
Script to prepare HamNoSys data for training
Implements batch processing and data partitioning
"""
import os
import json
import numpy as np
import argparse
from pathlib import Path
from collections import Counter
import random
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare HamNoSys data for training')
    parser.add_argument('--data-file', type=str, default='datasets/hamnosys2motion/data.json',
                      help='Path to the data.json file')
    parser.add_argument('--output-dir', type=str, default='data/hamnosys_data',
                      help='Directory to save prepared data')
    parser.add_argument('--min-length', type=int, default=5,
                      help='Minimum HamNoSys sequence length to include')
    parser.add_argument('--min-samples', type=int, default=6,
                      help='Minimum number of samples per class')
    parser.add_argument('--max-classes', type=int, default=25,
                      help='Maximum number of classes to include')
    parser.add_argument('--offset', type=int, default=0,
                      help='Offset for class selection (for batch processing)')
    parser.add_argument('--batch-size', type=int, default=100,
                      help='Batch size for processing data')
    parser.add_argument('--num-workers', type=int, default=None,
                      help='Number of workers for parallel processing (default: CPU count)')
    parser.add_argument('--include-wlasl', action='store_true',
                      help='Include WLASL dataset for additional features')
    parser.add_argument('--wlasl-dir', type=str, default='datasets/wlasl_pkls_cropFalse_defult_shape',
                      help='Directory containing WLASL pickle files')
    parser.add_argument('--include-phonex', action='store_true',
                      help='Include Phoenix dataset features')
    parser.add_argument('--phonex-dir', type=str, default='datasets/phonex_pkls_cropFalse_shapeFalse',
                      help='Directory containing Phoenix pickle files')
    parser.add_argument('--include-how2sign', action='store_true',
                      help='Include How2Sign dataset features')
    parser.add_argument('--how2sign-dir', type=str, default='datasets/how2sign_pkls_cropTrue_shapeTrue',
                      help='Directory containing How2Sign pickle files')
    parser.add_argument('--augment', action='store_true',
                      help='Apply data augmentation to increase samples')
    parser.add_argument('--augment-factor', type=int, default=4,
                      help='Augmentation factor - number of augmented samples per original')
    parser.add_argument('--balance', action='store_true', default=True,
                      help='Balance class distribution by oversampling minority classes')
    return parser.parse_args()

def encode_hamnosys(hamnosys):
    """Convert HamNoSys string to numerical features with improved encoding"""
    # Creating a more robust encoding
    encoded = []
    
    if len(hamnosys) == 0:
        # Return empty encoded array with padding
        return np.zeros((50, 4), dtype=np.float32)
    
    # Create detailed character groups for HamNoSys
    hand_shape_chars = "𝟙𝟚𝟛𝟜𝟝𝟞𝟠𝟡𝟘𝔄𝔅ℭ𝔇𝔈𝔉𝔊ℌℑ𝔍𝔎𝔏𝔐𝔑𝔒𝔓𝔔ℜ𝔖𝔗𝔘𝔙𝔚𝔛𝔜ℨ"
    orientation_chars = "↑↓→←↕↔↗↘↙↖⟲⟳↻↺⥀⥁"
    location_chars = "□■▲▼○●◊♢❍⬤✓✗"
    movement_chars = "↝↜↯↮↭↬↫↪↩↦↥↤↣↢↡↠↟↞↝↜↛↚↙↘↗↖↕↔↓↑→←"
    
    # Track previous character properties to enhance sequential patterns
    prev_char_type = 0
    
    for i, char in enumerate(hamnosys):
        # Calculate normalized position
        position = i / max(1, len(hamnosys) - 1)  # Avoid division by zero
        
        # Get Unicode code point
        code_point = ord(char)
        
        # More detailed character typing with fuzzy matching
        is_hand_shape = 1.0 if char in hand_shape_chars else 0.0
        is_orientation = 1.0 if char in orientation_chars else 0.0
        is_location = 1.0 if char in location_chars else 0.0
        is_movement = 1.0 if char in movement_chars else 0.0
        
        # Derive character type (one value between 0-1)
        if is_hand_shape + is_orientation + is_location + is_movement > 0:
            char_type = (is_hand_shape + is_orientation * 2 + is_location * 3 + is_movement * 4) / 4.0
        else:
            # For uncategorized characters
            char_type = 0.8  # Distinguish from zeros in padding
        
        # Calculate sequence pattern feature (relationship to previous character)
        if i > 0:
            sequence_pattern = abs(char_type - prev_char_type) * 0.5  # Normalized pattern feature
        else:
            sequence_pattern = 0.0  # First character has no previous
        
        prev_char_type = char_type  # Update for next iteration
        
        # Features: [normalized_code_point, position, char_type, sequence_pattern]
        encoded.append([
            code_point / 100000.0,  # Normalize code point
            position,
            char_type,
            sequence_pattern
        ])
    
    # Pad or truncate to fixed length of 50
    if len(encoded) < 50:
        padding = [[0, 0, 0, 0]] * (50 - len(encoded))
        encoded.extend(padding)
    else:
        encoded = encoded[:50]
    
    return np.array(encoded, dtype=np.float32)

def process_hamnosys_batch(batch_data):
    """
    Process a batch of HamNoSys entries
    
    Args:
        batch_data: List of (key, hamnosys, label) tuples
        
    Returns:
        List of (feature, label, key) tuples
    """
    results = []
    for key, hamnosys, label in batch_data:
        # Encode HamNoSys
        feature = encode_hamnosys(hamnosys)
        results.append((feature, label, key))
    
    return results

def categorize_hamnosys(data):
    """
    Group HamNoSys sequences by type_name for classification
    """
    # Extracting types from the data
    types = {}
    for key, entry in data.items():
        if 'type_name' in entry and 'hamnosys' in entry:
            type_name = entry['type_name']
            hamnosys = entry['hamnosys']
            
            # Only include entries with non-empty HamNoSys
            if hamnosys.strip():
                if type_name not in types:
                    types[type_name] = []
                types[type_name].append((key, hamnosys))
    
    return types

def load_wlasl_features(wlasl_dir, max_files=50):
    """
    Load WLASL pickle files to extract additional features
    
    Args:
        wlasl_dir: Directory containing WLASL pickle files
        max_files: Maximum number of files to load
        
    Returns:
        Dictionary mapping file IDs to features
    """
    wlasl_features = {}
    wlasl_files = list(Path(wlasl_dir).glob('*.pkl'))
    
    # Limit the number of files to process
    if max_files and len(wlasl_files) > max_files:
        wlasl_files = wlasl_files[:max_files]
    
    print(f"Loading {len(wlasl_files)} WLASL pickle files...")
    
    for pkl_file in tqdm(wlasl_files, desc="Loading WLASL files"):
        try:
            # Extract file ID from filename
            file_id = pkl_file.stem
            
            # Load pickle file with CPU mapping
            with open(pkl_file, 'rb') as f:
                try:
                    # First try loading with CPU mapping
                    wlasl_data = torch.load(f, map_location=torch.device('cpu'))
                except:
                    # If that fails, try older method
                    f.seek(0)  # Reset file pointer
                    wlasl_data = pickle.load(f, encoding='latin1')
            
            # Map CPU tensor if needed (from error message)
            if hasattr(wlasl_data, 'cpu'):
                wlasl_data = wlasl_data.cpu()
            
            # Store features
            wlasl_features[file_id] = wlasl_data
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
    
    return wlasl_features

def apply_augmentation(features, labels, file_names, augment_factor=4):
    """
    Apply data augmentation to increase the number of samples
    
    Args:
        features: Array of features
        labels: Array of labels
        file_names: List of file names
        augment_factor: Number of augmented samples to generate per original sample
        
    Returns:
        Augmented features, labels, and file names
    """
    print(f"Applying enhanced data augmentation with factor {augment_factor}...")
    
    aug_features = []
    aug_labels = []
    aug_file_names = []
    
    # Add original data
    aug_features.extend(features)
    aug_labels.extend(labels)
    aug_file_names.extend(file_names)
    
    # Create a lookup dictionary for similar samples to improve augmentation mixing
    label_to_samples = {}
    for i, label in enumerate(labels):
        if label not in label_to_samples:
            label_to_samples[label] = []
        label_to_samples[label].append(i)
    
    # Generate augmented samples
    for i in range(len(features)):
        feature = features[i]
        label = labels[i]
        file_name = file_names[i]
        
        # Create multiple augmentations with increasing intensity
        for j in range(augment_factor):
            # Start with the original feature
            aug_feature = feature.copy()
            
            # 1. Add random noise with varying intensity based on position and augmentation index
            noise_scale = 0.02 + (j * 0.01)  # Base noise level increases with each augmentation
            for k in range(len(aug_feature)):
                # Calculate position-dependent noise (U-shaped curve)
                pos = k / max(1, len(aug_feature) - 1)  # 0 to 1
                pos_factor = 1.0 + 0.5 * (2 * (pos - 0.5) ** 2)  # 1.0 at edges, 0.5 in middle
                
                # Apply noise scaled by position and augmentation index
                noise = np.random.normal(0, noise_scale * pos_factor, aug_feature[k].shape)
                aug_feature[k] = aug_feature[k] + noise
            
            # 2. Time-stretching or compressing with wider range of factors
            if j % 3 != 2:  # Apply to 2/3 of augmentations
                stretch_options = [0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2]
                stretch_factor = np.random.choice(stretch_options)
                
                # Only stretch the non-zero part
                non_zero_mask = np.any(feature > 0.01, axis=1)
                orig_len = np.sum(non_zero_mask)
                
                if orig_len > 0:
                    stretched_len = min(50, max(5, int(orig_len * stretch_factor)))
                    non_zero_part = feature[non_zero_mask]
                    
                    # Apply stretch by resampling with linear interpolation for smoother results
                    indices = np.linspace(0, orig_len-1, stretched_len)
                    
                    # Create stretched version with proper interpolation
                    new_feature = np.zeros_like(feature)
                    for dim in range(non_zero_part.shape[1]):
                        # Use numpy's interp function for better interpolation
                        interpolated = np.interp(
                            indices, 
                            np.arange(orig_len), 
                            non_zero_part[:, dim]
                        )
                        new_feature[:stretched_len, dim] = interpolated
                    
                    aug_feature = new_feature
            
            # 3. Drop or modify random frames
            if j % 4 != 3:  # Apply to 3/4 of augmentations
                drop_prob = min(0.15, 0.05 + j * 0.015)  # Progressive drop probability
                non_zero_mask = np.any(aug_feature > 0.01, axis=1)
                non_zero_indices = np.where(non_zero_mask)[0]
                
                if len(non_zero_indices) > 5:  # Ensure we have enough frames
                    num_to_drop = max(1, int(drop_prob * len(non_zero_indices)))
                    drop_indices = np.random.choice(non_zero_indices, size=num_to_drop, replace=False)
                    
                    # Apply random effects to dropped frames
                    for idx in drop_indices:
                        effect = np.random.choice(['zero', 'reduce', 'noise'])
                        if effect == 'zero':
                            aug_feature[idx] = 0
                        elif effect == 'reduce':
                            aug_feature[idx] = aug_feature[idx] * np.random.uniform(0.1, 0.3)
                        else:  # noise
                            aug_feature[idx] = np.random.normal(0, 0.1, aug_feature[idx].shape)
            
            # 4. Mix with other samples of the same class for harder examples
            if j % augment_factor == augment_factor - 1:  # Apply to last augmentation in the series
                similar_samples = label_to_samples[label]
                if len(similar_samples) > 1:
                    # Find multiple samples to mix
                    num_to_mix = min(3, len(similar_samples))
                    mix_indices = np.random.choice(
                        [idx for idx in similar_samples if idx != i], 
                        size=num_to_mix-1, 
                        replace=False
                    )
                    
                    # Progressive mixing with multiple samples
                    mix_feature = aug_feature.copy() * 0.7  # Base contribution
                    remaining_weight = 0.3  # Remaining weight to distribute
                    
                    for mix_idx, other_idx in enumerate(mix_indices):
                        # Distribute remaining weight among samples
                        mix_ratio = remaining_weight / len(mix_indices)
                        other_feature = features[other_idx]
                        mix_feature += other_feature * mix_ratio
                    
                    aug_feature = mix_feature
            
            # 5. Apply channel-specific augmentation
            if j % 2 == 0:  # Apply to half of augmentations
                # Scale character type dimension
                char_type_scale = np.random.uniform(0.75, 1.25)
                aug_feature[:, 2] = aug_feature[:, 2] * char_type_scale
                
                # Shift position values
                position_shift = np.random.uniform(-0.08, 0.08)
                aug_feature[:, 1] = np.clip(aug_feature[:, 1] + position_shift, 0, 1)
                
                # Modify sequence pattern
                if np.random.random() < 0.5:
                    seq_pattern_scale = np.random.uniform(0.8, 1.2)
                    aug_feature[:, 3] = aug_feature[:, 3] * seq_pattern_scale
            
            # Ensure all values are within reasonable bounds
            aug_feature = np.clip(aug_feature, -1.5, 1.5)
            
            # Add augmented sample
            aug_features.append(aug_feature)
            aug_labels.append(label)
            aug_file_names.append(f"{file_name}_aug{j}")
    
    return np.array(aug_features), np.array(aug_labels), aug_file_names

def balance_classes(features, labels, file_names, max_per_class=None):
    """
    Balance class distribution by generating additional samples for minority classes
    
    Args:
        features: Array of features
        labels: Array of labels
        file_names: List of file names
        max_per_class: Maximum number of samples per class (None = use max count)
        
    Returns:
        Balanced features, labels, and file names
    """
    # Count samples per class
    class_counts = Counter(labels)
    unique_classes = np.unique(labels)
    
    if max_per_class is None:
        # Use maximum count if not specified
        max_per_class = max(class_counts.values())
    
    print(f"Balancing classes to {max_per_class} samples per class...")
    
    # Organize samples by class
    class_samples = {label: [] for label in unique_classes}
    for i, label in enumerate(labels):
        class_samples[label].append((features[i], file_names[i]))
    
    # Create balanced dataset
    balanced_features = []
    balanced_labels = []
    balanced_file_names = []
    
    for label, samples in class_samples.items():
        # Add all original samples
        for feature, file_name in samples:
            balanced_features.append(feature)
            balanced_labels.append(label)
            balanced_file_names.append(file_name)
        
        # Oversample minority classes
        current_count = len(samples)
        if current_count < max_per_class:
            # Number of samples to add
            to_add = max_per_class - current_count
            
            # Generate additional samples
            for i in range(to_add):
                # Select a random sample to duplicate and modify
                idx = random.randint(0, len(samples) - 1)
                feature, file_name = samples[idx]
                
                # Apply small random noise
                noise = np.random.normal(0, 0.1, feature.shape)
                new_feature = feature + noise
                
                # Add to balanced dataset
                balanced_features.append(new_feature)
                balanced_labels.append(label)
                balanced_file_names.append(f"{file_name}_bal{i}")
    
    return np.array(balanced_features), np.array(balanced_labels), balanced_file_names

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set number of workers
    num_workers = args.num_workers or max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_workers} workers for parallel processing")
    
    # Load data
    print(f"Loading data from {args.data_file}...")
    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} entries")
    
    # Load WLASL features if specified
    wlasl_features = {}
    if args.include_wlasl and os.path.exists(args.wlasl_dir):
        wlasl_features = load_wlasl_features(args.wlasl_dir)
        print(f"Loaded {len(wlasl_features)} WLASL feature sets")
    
    # Load Phoenix features if specified
    phonex_features = {}
    if args.include_phonex and os.path.exists(args.phonex_dir):
        phonex_features = load_wlasl_features(args.phonex_dir, max_files=100)
        print(f"Loaded {len(phonex_features)} Phoenix feature sets")
    
    # Load How2Sign features if specified
    how2sign_features = {}
    if args.include_how2sign and os.path.exists(args.how2sign_dir):
        how2sign_features = load_wlasl_features(args.how2sign_dir, max_files=100)
        print(f"Loaded {len(how2sign_features)} How2Sign feature sets")
    
    # Categorize data by type_name
    types = categorize_hamnosys(data)
    print(f"Found {len(types)} unique type names")
    
    # Filter and select classes
    valid_types = {}
    for type_name, entries in types.items():
        # Filter by minimum length
        valid_entries = [(key, hamnosys) for key, hamnosys in entries if len(hamnosys) >= args.min_length]
        
        # Only include types with enough samples
        if len(valid_entries) >= args.min_samples:
            valid_types[type_name] = valid_entries
    
    print(f"Found {len(valid_types)} types with at least {args.min_samples} samples of minimum length {args.min_length}")
    
    # Sort by number of examples and select top classes
    type_counts = [(type_name, len(entries)) for type_name, entries in valid_types.items()]
    type_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Apply offset and limit based on batch parameters
    start_idx = args.offset
    end_idx = min(start_idx + args.max_classes, len(type_counts))
    
    if start_idx >= len(type_counts):
        print(f"Warning: Offset {start_idx} exceeds the number of available types {len(type_counts)}")
        start_idx = 0
        end_idx = min(args.max_classes, len(type_counts))
    
    selected_types = type_counts[start_idx:end_idx]
    print(f"Selected {len(selected_types)} types with offset {start_idx} (classes {start_idx} to {end_idx-1})")
    
    # Create label mapping
    label_mapping = {type_name: i for i, (type_name, _) in enumerate(selected_types)}
    
    # Prepare data for batch processing
    all_data = []
    for type_name, _ in selected_types:
        label = label_mapping[type_name]
        entries = valid_types[type_name]
        
        for key, hamnosys in entries:
            all_data.append((key, hamnosys, label))
    
    # Split into batches
    batch_size = args.batch_size
    total_batches = (len(all_data) + batch_size - 1) // batch_size
    batches = [all_data[i*batch_size:(i+1)*batch_size] for i in range(total_batches)]
    
    print(f"Processing {len(all_data)} entries in {total_batches} batches...")
    
    # Process batches in parallel
    all_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all batches for processing
        future_to_batch = {executor.submit(process_hamnosys_batch, batch): i for i, batch in enumerate(batches)}
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Processing batches"):
            batch_idx = future_to_batch[future]
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
    
    # Separate the results
    features = []
    labels = []
    file_names = []
    
    for feature, label, key in all_results:
        features.append(feature)
        labels.append(label)
        file_names.append(key)
    
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    # Apply data augmentation if specified
    if args.augment:
        features, labels, file_names = apply_augmentation(
            features, labels, file_names, 
            augment_factor=args.augment_factor
        )
    
    # Balance classes if specified
    if args.balance:
        features, labels, file_names = balance_classes(features, labels, file_names)
    
    print(f"Final processed data: {len(features)} samples")
    
    # Save dataset
    np.save(os.path.join(args.output_dir, 'features.npy'), features)
    np.save(os.path.join(args.output_dir, 'labels.npy'), labels)
    
    # Save file names and label mapping
    with open(os.path.join(args.output_dir, 'file_names.json'), 'w') as f:
        json.dump(file_names, f)
    
    with open(os.path.join(args.output_dir, 'label_mapping.json'), 'w') as f:
        json.dump({type_name: int(i) for type_name, i in label_mapping.items()}, f)
    
    # Save batch information
    batch_info = {
        'offset': args.offset,
        'max_classes': args.max_classes,
        'selected_classes': len(selected_types),
        'class_range': [start_idx, end_idx - 1]
    }
    
    # Save metadata
    metadata = {
        'num_classes': len(selected_types),
        'num_samples': len(features),
        'feature_dim': features.shape[2],
        'classes': {int(i): type_name for type_name, i in label_mapping.items()},
        'augmentation_applied': args.augment,
        'balance_applied': args.balance,
        'batch_info': batch_info
    }
    
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Calculate class distribution
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    class_distribution = {int(label): int(count) for label, count in zip(unique_labels, label_counts)}
    
    # Print class distribution statistics
    print(f"\nClass distribution:")
    print(f"  - Min examples per class: {min(label_counts)}")
    print(f"  - Max examples per class: {max(label_counts)}")
    print(f"  - Avg examples per class: {np.mean(label_counts):.2f}")
    
    # Save class distribution
    with open(os.path.join(args.output_dir, 'class_distribution.json'), 'w') as f:
        json.dump(class_distribution, f, indent=2)
    
    print(f"\nDataset prepared:")
    print(f"  - {len(features)} samples")
    print(f"  - {len(selected_types)} classes (offset: {args.offset})")
    print(f"  - Features shape: {features.shape}")
    print(f"  - Data saved to {args.output_dir}")

if __name__ == "__main__":
    main() 