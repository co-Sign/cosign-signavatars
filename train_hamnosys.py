#!/usr/bin/env python3
"""
Training script for HamNoSys-based sign language recognition
Uses batch processing for efficient training on the HamNoSys dataset
with advanced techniques for high accuracy
"""
import os
import argparse
import tensorflow as tf
import numpy as np
import json
import time
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

# Import custom modules
from src.model import ASLRecognitionModel

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train HamNoSys recognition model')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='data/hamnosys_data',
                        help='Directory containing prepared HamNoSys data')
    parser.add_argument('--seq-length', type=int, default=50,
                        help='Length of HamNoSys sequences')
    
    # Model parameters
    parser.add_argument('--conv-filters', type=int, nargs='+', default=[64, 128, 256, 512],
                        help='Number of filters in each convolutional layer')
    parser.add_argument('--kernel-sizes', type=int, nargs='+', default=[7, 5, 3, 3],
                        help='Kernel size for each convolutional layer')
    parser.add_argument('--pool-sizes', type=int, nargs='+', default=[2, 2, 2, 2],
                        help='Pool size for each MaxPooling layer')
    parser.add_argument('--lstm-units', type=int, nargs='+', default=[512, 256],
                        help='Number of units in each LSTM layer')
    parser.add_argument('--dropout-rate', type=float, default=0.3,
                        help='Dropout rate for regularization')
    parser.add_argument('--l2-reg', type=float, default=0.0003,
                        help='L2 regularization factor')
    parser.add_argument('--use-attention', action='store_true', default=True,
                        help='Use self-attention mechanism')
    parser.add_argument('--attention-heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--use-residual', action='store_true', default=True,
                        help='Use residual connections')
    parser.add_argument('--use-mixed-architecture', action='store_true', default=True,
                        help='Use mixed GRU/LSTM architecture')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Initial learning rate for optimizer')
    parser.add_argument('--clipnorm', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--val-size', type=float, default=0.15,
                        help='Fraction of training data to use for validation')
    parser.add_argument('--use-class-weights', action='store_true', default=True,
                        help='Use class weights for imbalanced data')
    parser.add_argument('--use-cross-validation', action='store_true', default=False,
                        help='Use cross-validation for better results')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--normalize-features', action='store_true', default=True,
                        help='Apply feature normalization')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='models/hamnosys',
                        help='Directory to save trained model and logs')
    
    # Preprocessing parameters
    parser.add_argument('--prepare-data', action='store_true',
                        help='Run data preparation script before training')
    parser.add_argument('--hamnosys-data-file', type=str, default='datasets/hamnosys2motion/data.json',
                        help='Path to the HamNoSys data JSON file')
    parser.add_argument('--min-length', type=int, default=5,
                        help='Minimum HamNoSys sequence length')
    parser.add_argument('--min-samples', type=int, default=6,
                        help='Minimum samples per class')
    parser.add_argument('--max-classes', type=int, default=10,
                        help='Maximum number of classes')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Apply data augmentation')
    parser.add_argument('--augment-factor', type=int, default=12,
                        help='Data augmentation factor')
    parser.add_argument('--include-wlasl', action='store_true',
                        help='Include WLASL dataset')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.conv_filters) != len(args.kernel_sizes) or len(args.conv_filters) != len(args.pool_sizes):
        raise ValueError(
            f"Number of conv filters ({len(args.conv_filters)}), "
            f"kernel sizes ({len(args.kernel_sizes)}), and "
            f"pool sizes ({len(args.pool_sizes)}) must be the same"
        )
    
    return args

def create_tf_dataset(features, labels, num_classes, batch_size=32, shuffle=True, buffer_size=1000,
                      augment_online=False, augment_prob=0.3):
    """
    Create a TensorFlow dataset from features and labels with optional online augmentation
    
    Args:
        features: NumPy array of features
        labels: NumPy array of labels
        num_classes: Number of classes for one-hot encoding
        batch_size: Batch size for the dataset
        shuffle: Whether to shuffle the dataset
        buffer_size: Buffer size for shuffling
        augment_online: Whether to apply online augmentation
        augment_prob: Probability of applying augmentation to a batch
        
    Returns:
        TensorFlow dataset
    """
    # Ensure features have consistent dtype
    features = features.astype(np.float32)
    
    # One-hot encode labels
    one_hot_labels = tf.one_hot(labels, depth=num_classes)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((features, one_hot_labels))
    
    # Shuffle if required
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    
    # Add online augmentation if requested
    if augment_online:
        def augment_batch(x, y):
            # Cast to ensure dtype compatibility
            x = tf.cast(x, tf.float32)
            
            # Multiple augmentation techniques with random application
            
            # 1. Random noise augmentation
            if tf.random.uniform([], 0, 1) < augment_prob:
                # Vary noise by position (more at the beginning and end)
                seq_len = tf.shape(x)[0]
                position = tf.range(0, seq_len, dtype=tf.float32) / tf.cast(seq_len - 1, tf.float32)
                position = tf.reshape(position, [-1, 1])  # Shape: [seq_len, 1]
                
                # Create U-shaped position factor (higher at edges, lower in middle)
                position_factor = 1.0 + 0.5 * tf.pow(2.0 * (position - 0.5), 2.0)
                
                # Generate and apply noise
                noise_scale = tf.random.uniform([], 0.01, 0.03)
                noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=noise_scale, dtype=tf.float32)
                noise = noise * position_factor
                x = x + noise
            
            # 2. Random feature scaling
            if tf.random.uniform([], 0, 1) < augment_prob * 0.8:
                # Scale different feature dimensions differently
                scale_factors = tf.random.uniform([tf.shape(x)[1]], 0.85, 1.15)
                x = x * scale_factors
            
            # 3. Random feature masking (simulate dropout at feature level)
            if tf.random.uniform([], 0, 1) < augment_prob * 0.6:
                mask_prob = tf.random.uniform([], 0.05, 0.15)
                random_tensor = tf.random.uniform(tf.shape(x))
                feature_mask = tf.cast(random_tensor >= mask_prob, tf.float32)
                x = x * feature_mask
            
            # 4. Sequence-level transformations (using tf.cond for conditional execution)
            def apply_seq_transform(x_input):
                # Random time warping-like effect
                transform_type = tf.random.uniform([], 0, 1)
                
                # Emphasize beginning or end randomly
                def emphasize_part():
                    seq_len = tf.shape(x_input)[0]
                    position = tf.range(0, seq_len, dtype=tf.float32) / tf.cast(seq_len - 1, tf.float32)
                    position = tf.reshape(position, [-1, 1])
                    
                    # Random emphasis pattern (beginning, end, or both)
                    pattern = tf.random.uniform([], 0, 3, dtype=tf.int32)
                    
                    def emph_begin():
                        return 1.0 - 0.4 * position
                        
                    def emph_end():
                        return 0.6 + 0.4 * position
                        
                    def emph_both():
                        return 0.6 + 0.4 * tf.abs(2.0 * position - 1.0)
                    
                    emphasis = tf.case([
                        (tf.equal(pattern, 0), emph_begin),
                        (tf.equal(pattern, 1), emph_end),
                    ], default=emph_both)
                    
                    return x_input * emphasis
                
                # Apply transformation based on random value
                return tf.cond(transform_type < 0.5, 
                              lambda: emphasize_part(),
                              lambda: x_input)  # No transform 50% of the time
            
            # Apply sequence transform with probability
            x = tf.cond(tf.random.uniform([], 0, 1) < augment_prob * 0.4,
                      lambda: apply_seq_transform(x),
                      lambda: x)
            
            # Ensure values stay in reasonable range
            x = tf.clip_by_value(x, -2.0, 2.0)
            
            return x, y
        
        dataset = dataset.map(augment_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

def normalize_features(features_train, features_val, features_test):
    """
    Normalize features using StandardScaler
    
    Args:
        features_train: Training features
        features_val: Validation features
        features_test: Test features
        
    Returns:
        Normalized features
    """
    # Ensure all inputs have consistent dtype
    features_train = features_train.astype(np.float32)
    if features_val is not None:
        features_val = features_val.astype(np.float32)
    if features_test is not None:
        features_test = features_test.astype(np.float32)
    
    # Reshape for 2D scaling
    original_shape = features_train.shape
    features_train_2d = features_train.reshape(-1, original_shape[-1])
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(features_train_2d)
    
    # Transform all datasets
    features_train_2d = scaler.transform(features_train_2d)
    features_train_normalized = features_train_2d.reshape(original_shape).astype(np.float32)
    
    # Validate
    if features_val is not None:
        features_val_2d = features_val.reshape(-1, features_val.shape[-1])
        features_val_2d = scaler.transform(features_val_2d)
        features_val_normalized = features_val_2d.reshape(features_val.shape).astype(np.float32)
    else:
        features_val_normalized = None
        
    # Test
    if features_test is not None:
        features_test_2d = features_test.reshape(-1, features_test.shape[-1])
        features_test_2d = scaler.transform(features_test_2d)
        features_test_normalized = features_test_2d.reshape(features_test.shape).astype(np.float32)
    else:
        features_test_normalized = None
    
    return features_train_normalized, features_val_normalized, features_test_normalized

def load_hamnosys_data(data_dir):
    """
    Load preprocessed HamNoSys data
    
    Args:
        data_dir: Directory containing preprocessed data
        
    Returns:
        features, labels, and metadata
    """
    # Load features and labels
    features_file = os.path.join(data_dir, 'features.npy')
    labels_file = os.path.join(data_dir, 'labels.npy')
    metadata_file = os.path.join(data_dir, 'metadata.json')
    
    if not os.path.exists(features_file) or not os.path.exists(labels_file):
        raise FileNotFoundError(f"Features or labels files not found in {data_dir}")
    
    # Load data
    features = np.load(features_file)
    labels = np.load(labels_file)
    
    # Sanity check on feature dimensions
    if features.shape[2] < 3:
        print(f"Warning: Feature dimension is {features.shape[2]}, expected at least 3.")
        print("This suggests you're using old data format. Consider regenerating data with the new script.")
    
    # Load metadata if available
    metadata = None
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    return features, labels, metadata

def prepare_data_if_needed(args):
    """
    Run data preparation script if required
    
    Args:
        args: Command line arguments
        
    Returns:
        True if data was prepared successfully, False otherwise
    """
    if args.prepare_data or not os.path.exists(os.path.join(args.data_dir, 'features.npy')):
        print("Preparing HamNoSys data...")
        # Import here to avoid dependency if not needed
        import subprocess
        
        # Create output directory if it doesn't exist
        os.makedirs(args.data_dir, exist_ok=True)
        
        # Run preparation script
        cmd = [
            "python", "prepare_hamnosys_data.py",
            "--data-file", args.hamnosys_data_file,
            "--output-dir", args.data_dir,
            "--min-length", str(args.min_length),
            "--min-samples", str(args.min_samples),
            "--max-classes", str(args.max_classes)
        ]
        
        # Add optional args
        if args.augment:
            cmd.append("--augment")
            cmd.append("--augment-factor")
            cmd.append(str(args.augment_factor))
        
        if args.include_wlasl:
            cmd.append("--include-wlasl")
        
        try:
            subprocess.run(cmd, check=True)
            print("Data preparation completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error preparing data: {e}")
            return False
    
    return True

def calculate_class_weights(labels):
    """
    Calculate class weights based on class frequencies
    
    Args:
        labels: Array of integer class labels
        
    Returns:
        Dictionary of class weights
    """
    # Get unique classes
    unique_classes = np.unique(labels)
    
    # Calculate class weights with balanced configuration
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels
    )
    
    # Smooth the weights to prevent extreme values
    weights = np.clip(weights, 0.5, 5.0)
    
    # Scale weights to have mean = 1.0
    weights = weights / weights.mean()
    
    # Create class weight dictionary
    class_weights = {i: w for i, w in zip(unique_classes, weights)}
    
    print("Class weights:")
    for cls, weight in class_weights.items():
        print(f"  Class {cls}: {weight:.4f}")
    
    return class_weights

def train_model(features, labels, args, fold=None):
    """
    Train model with given parameters and data
    
    Args:
        features: Feature array
        labels: Label array
        args: Command line arguments
        fold: Fold number for cross-validation
        
    Returns:
        Trained model, best validation accuracy, and test accuracy
    """
    # Get number of classes
    num_classes = len(np.unique(labels))
    print(f"Training with {len(features)} sequences and {num_classes} classes")
    
    # Use stratified split for training/validation/test
    if fold is None:
        # Standard train/validation/test split
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features, labels, test_size=args.test_size, random_state=42, stratify=labels
        )
        
        # Adjusted validation size
        val_size_adjusted = args.val_size / (1 - args.test_size)
        
        # Split train+val into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, random_state=42, 
            stratify=y_train_val
        )
    else:
        # Cross-validation fold
        kf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        
        # Get the specific fold
        train_indices, test_indices = list(kf.split(features, labels))[fold]
        X_train_val, X_test = features[train_indices], features[test_indices]
        y_train_val, y_test = labels[train_indices], labels[test_indices]
        
        # Use a portion of training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=args.val_size, random_state=42, 
            stratify=y_train_val
        )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Apply feature normalization if requested
    if args.normalize_features:
        print("Applying feature normalization...")
        X_train, X_val, X_test = normalize_features(X_train, X_val, X_test)
    
    # Create TensorFlow datasets with batching
    train_dataset = create_tf_dataset(
        X_train, y_train, num_classes,
        batch_size=args.batch_size,
        shuffle=True,
        augment_online=True,
        augment_prob=0.4
    )
    
    val_dataset = create_tf_dataset(
        X_val, y_val, num_classes,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    test_dataset = create_tf_dataset(
        X_test, y_test, num_classes,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Calculate class weights if enabled
    class_weights = None
    if args.use_class_weights:
        class_weights = calculate_class_weights(y_train)
    
    # Create output directory
    if fold is None:
        # Single training run - use timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, f'hamnosys_model_{timestamp}')
    else:
        # Cross-validation - use fold number
        output_dir = os.path.join(args.output_dir, f'fold_{fold}')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save args for reproducibility
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create model
    print("Building model...")
    model = ASLRecognitionModel(
        seq_length=args.seq_length,
        feature_dim=X_train.shape[2],
        num_classes=num_classes,
        conv_filters=args.conv_filters,
        kernel_sizes=args.kernel_sizes,
        pool_sizes=args.pool_sizes,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout_rate,
        l2_reg=args.l2_reg,
        use_attention=args.use_attention,
        attention_heads=args.attention_heads,
        use_residual=args.use_residual,
        use_mixed_architecture=args.use_mixed_architecture
    )
    
    # Compile model
    model.compile_model(learning_rate=args.lr, clipnorm=args.clipnorm)
    
    # Print model summary
    model.summary()
    
    # Get callbacks
    callbacks = model.get_callbacks(output_dir=output_dir)
    
    # Train model
    print(f"Training model for {args.epochs} epochs...")
    start_time = time.time()
    
    history = model.fit(
        train_dataset,
        validation_dataset=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weights=class_weights
    )
    
    training_time = time.time() - start_time
    
    # Evaluate model on test set
    print("Evaluating model on test set...")
    # Get evaluation metrics (returns [loss, accuracy, top_2_accuracy, top_3_accuracy])
    eval_metrics = model.evaluate(test_dataset)
    
    # Unpack metrics correctly
    test_loss = eval_metrics[0]
    test_acc = eval_metrics[1]
    test_top2_acc = eval_metrics[2]
    test_top3_acc = eval_metrics[3] if len(eval_metrics) > 3 else 0.0
    
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test accuracy (%): {test_acc*100:.2f}%")
    print(f"Test top-2 accuracy: {test_top2_acc:.4f}")
    print(f"Test top-3 accuracy: {test_top3_acc:.4f}")
    
    # Get validation accuracy from history
    val_accs = history.history['val_accuracy']
    best_val_acc = max(val_accs) if val_accs else 0
    
    # Save evaluation results
    eval_results = {
        'training_time': training_time,
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_accuracy_percent': float(test_acc * 100),
        'test_top2_accuracy': float(test_top2_acc),
        'test_top3_accuracy': float(test_top3_acc),
        'best_val_accuracy': float(best_val_acc)
    }
    
    with open(os.path.join(output_dir, 'evaluation.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    return model, best_val_acc, test_acc

def main():
    """
    Main function for training HamNoSys recognition model
    """
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare data if needed
    if not prepare_data_if_needed(args):
        print("Data preparation failed. Please check the error messages and try again.")
        return
    
    # Load preprocessed data
    print(f"Loading data from {args.data_dir}...")
    features, labels, metadata = load_hamnosys_data(args.data_dir)
    
    if args.use_cross_validation:
        # Cross-validation training
        print(f"Running {args.n_folds}-fold cross-validation...")
        
        val_accs = []
        test_accs = []
        
        for fold in range(args.n_folds):
            print(f"\n--- Fold {fold+1}/{args.n_folds} ---")
            model, val_acc, test_acc = train_model(features, labels, args, fold=fold)
            val_accs.append(val_acc)
            test_accs.append(test_acc)
        
        # Print cross-validation results
        mean_val_acc = np.mean(val_accs)
        mean_test_acc = np.mean(test_accs)
        std_val_acc = np.std(val_accs)
        std_test_acc = np.std(test_accs)
        
        print("\nCross-validation results:")
        print(f"Mean validation accuracy: {mean_val_acc:.4f} ± {std_val_acc:.4f}")
        print(f"Mean test accuracy: {mean_test_acc:.4f} ± {std_test_acc:.4f}")
        
        # Save cross-validation results
        cv_results = {
            'fold_val_accs': [float(acc) for acc in val_accs],
            'fold_test_accs': [float(acc) for acc in test_accs],
            'mean_val_acc': float(mean_val_acc),
            'mean_test_acc': float(mean_test_acc),
            'std_val_acc': float(std_val_acc),
            'std_test_acc': float(std_test_acc)
        }
        
        with open(os.path.join(args.output_dir, 'cv_results.json'), 'w') as f:
            json.dump(cv_results, f, indent=2)
    else:
        # Single training run
        train_model(features, labels, args)
    
    print(f"Training completed. Results saved to {args.output_dir}")

if __name__ == '__main__':
    # Enable memory growth to avoid OOM errors
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Found {len(physical_devices)} GPU(s), memory growth enabled")
    
    main() 