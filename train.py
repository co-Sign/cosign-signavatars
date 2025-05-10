#!/usr/bin/env python3
"""
Unified Training Script for HamNoSys Recognition Model

This script efficiently trains a high-performance CNN-LSTM hybrid model
for sign language recognition from HamNoSys data, with automatic
preprocessing, data augmentation, and evaluation.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import custom modules
from src.model import ASLRecognitionModel

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train Sign Language Recognition Model')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='data/hamnosys_data',
                        help='Directory containing prepared data')
    parser.add_argument('--seq-length', type=int, default=50,
                        help='Length of input sequences')
    parser.add_argument('--prepare-data', action='store_true',
                        help='Run data preparation before training')
    
    # Model parameters
    parser.add_argument('--conv-filters', type=int, nargs='+', default=[64, 128, 256, 512, 768],
                        help='Number of filters in convolutional layers')
    parser.add_argument('--kernel-sizes', type=int, nargs='+', default=[9, 7, 5, 3, 3],
                        help='Kernel sizes for convolutional layers')
    parser.add_argument('--pool-sizes', type=int, nargs='+', default=[2, 2, 2, 2, 2],
                        help='Pool sizes for MaxPooling layers')
    parser.add_argument('--lstm-units', type=int, nargs='+', default=[1024, 512, 256],
                        help='Number of units in LSTM layers')
    parser.add_argument('--dropout-rate', type=float, default=0.4,
                        help='Dropout rate for regularization')
    parser.add_argument('--l2-reg', type=float, default=0.0005,
                        help='L2 regularization factor')
    parser.add_argument('--attention-heads', type=int, default=16,
                        help='Number of attention heads')
    parser.add_argument('--attention', action='store_true', default=True,
                        help='Use self-attention mechanism')
    parser.add_argument('--residual', action='store_true', default=True,
                        help='Use residual connections')
    parser.add_argument('--mixed-architecture', action='store_true', default=True,
                        help='Use mixed LSTM/GRU architecture')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=40,
                        help='Patience for early stopping')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Initial learning rate')
    parser.add_argument('--clipnorm', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data for testing')
    parser.add_argument('--val-size', type=float, default=0.15,
                        help='Fraction of training data for validation')
    parser.add_argument('--use-class-weights', action='store_true', default=True,
                        help='Use class weights for imbalanced data')
    parser.add_argument('--normalize-features', action='store_true', default=True,
                        help='Normalize features using StandardScaler')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Apply data augmentation during training')
    parser.add_argument('--augment-prob', type=float, default=0.5,
                        help='Probability of applying augmentation to a batch')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='models/unified',
                        help='Directory to save models and results')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Create visualizations of training progress')
    
    return parser.parse_args()

def load_data(data_dir):
    """
    Load preprocessed training data
    
    Args:
        data_dir: Directory containing features.npy and labels.npy
        
    Returns:
        features, labels, and metadata if available
    """
    features_path = os.path.join(data_dir, 'features.npy')
    labels_path = os.path.join(data_dir, 'labels.npy')
    metadata_path = os.path.join(data_dir, 'metadata.json')
    
    # Check if files exist
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Features or labels not found in {data_dir}")
    
    # Load data
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    # Load metadata if available
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return features, labels, metadata

def create_tf_dataset(features, labels, num_classes, batch_size=32, 
                      shuffle=True, buffer_size=1000, augment=False, augment_prob=0.5):
    """
    Create a TensorFlow dataset with optional augmentation
    
    Args:
        features: Feature arrays
        labels: Label arrays 
        num_classes: Number of classes for one-hot encoding
        batch_size: Size of batches
        shuffle: Whether to shuffle the data
        buffer_size: Buffer size for shuffling
        augment: Whether to apply data augmentation
        augment_prob: Probability of applying augmentation
        
    Returns:
        TensorFlow dataset
    """
    # Ensure consistent dtype
    features = features.astype(np.float32)
    
    # One-hot encode labels
    one_hot_labels = tf.one_hot(labels, depth=num_classes)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((features, one_hot_labels))
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    
    # Apply online augmentation if requested
    if augment:
        def augment_sequence(x, y):
            # Cast to ensure dtype compatibility
            x = tf.cast(x, tf.float32)
            
            # Random noise augmentation
            if tf.random.uniform([], 0, 1) < augment_prob:
                # Add time-varying noise
                seq_len = tf.shape(x)[0]
                position = tf.range(0, seq_len, dtype=tf.float32) / tf.cast(seq_len - 1, tf.float32)
                position = tf.reshape(position, [-1, 1])
                
                # U-shaped noise (stronger at edges)
                position_factor = 1.0 + 0.5 * tf.pow(2.0 * (position - 0.5), 2.0)
                
                # Generate noise
                noise_scale = tf.random.uniform([], 0.01, 0.03)
                noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=noise_scale)
                noise = noise * position_factor
                x = x + noise
            
            # Random feature scaling
            if tf.random.uniform([], 0, 1) < augment_prob * 0.8:
                # Scale features by random factors
                scale_factors = tf.random.uniform([tf.shape(x)[1]], 0.85, 1.15)
                x = x * scale_factors
            
            # Feature masking (simulate dropout)
            if tf.random.uniform([], 0, 1) < augment_prob * 0.6:
                mask_prob = tf.random.uniform([], 0.05, 0.15)
                random_tensor = tf.random.uniform(tf.shape(x))
                feature_mask = tf.cast(random_tensor >= mask_prob, tf.float32)
                x = x * feature_mask
            
            # Time warping (emphasize beginning/end)
            if tf.random.uniform([], 0, 1) < augment_prob * 0.4:
                seq_len = tf.shape(x)[0]
                position = tf.range(0, seq_len, dtype=tf.float32) / tf.cast(seq_len - 1, tf.float32)
                position = tf.reshape(position, [-1, 1])
                
                # Random pattern: beginning, end, or both
                pattern = tf.random.uniform([], 0, 3, dtype=tf.int32)
                
                if pattern == 0:  # Emphasize beginning
                    emphasis = 1.0 - 0.4 * position
                elif pattern == 1:  # Emphasize end
                    emphasis = 0.6 + 0.4 * position
                else:  # Emphasize both ends
                    emphasis = 0.6 + 0.4 * tf.abs(2.0 * position - 1.0)
                
                x = x * emphasis
            
            # Ensure values stay reasonable
            x = tf.clip_by_value(x, -2.0, 2.0)
            
            return x, y
        
        dataset = dataset.map(augment_sequence, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def normalize_features(train_features, val_features=None, test_features=None):
    """
    Normalize features using StandardScaler
    
    Args:
        train_features: Training features
        val_features: Validation features
        test_features: Test features
        
    Returns:
        Normalized features
    """
    # Store original shapes
    train_shape = train_features.shape
    
    # Reshape to 2D for scaling
    train_2d = train_features.reshape(-1, train_shape[-1])
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(train_2d)
    
    # Transform training data
    train_scaled = scaler.transform(train_2d).reshape(train_shape)
    
    # Transform validation data if provided
    val_scaled = None
    if val_features is not None:
        val_shape = val_features.shape
        val_2d = val_features.reshape(-1, val_shape[-1])
        val_scaled = scaler.transform(val_2d).reshape(val_shape)
    
    # Transform test data if provided
    test_scaled = None
    if test_features is not None:
        test_shape = test_features.shape
        test_2d = test_features.reshape(-1, test_shape[-1])
        test_scaled = scaler.transform(test_2d).reshape(test_shape)
    
    return train_scaled, val_scaled, test_scaled

def calculate_class_weights(labels):
    """
    Calculate balanced class weights
    
    Args:
        labels: Array of class labels
        
    Returns:
        Dictionary of class weights
    """
    # Get unique classes
    unique_classes = np.unique(labels)
    
    # Calculate weights
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels
    )
    
    # Smooth weights to prevent extreme values
    weights = np.clip(weights, 0.5, 5.0)
    
    # Normalize weights to have mean of 1.0
    weights = weights / weights.mean()
    
    # Create dictionary
    class_weights = {i: w for i, w in zip(unique_classes, weights)}
    
    print("Class weights:")
    for cls, weight in sorted(class_weights.items()):
        print(f"  Class {cls}: {weight:.4f}")
    
    return class_weights

def prepare_data(args):
    """
    Run data preparation if needed
    
    Args:
        args: Command line arguments
        
    Returns:
        True if successful, False otherwise
    """
    if not args.prepare_data:
        # Check if data already exists
        features_path = os.path.join(args.data_dir, 'features.npy')
        labels_path = os.path.join(args.data_dir, 'labels.npy')
        
        if os.path.exists(features_path) and os.path.exists(labels_path):
            print(f"Using existing data in {args.data_dir}")
            return True
        else:
            print(f"Data not found in {args.data_dir}, preparation needed")
            args.prepare_data = True
    
    if args.prepare_data:
        print("Preparing data...")
        try:
            # Run the preparation script
            import subprocess
            cmd = [
                "python", "prepare_hamnosys_data.py",
                "--output-dir", args.data_dir,
                "--min-length", "5",
                "--min-samples", "6"
            ]
            # Add additional arguments if needed
            if args.augment:
                cmd.extend(["--augment"])
            
            subprocess.run(cmd, check=True)
            print("Data preparation completed")
            return True
        except Exception as e:
            print(f"Error preparing data: {e}")
            return False
    
    return True

def visualize_training(history, output_dir):
    """
    Create visualizations of training metrics
    
    Args:
        history: Training history
        output_dir: Directory to save visualizations
    """
    # Create output directory if needed
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'training_metrics.png'), dpi=300)
    
    # Plot additional metrics if available
    if 'top_2_accuracy' in history.history:
        plt.figure(figsize=(12, 5))
        
        # Top-2 accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['top_2_accuracy'], label='Training')
        plt.plot(history.history['val_top_2_accuracy'], label='Validation')
        plt.title('Top-2 Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Top-3 accuracy if available
        if 'top_3_accuracy' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['top_3_accuracy'], label='Training')
            plt.plot(history.history['val_top_3_accuracy'], label='Validation')
            plt.title('Top-3 Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'topk_accuracy.png'), dpi=300)
    
    # Plot learning rate if available
    if 'learning_rate' in history.history:
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['learning_rate'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'learning_rate.png'), dpi=300)
    
    print(f"Training visualizations saved to {vis_dir}")

def save_training_summary(args, history, eval_metrics, training_time, output_dir):
    """
    Save training summary to JSON file
    
    Args:
        args: Command line arguments
        history: Training history
        eval_metrics: Evaluation metrics
        training_time: Training time in seconds
        output_dir: Directory to save summary
    """
    # Create a summary dictionary
    summary = {
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_seconds": training_time,
        "training_time_hours": training_time / 3600,
        "model_parameters": {
            "conv_filters": args.conv_filters,
            "kernel_sizes": args.kernel_sizes,
            "pool_sizes": args.pool_sizes,
            "lstm_units": args.lstm_units,
            "dropout_rate": args.dropout_rate,
            "l2_reg": args.l2_reg,
            "attention": args.attention,
            "attention_heads": args.attention_heads,
            "residual": args.residual,
            "mixed_architecture": args.mixed_architecture
        },
        "training_parameters": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.lr,
            "clipnorm": args.clipnorm,
            "use_class_weights": args.use_class_weights,
            "normalize_features": args.normalize_features,
            "augment": args.augment,
            "augment_prob": args.augment_prob
        },
        "data_splits": {
            "test_size": args.test_size,
            "val_size": args.val_size
        },
        "final_metrics": {
            "test_loss": float(eval_metrics[0]),
            "test_accuracy": float(eval_metrics[1]),
            "test_accuracy_percent": float(eval_metrics[1] * 100)
        },
        "best_epoch": {
            "val_accuracy": float(max(history.history['val_accuracy'])),
            "val_loss": float(min(history.history['val_loss']))
        },
        "training_history": {
            metric: [float(val) for val in values] 
            for metric, values in history.history.items()
        }
    }
    
    # Add top-k metrics if available
    if len(eval_metrics) > 2:
        summary["final_metrics"]["test_top2_accuracy"] = float(eval_metrics[2])
    if len(eval_metrics) > 3:
        summary["final_metrics"]["test_top3_accuracy"] = float(eval_metrics[3])
    
    # Save to file
    summary_path = os.path.join(output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Also save a readable text summary
    text_summary_path = os.path.join(output_dir, 'training_summary.txt')
    with open(text_summary_path, 'w') as f:
        f.write(f"Training Summary ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write("="*50 + "\n\n")
        
        f.write("Model Architecture:\n")
        f.write(f"  Conv filters: {args.conv_filters}\n")
        f.write(f"  Kernel sizes: {args.kernel_sizes}\n")
        f.write(f"  LSTM units: {args.lstm_units}\n")
        f.write(f"  Dropout: {args.dropout_rate}\n")
        f.write(f"  L2 regularization: {args.l2_reg}\n")
        f.write(f"  Attention: {args.attention} (heads: {args.attention_heads})\n")
        f.write(f"  Residual connections: {args.residual}\n")
        f.write(f"  Mixed architecture: {args.mixed_architecture}\n\n")
        
        f.write("Training Parameters:\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Initial learning rate: {args.lr}\n")
        f.write(f"  Data augmentation: {args.augment}\n\n")
        
        f.write("Results:\n")
        f.write(f"  Training time: {training_time/3600:.2f} hours\n")
        f.write(f"  Final test accuracy: {eval_metrics[1]*100:.2f}%\n")
        f.write(f"  Final test loss: {eval_metrics[0]:.4f}\n")
        if len(eval_metrics) > 2:
            f.write(f"  Top-2 accuracy: {eval_metrics[2]*100:.2f}%\n")
        if len(eval_metrics) > 3:
            f.write(f"  Top-3 accuracy: {eval_metrics[3]*100:.2f}%\n")
        f.write(f"  Best validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%\n")
    
    print(f"Training summary saved to {summary_path}")

def main():
    """
    Main training function
    """
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare data if needed
    if not prepare_data(args):
        print("Failed to prepare data. Please check errors.")
        return
    
    # Load data
    print(f"Loading data from {args.data_dir}...")
    features, labels, metadata = load_data(args.data_dir)
    
    # Print data information
    print(f"Loaded {len(features)} samples with {len(np.unique(labels))} classes")
    print(f"Feature shape: {features.shape}")
    
    # Split data
    num_classes = len(np.unique(labels))
    
    # First split out test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, labels, test_size=args.test_size, random_state=42, stratify=labels
    )
    
    # Then split training into train and validation
    val_ratio = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=42, stratify=y_train_val
    )
    
    # Print split information
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Normalize features if requested
    if args.normalize_features:
        print("Normalizing features...")
        X_train, X_val, X_test = normalize_features(X_train, X_val, X_test)
    
    # Create TensorFlow datasets
    train_dataset = create_tf_dataset(
        X_train, y_train, num_classes,
        batch_size=args.batch_size,
        shuffle=True,
        augment=args.augment,
        augment_prob=args.augment_prob
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
    
    # Calculate class weights if requested
    class_weights = None
    if args.use_class_weights:
        class_weights = calculate_class_weights(y_train)
    
    # Build model
    print("Building model...")
    model = ASLRecognitionModel(
        seq_length=args.seq_length,
        feature_dim=features.shape[2],
        num_classes=num_classes,
        conv_filters=args.conv_filters,
        kernel_sizes=args.kernel_sizes,
        pool_sizes=args.pool_sizes,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout_rate,
        l2_reg=args.l2_reg,
        use_attention=args.attention,
        attention_heads=args.attention_heads,
        use_residual=args.residual,
        use_mixed_architecture=args.mixed_architecture,
        use_self_attention=True
    )
    
    # Compile model
    model.compile_model(learning_rate=args.lr, clipnorm=args.clipnorm)
    
    # Print model summary
    model.summary()
    
    # Set up callbacks
    callbacks = model.get_callbacks(args.output_dir, patience=args.patience)
    
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
    print(f"Training completed in {training_time/3600:.2f} hours")
    
    # Evaluate model
    print("Evaluating model on test set...")
    eval_metrics = model.evaluate(test_dataset)
    
    # Print evaluation results
    print(f"Test loss: {eval_metrics[0]:.4f}")
    print(f"Test accuracy: {eval_metrics[1]*100:.2f}%")
    if len(eval_metrics) > 2:
        print(f"Test Top-2 accuracy: {eval_metrics[2]*100:.2f}%")
    if len(eval_metrics) > 3:
        print(f"Test Top-3 accuracy: {eval_metrics[3]*100:.2f}%")
    
    # Visualize training if requested
    if args.visualize:
        visualize_training(history, args.output_dir)
    
    # Save training summary
    save_training_summary(args, history, eval_metrics, training_time, args.output_dir)
    
    # Save best model to main file
    model.save(os.path.join(args.output_dir, 'model.keras'))
    
    print(f"Training completed. Model and results saved to {args.output_dir}")

if __name__ == "__main__":
    # Enable memory growth for GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except:
            pass
    
    main() 