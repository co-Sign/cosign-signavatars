import os
import argparse
import tensorflow as tf
import json
from datetime import datetime

# Import custom modules
from preprocessing import DataLoader
from model import ASLRecognitionModel

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train ASL recognition model')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing .pkl files')
    parser.add_argument('--labels-file', type=str, default=None,
                        help='JSON file containing labels for each sequence')
    parser.add_argument('--seq-length', type=int, default=50,
                        help='Number of frames to pad/truncate sequences to')
    
    # Model parameters
    parser.add_argument('--conv-filters', type=int, nargs='+', default=[64, 128],
                        help='Number of filters in each convolutional layer')
    parser.add_argument('--kernel-sizes', type=int, nargs='+', default=[3, 3],
                        help='Kernel size for each convolutional layer')
    parser.add_argument('--pool-sizes', type=int, nargs='+', default=[2, 2],
                        help='Pool size for each MaxPooling layer')
    parser.add_argument('--lstm-units', type=int, nargs='+', default=[128, 64],
                        help='Number of units in each LSTM layer')
    parser.add_argument('--dropout-rate', type=float, default=0.5,
                        help='Dropout rate for regularization')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--clipnorm', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--val-size', type=float, default=0.1,
                        help='Fraction of training data to use for validation')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='./models',
                        help='Directory to save trained model and logs')
    
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

def main():
    """
    Main function for training ASL recognition model
    """
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f'asl_model_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save args for reproducibility
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load and preprocess data
    print(f"Loading data from {args.data_dir}...")
    data_loader = DataLoader(
        data_dir=args.data_dir,
        seq_length=args.seq_length
    )
    
    # Load dataset
    features, labels, file_names = data_loader.load_dataset(labels_file=args.labels_file)
    
    if labels is None:
        raise ValueError("No labels found. Please provide a valid labels file.")
    
    print(f"Loaded {len(features)} sequences with {len(set(labels))} unique classes")
    
    # Normalize features
    features = data_loader.normalize_features(features)
    
    # Get number of classes
    num_classes = len(set(labels))
    
    # Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.split_data(
        features, labels, 
        test_size=args.test_size,
        val_size=args.val_size
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create TensorFlow datasets
    train_dataset = data_loader.create_tf_dataset(
        X_train, y_train, 
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_dataset = data_loader.create_tf_dataset(
        X_val, y_val, 
        batch_size=args.batch_size,
        shuffle=False
    )
    
    test_dataset = data_loader.create_tf_dataset(
        X_test, y_test, 
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Create model
    print("Building model...")
    feature_dim = features.shape[2]  # Get actual feature dimension
    model = ASLRecognitionModel(
        seq_length=args.seq_length,
        feature_dim=feature_dim,
        num_classes=num_classes,
        conv_filters=args.conv_filters,
        kernel_sizes=args.kernel_sizes,
        pool_sizes=args.pool_sizes,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout_rate
    )
    
    # Compile model
    model.compile_model(learning_rate=args.lr, clipnorm=args.clipnorm)
    
    # Print model summary
    model.summary()
    
    # Get callbacks
    callbacks = model.get_callbacks(output_dir=output_dir)
    
    # Model checkpoint
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_dir, 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # TensorBoard logging
    callbacks.append(model_checkpoint)
    
    # Train model
    print(f"Training model for {args.epochs} epochs...")
    history = model.fit(
        train_dataset,
        validation_dataset=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Evaluate model on test set
    print("Evaluating model on test set...")
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test accuracy (%): {test_acc*100:.2f}%")
    
    # Save evaluation results
    eval_results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_accuracy_percent': float(test_acc * 100)
    }
    
    with open(os.path.join(output_dir, 'evaluation.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # Save final model
    model.save(os.path.join(output_dir, 'final_model.keras'))
    print(f"Model saved to {output_dir}")

if __name__ == '__main__':
    # Enable memory growth to avoid OOM errors
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Found {len(physical_devices)} GPU(s), memory growth enabled")
    
    main() 