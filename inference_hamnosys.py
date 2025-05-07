#!/usr/bin/env python3
"""
Inference script for HamNoSys recognition model

This script loads a trained model and runs inference on test data
or a specific input sequence. It also provides visualization of
model predictions and confidence scores.
"""
import os
import argparse
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from prepare_hamnosys_data import encode_hamnosys
from sklearn.metrics import confusion_matrix
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with trained HamNoSys model')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to the trained model (.keras file)')
    parser.add_argument('--data-dir', type=str, default='data/hamnosys_data',
                      help='Directory containing test data')
    parser.add_argument('--input', type=str, default=None,
                      help='Input HamNoSys sequence to classify')
    parser.add_argument('--top-k', type=int, default=3,
                      help='Number of top predictions to display')
    parser.add_argument('--visualize', action='store_true',
                      help='Generate visualization of results')
    parser.add_argument('--output-dir', type=str, default='inference_results',
                      help='Directory to save visualization results')
    return parser.parse_args()

def categorical_focal_loss(alpha=0.3, gamma=2.5):
    """
    Implementation of Focal Loss with updated parameters
    
    Args:
        alpha: Weighting factor in range (0,1) to balance positive vs negative examples
        gamma: Exponent of the modulating factor (1 - p_t) to focus on hard examples
        
    Returns:
        Loss function
    """
    def loss(y_true, y_pred):
        # Clip predictions to prevent NaN
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        
        # Calculate focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1.0 - y_pred, gamma) * y_true
        focal_loss = alpha * weight * cross_entropy
        
        return tf.reduce_sum(focal_loss, axis=-1)
    
    return loss

def load_test_data(data_dir):
    """
    Load and split data for testing
    
    Args:
        data_dir: Directory containing data
        
    Returns:
        Test features, labels, and class names
    """
    # Load features and labels
    features_file = os.path.join(data_dir, 'features.npy')
    labels_file = os.path.join(data_dir, 'labels.npy')
    
    if not os.path.exists(features_file) or not os.path.exists(labels_file):
        raise FileNotFoundError(f"Data files not found in {data_dir}")
    
    # Load data
    features = np.load(features_file)
    labels = np.load(labels_file)
    
    print(f"Loaded data with shape: {features.shape}, {labels.shape}")
    
    # Load metadata
    metadata_file = os.path.join(data_dir, 'metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        # Try both class naming conventions
        class_names = metadata.get('classes', metadata.get('label_map', {}))
        if not class_names and 'classes' in metadata:
            class_names = metadata['classes']
    else:
        # Try label_mapping.json as fallback
        mapping_file = os.path.join(data_dir, 'label_mapping.json')
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                class_names = json.load(f)
        else:
            print("Warning: No class mapping found, using numeric labels")
            class_names = {i: f"Class {i}" for i in range(len(np.unique(labels)))}
    
    # Convert string keys to integers if needed
    if any(isinstance(k, str) for k in class_names.keys()):
        class_names = {int(k): v for k, v in class_names.items()}
    
    # Apply feature normalization
    features = normalize_features(features)
    
    # Split data into test set (20% of data)
    num_samples = len(features)
    indices = np.random.permutation(num_samples)
    
    test_size = int(0.2 * num_samples)
    test_indices = indices[:test_size]
    
    # Get test data
    X_test = features[test_indices]
    y_test = labels[test_indices]
    
    return X_test, y_test, class_names

def normalize_features(features):
    """
    Normalize features using standardization
    
    Args:
        features: Features to normalize
        
    Returns:
        Normalized features
    """
    # Reshape for 2D scaling
    original_shape = features.shape
    features_2d = features.reshape(-1, original_shape[-1])
    
    # Calculate mean and std for each feature
    means = np.mean(features_2d, axis=0)
    stds = np.std(features_2d, axis=0)
    stds[stds == 0] = 1.0  # Avoid division by zero
    
    # Normalize
    features_2d = (features_2d - means) / stds
    
    # Reshape back and convert to float32
    normalized_features = features_2d.reshape(original_shape).astype(np.float32)
    
    return normalized_features

def predict_sequence(model, sequence, class_names, top_k=3):
    """
    Make a prediction for a single HamNoSys sequence
    
    Args:
        model: Trained model
        sequence: Input HamNoSys sequence
        class_names: Dictionary mapping class indices to names
        top_k: Number of top predictions to display
        
    Returns:
        Top k predictions and their probabilities
    """
    # Encode sequence
    encoded = encode_hamnosys(sequence)
    encoded = np.expand_dims(encoded, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(encoded)[0]
    
    # Get top k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_probs = predictions[top_indices]
    top_classes = [class_names.get(str(idx), f"Class {idx}") for idx in top_indices]
    
    return list(zip(top_classes, top_probs))

def visualize_results(X_test, y_test, y_pred, class_names, output_dir, top_k=3):
    """
    Visualize prediction results with enhanced visualizations
    
    Args:
        X_test: Test features
        y_test: True labels
        y_pred: Predicted probabilities
        class_names: Dictionary mapping class indices to names
        output_dir: Directory to save visualization results
        top_k: Number of top predictions to display
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Overall accuracy
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_pred_classes == y_test)
    
    # Calculate per-class metrics
    unique_classes = np.unique(y_test)
    per_class_acc = {}
    
    for cls in unique_classes:
        class_mask = y_test == cls
        class_acc = np.mean(y_pred_classes[class_mask] == cls)
        class_name = class_names.get(cls, f"Class {cls}")
        per_class_acc[class_name] = class_acc
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Get class names for plotting
    class_list = [class_names.get(i, f"Class {i}") for i in range(len(cm))]
    
    # Enhanced confusion matrix with better styling
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.1)
    
    # Use a custom colormap with better contrast
    heatmap = sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='viridis',
        xticklabels=class_list, 
        yticklabels=class_list,
        cbar_kws={'label': 'Normalized Frequency'}
    )
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Normalized Confusion Matrix\nOverall Accuracy: {accuracy:.2%}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Plot top-k accuracy
    top_k_accuracy = []
    for k in range(1, top_k + 1):
        top_k_preds = np.argsort(y_pred, axis=1)[:, -k:]
        top_k_acc = np.mean([y_test[i] in top_k_preds[i] for i in range(len(y_test))])
        top_k_accuracy.append(top_k_acc)
    
    # Enhanced top-k accuracy plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        range(1, top_k + 1), 
        top_k_accuracy, 
        color=sns.color_palette("viridis", top_k)
    )
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.01,
            f'{height:.2%}',
            ha='center', 
            va='bottom',
            fontsize=12
        )
    
    plt.xlabel('k', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Top-k Accuracy', fontsize=14)
    plt.xticks(range(1, top_k + 1))
    plt.ylim(0, min(1.0, max(top_k_accuracy) * 1.2))  # Set y-limit with some padding
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_k_accuracy.png'), dpi=300)
    plt.close()
    
    # Save per-class accuracy as a bar chart
    plt.figure(figsize=(12, len(per_class_acc) * 0.4))
    
    # Sort classes by accuracy
    sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
    class_names_sorted = [x[0] for x in sorted_classes]
    class_accs_sorted = [x[1] for x in sorted_classes]
    
    # Plot horizontal bar chart
    bars = plt.barh(
        class_names_sorted,
        class_accs_sorted,
        color=sns.color_palette("viridis", len(per_class_acc))
    )
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(
            max(width + 0.02, 0.02),
            bar.get_y() + bar.get_height()/2.,
            f'{width:.2%}',
            ha='left',
            va='center',
            fontsize=10
        )
    
    plt.xlabel('Accuracy', fontsize=12)
    plt.title('Per-class Accuracy', fontsize=14)
    plt.xlim(0, 1.05)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_accuracy.png'), dpi=300)
    plt.close()
    
    # Generate sample predictions with enhanced visualization
    num_samples = min(10, len(X_test))
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        true_label = y_test[idx]
        true_class = class_names.get(true_label, f"Class {true_label}")
        pred_probs = y_pred[idx]
        top_indices = np.argsort(pred_probs)[-top_k:][::-1]
        top_probs = pred_probs[top_indices]
        top_classes = [class_names.get(idx, f"Class {idx}") for idx in top_indices]
        
        # Create a color map: green for correct prediction, red for wrong
        colors = ['#ff9999' for _ in range(top_k)]  # Default to red
        if true_label in top_indices:
            # Find position of true label in top_indices
            correct_idx = np.where(top_indices == true_label)[0][0]
            colors[correct_idx] = '#99ff99'  # Green for correct prediction
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(top_classes, top_probs, color=colors)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 0.01,
                bar.get_y() + bar.get_height()/2.,
                f'{width:.2%}',
                ha='left',
                va='center',
                fontsize=10
            )
        
        plt.xlabel('Probability', fontsize=12)
        plt.title(f'Sample {i+1} Predictions\nTrue: {true_class}', fontsize=14)
        plt.xlim(0, 1.1)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{i+1}_predictions.png'), dpi=300)
        plt.close()
    
    # Save overall results as text file
    with open(os.path.join(output_dir, 'detailed_results.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy:.2%})\n\n")
        
        for k in range(1, top_k + 1):
            f.write(f"Top-{k} Accuracy: {top_k_accuracy[k-1]:.4f} ({top_k_accuracy[k-1]:.2%})\n")
        
        f.write("\nPer-class Accuracy:\n")
        for cls_name, cls_acc in sorted_classes:
            f.write(f"{cls_name}: {cls_acc:.4f} ({cls_acc:.2%})\n")
    
    print(f"Enhanced visualizations saved to {output_dir}")

def main():
    """
    Main function for inference
    """
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Register the custom loss function
    print(f"Loading model from {args.model_path}...")
    
    # Custom objects dictionary for the model
    custom_objects = {
        'categorical_focal_loss': categorical_focal_loss(),
        'top_2_accuracy': tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
        'top_3_accuracy': tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
    }
    
    # Load model with custom objects
    try:
        model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading method...")
        # Alternative loading method
        model = tf.keras.models.load_model(args.model_path, compile=False)
        model.compile(
            optimizer='adam',
            loss=categorical_focal_loss(),
            metrics=['accuracy', 
                    tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        print("Model loaded with alternative method")
    
    # Print model summary
    model.summary()
    
    # Load test data
    X_test, y_test, class_names = load_test_data(args.data_dir)
    print(f"Loaded test data with {len(X_test)} samples and {len(class_names)} classes")
    
    # Run inference on test data
    print("Running inference on test data...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    
    avg_time_per_sample = inference_time / len(X_test) * 1000  # in milliseconds
    print(f"Inference completed in {inference_time:.2f} seconds")
    print(f"Average inference time: {avg_time_per_sample:.2f} ms per sample")
    
    # Calculate and print accuracy
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_pred_classes == y_test)
    print(f"Test accuracy: {accuracy:.2%}")
    
    # Calculate top-k accuracy
    top_k = args.top_k
    top_k_preds = np.argsort(y_pred, axis=1)[:, -top_k:]
    top_k_acc = np.mean([y_test[i] in top_k_preds[i] for i in range(len(y_test))])
    print(f"Top-{top_k} accuracy: {top_k_acc:.2%}")
    
    # If input sequence is provided, predict it
    if args.input:
        print(f"\nPredicting input sequence: {args.input}")
        predictions = predict_sequence(model, args.input, class_names, top_k)
        print("\nTop predictions:")
        for cls, prob in predictions:
            print(f"{cls}: {prob:.2%}")
    
    # Generate visualizations if requested
    if args.visualize:
        print("\nGenerating enhanced visualizations...")
        visualize_results(X_test, y_test, y_pred, class_names, args.output_dir, top_k)
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        f'top_{top_k}_accuracy': float(top_k_acc),
        'inference_time_seconds': float(inference_time),
        'inference_time_per_sample_ms': float(avg_time_per_sample),
        'num_test_samples': len(X_test),
        'num_classes': len(class_names)
    }
    
    with open(os.path.join(args.output_dir, 'inference_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {os.path.join(args.output_dir, 'inference_results.json')}")

if __name__ == '__main__':
    main() 