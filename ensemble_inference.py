#!/usr/bin/env python3
"""
Ensemble Inference for HamNoSys Recognition

This script combines predictions from multiple models to create a more robust
classification system for sign language recognition.
"""
import os
import argparse
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from prepare_hamnosys_data import encode_hamnosys
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble inference for HamNoSys recognition')
    parser.add_argument('--model-paths', type=str, nargs='+', required=True,
                      help='List of paths to trained models')
    parser.add_argument('--weights', type=float, nargs='+', default=None,
                      help='Optional weights for each model (default: equal weighting)')
    parser.add_argument('--data-dir', type=str, default='data/hamnosys_data',
                      help='Directory containing test data')
    parser.add_argument('--input', type=str, default=None,
                      help='Input HamNoSys sequence to classify')
    parser.add_argument('--top-k', type=int, default=3,
                      help='Number of top predictions to display')
    parser.add_argument('--visualize', action='store_true',
                      help='Generate visualization of results')
    parser.add_argument('--output-dir', type=str, default='ensemble_results',
                      help='Directory to save results')
    parser.add_argument('--ensemble-method', type=str, default='weighted_average',
                      choices=['average', 'weighted_average', 'max_vote', 'product'],
                      help='Method to combine model predictions')
    return parser.parse_args()

def categorical_focal_loss(alpha=0.3, gamma=2.5):
    """
    Focal Loss implementation for model loading
    
    Args:
        alpha: Weighting factor
        gamma: Focusing parameter
        
    Returns:
        Focal loss function
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

def load_model(model_path):
    """
    Load a trained model with custom objects
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model
    """
    print(f"Loading model from {model_path}...")
    
    # Custom objects dictionary for the model
    custom_objects = {
        'categorical_focal_loss': categorical_focal_loss(),
        'top_2_accuracy': tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
        'top_3_accuracy': tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
    }
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"Model loaded successfully: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading method...")
        try:
            # Alternative loading method
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(
                optimizer='adam',
                loss=categorical_focal_loss(),
                metrics=['accuracy', 
                        tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
                        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
            )
            print(f"Model loaded with alternative method: {model_path}")
            return model
        except Exception as e2:
            print(f"Failed to load model with alternative method: {e2}")
            return None

def load_test_data(data_dir):
    """
    Load and preprocess test data
    
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
    
    # Load class mapping
    class_names = {}
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

def ensemble_predict(models, X_test, method='weighted_average', weights=None):
    """
    Generate ensemble predictions using multiple models
    
    Args:
        models: List of loaded models
        X_test: Test features
        method: Ensemble method (average, weighted_average, max_vote, product)
        weights: Optional weights for each model
        
    Returns:
        Ensemble predictions
    """
    # Get individual model predictions
    predictions = []
    for i, model in enumerate(models):
        print(f"Getting predictions from model {i+1}/{len(models)}...")
        pred = model.predict(X_test)
        predictions.append(pred)
    
    # Set default weights if not provided
    if weights is None:
        weights = np.ones(len(models)) / len(models)
    else:
        # Normalize weights to sum to 1
        weights = np.array(weights) / np.sum(weights)
    
    # Apply ensemble method
    if method == 'average':
        # Simple averaging
        ensemble_pred = np.mean(predictions, axis=0)
    
    elif method == 'weighted_average':
        # Weighted averaging
        ensemble_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            ensemble_pred += pred * weights[i]
    
    elif method == 'max_vote':
        # Maximum confidence
        ensemble_pred = np.max(predictions, axis=0)
    
    elif method == 'product':
        # Product of probabilities (tends to be more conservative)
        ensemble_pred = predictions[0].copy()
        for pred in predictions[1:]:
            ensemble_pred *= pred
        # Normalize to sum to 1 along class axis
        ensemble_pred /= np.sum(ensemble_pred, axis=1, keepdims=True)
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    return ensemble_pred

def predict_sequence(models, sequence, class_names, top_k=3, method='weighted_average', weights=None):
    """
    Make ensemble prediction for a single HamNoSys sequence
    
    Args:
        models: List of loaded models
        sequence: Input HamNoSys sequence
        class_names: Dictionary mapping class indices to names
        top_k: Number of top predictions to display
        method: Ensemble method
        weights: Optional weights for each model
        
    Returns:
        Top k predictions and their probabilities
    """
    # Encode sequence
    encoded = encode_hamnosys(sequence)
    encoded = np.expand_dims(encoded, axis=0)  # Add batch dimension
    encoded = normalize_features(encoded)
    
    # Get individual model predictions
    predictions = []
    for model in models:
        pred = model.predict(encoded)[0]
        predictions.append(pred)
    
    # Set default weights if not provided
    if weights is None:
        weights = np.ones(len(models)) / len(models)
    else:
        # Normalize weights to sum to 1
        weights = np.array(weights) / np.sum(weights)
    
    # Apply ensemble method
    if method == 'average':
        ensemble_pred = np.mean(predictions, axis=0)
    elif method == 'weighted_average':
        ensemble_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            ensemble_pred += pred * weights[i]
    elif method == 'max_vote':
        ensemble_pred = np.max(predictions, axis=0)
    elif method == 'product':
        ensemble_pred = predictions[0].copy()
        for pred in predictions[1:]:
            ensemble_pred *= pred
        ensemble_pred /= np.sum(ensemble_pred)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    # Get top k predictions
    top_indices = np.argsort(ensemble_pred)[-top_k:][::-1]
    top_probs = ensemble_pred[top_indices]
    top_classes = [class_names.get(idx, f"Class {idx}") for idx in top_indices]
    
    return list(zip(top_classes, top_probs))

def visualize_ensemble_results(X_test, y_test, y_pred, class_names, output_dir, top_k=3, method='weighted_average'):
    """
    Visualize ensemble prediction results
    
    Args:
        X_test: Test features
        y_test: True labels
        y_pred: Predicted probabilities from ensemble
        class_names: Dictionary mapping class indices to names
        output_dir: Output directory
        top_k: Number of top predictions to display
        method: Ensemble method used
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate accuracy metrics
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_pred_classes == y_test)
    
    # Per-class metrics
    unique_classes = np.unique(y_test)
    class_metrics = {}
    
    for cls in unique_classes:
        mask = y_test == cls
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred_classes[mask] == cls)
            class_name = class_names.get(cls, f"Class {cls}")
            class_metrics[class_name] = {
                'accuracy': float(class_acc),
                'count': int(np.sum(mask))
            }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Class names for plotting
    class_list = [class_names.get(i, f"Class {i}") for i in range(len(cm))]
    
    # Save confusion matrix
    plt.figure(figsize=(14, 12))
    sns.set(font_scale=1.2)
    heatmap = sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='viridis',
        xticklabels=class_list, 
        yticklabels=class_list,
        cbar_kws={'label': 'Normalized Frequency'}
    )
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(f'Ensemble Confusion Matrix ({method})\nAccuracy: {accuracy:.2%}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ensemble_confusion_matrix_{method}.png'), dpi=300)
    plt.close()
    
    # Top-k accuracy
    top_k_accuracy = []
    for k in range(1, top_k + 1):
        top_k_preds = np.argsort(y_pred, axis=1)[:, -k:]
        top_k_acc = np.mean([y_test[i] in top_k_preds[i] for i in range(len(y_test))])
        top_k_accuracy.append(top_k_acc)
    
    # Plot top-k accuracy
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        range(1, top_k + 1), 
        top_k_accuracy, 
        color=sns.color_palette("viridis", top_k)
    )
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
    plt.xlabel('k', fontsize=13)
    plt.ylabel('Accuracy', fontsize=13)
    plt.title(f'Ensemble Top-k Accuracy ({method})', fontsize=15)
    plt.xticks(range(1, top_k + 1))
    plt.ylim(0, min(1.1, max(top_k_accuracy) * 1.15))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ensemble_top_k_accuracy_{method}.png'), dpi=300)
    plt.close()
    
    # Per-class accuracy bar chart
    sorted_metrics = sorted(class_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    class_names_sorted = [x[0] for x in sorted_metrics]
    class_accs_sorted = [x[1]['accuracy'] for x in sorted_metrics]
    class_counts_sorted = [x[1]['count'] for x in sorted_metrics]
    
    # Plot horizontal bar chart with counts
    plt.figure(figsize=(14, len(class_metrics) * 0.5))
    bars = plt.barh(
        class_names_sorted,
        class_accs_sorted,
        color=sns.color_palette("viridis", len(class_metrics))
    )
    for i, bar in enumerate(bars):
        width = bar.get_width()
        count = class_counts_sorted[i]
        plt.text(
            max(width + 0.02, 0.02),
            bar.get_y() + bar.get_height()/2.,
            f'{width:.2%} (n={count})',
            ha='left',
            va='center',
            fontsize=11
        )
    plt.xlabel('Accuracy', fontsize=13)
    plt.title(f'Ensemble Per-class Accuracy ({method})', fontsize=15)
    plt.xlim(0, 1.05)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ensemble_per_class_accuracy_{method}.png'), dpi=300)
    plt.close()
    
    # Generate classification report
    report = classification_report(
        y_test, 
        y_pred_classes, 
        target_names=[class_names.get(i, f"Class {i}") for i in np.unique(y_test)]
    )
    
    # Save detailed metrics
    metrics = {
        'ensemble_method': method,
        'accuracy': float(accuracy),
        'top_k_accuracy': {f'top_{k+1}': float(acc) for k, acc in enumerate(top_k_accuracy)},
        'per_class_metrics': class_metrics
    }
    
    with open(os.path.join(output_dir, f'ensemble_metrics_{method}.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    with open(os.path.join(output_dir, f'classification_report_{method}.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Ensemble Method: {method}\n\n")
        f.write(report)
    
    print(f"Visualizations and metrics saved to {output_dir}")
    print(f"\nEnsemble Results ({method}):")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Top-3 Accuracy: {top_k_accuracy[2]:.2%}")

def main():
    """
    Main function for ensemble inference
    """
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    models = []
    for model_path in args.model_paths:
        model = load_model(model_path)
        if model is not None:
            models.append(model)
    
    if not models:
        print("No models could be loaded. Exiting.")
        return
    
    print(f"Loaded {len(models)} models for ensemble prediction")
    
    # Set model weights
    weights = args.weights
    if weights is not None and len(weights) != len(models):
        print(f"Warning: Number of weights ({len(weights)}) doesn't match number of models ({len(models)})")
        print("Using equal weights instead")
        weights = None
    
    # Load test data
    X_test, y_test, class_names = load_test_data(args.data_dir)
    print(f"Loaded test data with {len(X_test)} samples and {len(class_names)} classes")
    
    # Predict single input if provided
    if args.input:
        print(f"\nPredicting input sequence: {args.input}")
        predictions = predict_sequence(
            models, 
            args.input, 
            class_names, 
            args.top_k, 
            args.ensemble_method, 
            weights
        )
        print(f"\nTop {args.top_k} predictions ({args.ensemble_method}):")
        for cls, prob in predictions:
            print(f"{cls}: {prob:.2%}")
    
    # Run ensemble inference on test data
    print(f"\nRunning ensemble inference using {args.ensemble_method} method...")
    start_time = time.time()
    y_pred = ensemble_predict(models, X_test, args.ensemble_method, weights)
    inference_time = time.time() - start_time
    
    print(f"Inference completed in {inference_time:.2f} seconds")
    print(f"Average inference time: {inference_time / len(X_test) * 1000:.2f} ms per sample")
    
    # Calculate and print accuracy
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_pred_classes == y_test)
    print(f"Ensemble accuracy: {accuracy:.2%}")
    
    # Calculate top-k accuracy
    top_k = args.top_k
    top_k_preds = np.argsort(y_pred, axis=1)[:, -top_k:]
    top_k_acc = np.mean([y_test[i] in top_k_preds[i] for i in range(len(y_test))])
    print(f"Ensemble top-{top_k} accuracy: {top_k_acc:.2%}")
    
    # Generate visualizations
    if args.visualize:
        print("\nGenerating ensemble visualizations...")
        visualize_ensemble_results(
            X_test, 
            y_test, 
            y_pred, 
            class_names, 
            args.output_dir, 
            args.top_k, 
            args.ensemble_method
        )
    
    # Save ensemble results
    results = {
        'ensemble_method': args.ensemble_method,
        'num_models': len(models),
        'model_paths': args.model_paths,
        'accuracy': float(accuracy),
        f'top_{top_k}_accuracy': float(top_k_acc),
        'inference_time_seconds': float(inference_time),
        'inference_time_per_sample_ms': float(inference_time / len(X_test) * 1000),
        'num_test_samples': len(X_test),
        'num_classes': len(np.unique(y_test))
    }
    
    with open(os.path.join(args.output_dir, f'ensemble_results_{args.ensemble_method}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {os.path.join(args.output_dir, f'ensemble_results_{args.ensemble_method}.json')}")

if __name__ == '__main__':
    main() 