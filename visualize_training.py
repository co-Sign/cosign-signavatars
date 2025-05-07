#!/usr/bin/env python3
"""
Visualization script for HamNoSys model training results
Generates plots and statistics from training history
"""
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize HamNoSys model training results')
    parser.add_argument('--model-dir', type=str, required=True,
                      help='Directory containing model and training logs')
    parser.add_argument('--data-dir', type=str, default='data/hamnosys_data',
                      help='Directory containing test data')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Directory to save visualizations (defaults to model_dir/visualizations)')
    parser.add_argument('--include-model-analysis', action='store_true',
                      help='Include detailed model analysis (loads model)')
    parser.add_argument('--show-plots', action='store_true',
                      help='Show plots instead of just saving them')
    return parser.parse_args()

def load_tensorboard_data(log_dir):
    """
    Load training data from TensorBoard logs
    
    Args:
        log_dir: Directory containing TensorBoard logs
        
    Returns:
        Dictionary of metrics and their values
    """
    metrics = defaultdict(list)
    steps = defaultdict(list)
    
    # Find event files
    event_files = list(Path(log_dir).glob('events.out.tfevents.*'))
    
    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        return metrics, steps
    
    try:
        # Load events from each file
        for event_file in event_files:
            try:
                for e in tf.compat.v1.train.summary_iterator(str(event_file)):
                    for v in e.summary.value:
                        # Extract tag (metric name) and value
                        tag = v.tag
                        if hasattr(v, 'simple_value'):
                            val = v.simple_value
                            metrics[tag].append(val)
                            steps[tag].append(e.step)
            except Exception as inner_e:
                print(f"Error reading event file {event_file}: {inner_e}")
    except Exception as e:
        print(f"Error loading TensorBoard data: {e}")
    
    return metrics, steps

def visualize_training_history(metrics, steps, output_dir, show_plots=False):
    """
    Generate plots from training history
    
    Args:
        metrics: Dictionary of metrics
        steps: Dictionary of steps
        output_dir: Output directory for plots
        show_plots: Whether to show plots in addition to saving them
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a list of metrics to plot
    plot_groups = {
        'loss': ['loss', 'val_loss'],
        'accuracy': ['accuracy', 'val_accuracy'],
        'top_k_accuracy': ['top_2_accuracy', 'val_top_2_accuracy', 'top_3_accuracy', 'val_top_3_accuracy'],
        'learning_rate': ['learning_rate']
    }
    
    # Custom colors for different metrics
    colors = {
        'loss': 'tab:blue',
        'val_loss': 'tab:orange',
        'accuracy': 'tab:green',
        'val_accuracy': 'tab:red',
        'top_2_accuracy': 'tab:purple',
        'val_top_2_accuracy': 'tab:brown',
        'top_3_accuracy': 'tab:pink',
        'val_top_3_accuracy': 'tab:gray',
        'learning_rate': 'tab:cyan'
    }
    
    # Generate plots for each group
    for group_name, group_metrics in plot_groups.items():
        # Check if any of the metrics in this group exist
        if not any(m in metrics for m in group_metrics):
            continue
        
        plt.figure(figsize=(12, 8))
        
        for metric in group_metrics:
            if metric in metrics and len(metrics[metric]) > 0:
                # Get the data
                y = metrics[metric]
                x = steps[metric]
                
                # Apply smoothing for better visualization
                if len(y) > 10:
                    window_size = max(3, len(y) // 50)
                    y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
                    x_smooth = x[window_size-1:]
                    
                    # Plot both raw and smoothed data
                    plt.plot(x, y, alpha=0.3, color=colors.get(metric, 'tab:blue'))
                    plt.plot(x_smooth, y_smooth, label=metric, linewidth=2, color=colors.get(metric, 'tab:blue'))
                else:
                    plt.plot(x, y, label=metric, linewidth=2, color=colors.get(metric, 'tab:blue'))
        
        plt.xlabel('Epoch')
        plt.title(f'Training {group_name.replace("_", " ").title()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add y-axis label based on the group
        if 'loss' in group_name:
            plt.ylabel('Loss')
        elif 'accuracy' in group_name:
            plt.ylabel('Accuracy')
        elif 'learning_rate' in group_name:
            plt.ylabel('Learning Rate')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{group_name}.png'), dpi=300)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # Create a consolidated plot with accuracy and loss
    if ('accuracy' in metrics and 'val_accuracy' in metrics and 
        'loss' in metrics and 'val_loss' in metrics):
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Loss subplot
        ax1.plot(steps['loss'], metrics['loss'], label='Training Loss', color='tab:blue')
        ax1.plot(steps['val_loss'], metrics['val_loss'], label='Validation Loss', color='tab:orange')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Training and Validation Loss')
        
        # Accuracy subplot
        ax2.plot(steps['accuracy'], metrics['accuracy'], label='Training Accuracy', color='tab:green')
        ax2.plot(steps['val_accuracy'], metrics['val_accuracy'], label='Validation Accuracy', color='tab:red')
        
        # Add top-k accuracy if available
        if 'top_2_accuracy' in metrics and 'val_top_2_accuracy' in metrics:
            ax2.plot(steps['top_2_accuracy'], metrics['top_2_accuracy'], 
                    label='Training Top-2 Accuracy', linestyle='--', color='tab:purple')
            ax2.plot(steps['val_top_2_accuracy'], metrics['val_top_2_accuracy'], 
                    label='Validation Top-2 Accuracy', linestyle='--', color='tab:brown')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Training and Validation Accuracy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_metrics.png'), dpi=300)
        
        if show_plots:
            plt.show()
        else:
            plt.close()

def analyze_model(model_path, data_dir, output_dir, show_plots=False):
    """
    Analyze model performance on test data
    
    Args:
        model_path: Path to the model file
        data_dir: Path to the data directory
        output_dir: Output directory for plots
        show_plots: Whether to show plots in addition to saving them
    """
    # Custom objects for loading the model
    try:
        from inference_hamnosys import categorical_focal_loss
        custom_objects = {
            'categorical_focal_loss': categorical_focal_loss(),
            'top_2_accuracy': tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
            'top_3_accuracy': tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        }
    except ImportError:
        print("Warning: Could not import custom loss function. Using default objects.")
        custom_objects = {}
    
    try:
        # Load model
        print(f"Loading model from {model_path}...")
        model = load_model(model_path, custom_objects=custom_objects)
        
        # Print model summary
        model.summary()
        
        # Load test data
        print(f"Loading test data from {data_dir}...")
        
        # Load features and labels
        features_file = os.path.join(data_dir, 'features.npy')
        labels_file = os.path.join(data_dir, 'labels.npy')
        
        if not os.path.exists(features_file) or not os.path.exists(labels_file):
            print(f"Data files not found in {data_dir}. Skipping model analysis.")
            return
        
        features = np.load(features_file)
        labels = np.load(labels_file)
        
        print(f"Loaded data with shape: {features.shape}, {labels.shape}")
        
        # Split into test set
        num_samples = len(features)
        indices = np.random.permutation(num_samples)
        test_size = int(0.2 * num_samples)
        test_indices = indices[:test_size]
        
        X_test = features[test_indices]
        y_test = labels[test_indices]
        
        # Make predictions
        print("Running inference on test data...")
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred_classes == y_test)
        print(f"Test accuracy: {accuracy:.2%}")
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Load class names
        class_names = {}
        metadata_file = os.path.join(data_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            if 'classes' in metadata:
                class_names = metadata['classes']
        
        if not class_names:
            mapping_file = os.path.join(data_dir, 'label_mapping.json')
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    class_names = json.load(f)
        
        # Convert string keys to integers if needed
        if any(isinstance(k, str) for k in class_names.keys()):
            class_names = {int(k): v for k, v in class_names.items()}
        
        # Get class names for plotting
        class_list = [class_names.get(i, f"Class {i}") for i in range(len(cm))]
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='viridis',
                   xticklabels=class_list, yticklabels=class_list)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Normalized Confusion Matrix (Accuracy: {accuracy:.2%})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_confusion_matrix.png'), dpi=300)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Save model analysis results
        analysis_results = {
            'test_accuracy': float(accuracy),
            'test_size': int(test_size),
            'model_file': model_path
        }
        
        with open(os.path.join(output_dir, 'model_analysis.json'), 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print(f"Model analysis completed and saved to {output_dir}")
        
    except Exception as e:
        print(f"Error during model analysis: {e}")

def main():
    args = parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, 'visualizations')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find TensorBoard log directory
    log_dir = os.path.join(args.model_dir, 'logs')
    if not os.path.exists(log_dir):
        print(f"TensorBoard log directory not found at {log_dir}")
        # Try to find any logs directory
        for dirpath, dirnames, _ in os.walk(args.model_dir):
            if 'logs' in dirnames:
                log_dir = os.path.join(dirpath, 'logs')
                print(f"Found alternative log directory: {log_dir}")
                break
    
    # Load and visualize training history
    if os.path.exists(log_dir):
        print(f"Loading TensorBoard data from {log_dir}...")
        metrics, steps = load_tensorboard_data(log_dir)
        
        if metrics:
            print("Generating training history visualizations...")
            visualize_training_history(metrics, steps, args.output_dir, args.show_plots)
        else:
            print("No metrics found in TensorBoard logs")
    
    # Find and analyze saved model
    if args.include_model_analysis:
        # Find best model
        model_path = None
        for model_file in ['model_best.keras', 'model_best.h5']:
            if os.path.exists(os.path.join(args.model_dir, model_file)):
                model_path = os.path.join(args.model_dir, model_file)
                break
        
        if model_path:
            print(f"Found model file: {model_path}")
            analyze_model(model_path, args.data_dir, args.output_dir, args.show_plots)
        else:
            print("No model file found in the model directory")
    
    print(f"Visualizations saved to {args.output_dir}")

if __name__ == '__main__':
    main() 