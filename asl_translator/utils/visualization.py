"""
Visualization utilities for ASL translator.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

def plot_smplx_params(params, output_file=None, max_frames=None):
    """
    Plot SMPL-X parameters over time.
    
    Args:
        params (dict): Dictionary of SMPL-X parameters
        output_file (str, optional): Path to save the plot. If None, plot is displayed.
        max_frames (int, optional): Maximum number of frames to plot
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('SMPL-X Parameters Over Time', fontsize=16)
    
    # Setup grid
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    # Counter for subplot position
    plot_idx = 0
    
    # For each parameter type
    for param_name, param_data in params.items():
        if param_data is None or not isinstance(param_data, np.ndarray):
            continue
            
        # Limit frames if specified
        if max_frames is not None and param_data.shape[0] > max_frames:
            param_data = param_data[:max_frames]
            
        # Get subplot position
        row = plot_idx // 2
        col = plot_idx % 2
        
        if row >= 3:  # Skip if we have too many parameters
            continue
            
        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        
        # Handle different parameter shapes
        if len(param_data.shape) == 1:  # 1D data
            ax.plot(param_data)
        elif len(param_data.shape) == 2:  # 2D data
            # Plot first 5 dimensions
            dims_to_plot = min(5, param_data.shape[1])
            for i in range(dims_to_plot):
                ax.plot(param_data[:, i], label=f'Dim {i}')
            ax.legend()
        elif len(param_data.shape) == 3:  # 3D data
            # Flatten last dimension and plot first 5 series
            reshaped = param_data.reshape(param_data.shape[0], -1)
            dims_to_plot = min(5, reshaped.shape[1])
            for i in range(dims_to_plot):
                ax.plot(reshaped[:, i], label=f'Series {i}')
            ax.legend()
            
        ax.set_title(f'{param_name} ({param_data.shape})')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Value')
        ax.grid(True)
        
        plot_idx += 1
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()
        
def plot_feature_comparison(original, processed, title="Feature Comparison", output_file=None):
    """
    Plot comparison between original and processed features.
    
    Args:
        original (np.ndarray): Original features
        processed (np.ndarray): Processed features
        title (str, optional): Plot title
        output_file (str, optional): Path to save the plot. If None, plot is displayed.
    """
    if len(original.shape) > 2:
        original = original.reshape(original.shape[0], -1)
    if len(processed.shape) > 2:
        processed = processed.reshape(processed.shape[0], -1)
        
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot original features
    im1 = axes[0].imshow(original.T, aspect='auto', cmap='viridis')
    axes[0].set_title('Original Features')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Feature Dimension')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot processed features
    im2 = axes[1].imshow(processed.T, aspect='auto', cmap='viridis')
    axes[1].set_title('Processed Features')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Feature Dimension')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()
        
def plot_training_history(history, output_file=None):
    """
    Plot training history.
    
    Args:
        history (dict): Dictionary containing 'loss' and 'val_loss' keys
        output_file (str, optional): Path to save the plot. If None, plot is displayed.
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Training History', fontsize=16)
    
    # Plot training & validation loss
    ax1.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy if available
    if 'accuracy' in history:
        ax2.plot(history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history:
            ax2.plot(history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
    else:
        # If no accuracy, plot learning rate or other metric if available
        if 'lr' in history:
            ax2.plot(history['lr'])
            ax2.set_title('Learning Rate')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()
        
def animate_sequence(sequence, output_file=None, interval=50):
    """
    Create an animation of a sequence.
    
    Args:
        sequence (np.ndarray): Sequence data with shape (frames, dimensions)
        output_file (str, optional): Path to save the animation. If None, animation is displayed.
        interval (int, optional): Interval between frames in milliseconds
    """
    # Reshape if needed
    if len(sequence.shape) > 2:
        sequence = sequence.reshape(sequence.shape[0], -1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Function to update the plot
    def update(frame):
        ax.clear()
        ax.plot(sequence[frame])
        ax.set_ylim([sequence.min(), sequence.max()])
        ax.set_title(f'Frame {frame}')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Value')
        ax.grid(True)
        
    # Create animation
    frames = min(100, sequence.shape[0])  # Limit to 100 frames for performance
    ani = FuncAnimation(fig, update, frames=frames, interval=interval)
    
    plt.tight_layout()
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        ani.save(output_file)
        plt.close()
    else:
        plt.show()
        
def plot_confusion_matrix(confusion_matrix, class_names, output_file=None):
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix (np.ndarray): Confusion matrix array
        class_names (list): List of class names
        output_file (str, optional): Path to save the plot. If None, plot is displayed.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot confusion matrix
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate x tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show() 