import os
import argparse
import torch
import json
import numpy as np
from datetime import datetime
from models.lstm_model import ASLSequenceClassifier, Trainer
from models.dataset import get_data_loaders

def save_config(args, config_path):
    """
    Save the training configuration.
    
    Args:
        args: Command line arguments
        config_path (str): Path to save the configuration
    """
    config = vars(args)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def main():
    """
    Main function to train the ASL translation model.
    """
    parser = argparse.ArgumentParser(description="Train ASL translation model")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing preprocessed data")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test split ratio")
    
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--bidirectional", type=bool, default=True, help="Whether to use bidirectional LSTM")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for regularization")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Save configuration
    config_path = os.path.join(output_dir, "config.json")
    save_config(args, config_path)
    
    # Load data
    print(f"Loading data from {args.data_dir}")
    train_loader, val_loader, test_loader, label_to_idx, idx_to_label = get_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed
    )
    
    # Print dataset information
    num_classes = len(label_to_idx)
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Get input dimension from first batch
    input_dim = None
    for batch in train_loader:
        input_dim = batch['smplx_params'].shape[2]
        break
    
    # Check if we have a valid input_dim
    if input_dim is None:
        raise ValueError("Could not determine input dimension from training data. The dataset may be empty.")
    
    # Save label mappings
    label_mappings = {
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label
    }
    with open(os.path.join(output_dir, "label_mappings.json"), 'w') as f:
        json.dump(label_mappings, f, indent=2)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ASLSequenceClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout,
        bidirectional=args.bidirectional
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience,
        checkpoint_dir=checkpoint_dir
    )
    
    # Save training history
    with open(os.path.join(output_dir, "history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # Save test results
    test_results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc
    }
    with open(os.path.join(output_dir, "test_results.json"), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"Training completed! Results saved to {output_dir}")

if __name__ == "__main__":
    main() 