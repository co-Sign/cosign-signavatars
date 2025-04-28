import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm

class ASLTranslationLSTM(nn.Module):
    """
    LSTM-based model for ASL translation.
    This model translates sequences of SMPL-X parameters to ASL gloss.
    Enhanced with stronger regularization and residual connections.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3, bidirectional=True,
                 weight_decay=1e-4, batch_norm=True):
        """
        Initialize the LSTM model.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            num_layers (int): Number of LSTM layers
            output_dim (int): Output dimension (usually the vocabulary size)
            dropout (float): Dropout probability
            bidirectional (bool): Whether to use bidirectional LSTM
            weight_decay (float): L2 regularization strength
            batch_norm (bool): Whether to use batch normalization
        """
        super(ASLTranslationLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        self.weight_decay = weight_decay
        self.batch_norm = batch_norm
        
        # Input projection with batch normalization
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        if self.batch_norm:
            self.input_bn = nn.BatchNorm1d(hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism with scaled dot-product attention
        self.attention_query = nn.Linear(hidden_dim * self.directions, hidden_dim)
        self.attention_key = nn.Linear(hidden_dim * self.directions, hidden_dim)
        self.attention_value = nn.Linear(hidden_dim * self.directions, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))
        
        # Output layers with residual connection
        hidden_out_dim = hidden_dim * self.directions
        self.fc1 = nn.Linear(hidden_out_dim, hidden_out_dim)
        if self.batch_norm:
            self.fc1_bn = nn.BatchNorm1d(hidden_out_dim)
        self.fc2 = nn.Linear(hidden_out_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Apply L2 regularization through weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with L2 regularization in mind"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim)
            mask (torch.Tensor): Mask tensor of shape (batch_size, seq_length) where 1 indicates valid positions
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        batch_size, seq_length, _ = x.shape
        
        # Input projection
        x = self.input_proj(x.view(batch_size * seq_length, -1))
        if self.batch_norm:
            x = self.input_bn(x)
        x = torch.relu(x)
        x = x.view(batch_size, seq_length, -1)
        x = self.dropout(x)
        
        # Scale inputs to prevent exploding gradients
        device = x.device
        if not hasattr(self, 'scale') or self.scale.device != device:
            self.scale = torch.sqrt(torch.FloatTensor([self.hidden_dim])).to(device)
        
        # LSTM output
        lstm_out, (hidden, _) = self.lstm(x)  # lstm_out: (batch_size, seq_length, hidden_dim * directions)
        
        # Apply scaled dot-product attention
        Q = self.attention_query(lstm_out)  # (batch_size, seq_length, hidden_dim)
        K = self.attention_key(lstm_out)    # (batch_size, seq_length, hidden_dim)
        V = self.attention_value(lstm_out)  # (batch_size, seq_length, hidden_dim)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(1, 2)) / self.scale  # (batch_size, seq_length, seq_length)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size, 1, seq_length)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch_size, seq_length, seq_length)
        
        # Apply attention weights to get context
        context = torch.matmul(attention_weights, V)  # (batch_size, seq_length, hidden_dim)
        
        # Get sequence representation (use the last token's context)
        sequence_rep = context[:, -1, :]  # (batch_size, hidden_dim)
        if self.bidirectional:
            # Concatenate forward and backward representations
            sequence_rep = torch.cat([sequence_rep[:, :self.hidden_dim], sequence_rep[:, self.hidden_dim:]], dim=1)
        
        # Apply dropout
        sequence_rep = self.dropout(sequence_rep)
        
        # First output layer
        output1 = self.fc1(sequence_rep)
        if self.batch_norm:
            output1 = self.fc1_bn(output1)
        output1 = torch.relu(output1)
        output1 = self.dropout(output1)
        
        # Residual connection
        output1 = output1 + sequence_rep
        
        # Final output layer
        output = self.fc2(output1)
        
        return output

class ASLSequenceClassifier(nn.Module):
    """
    LSTM-based model for classifying ASL sequences into gloss labels.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.3, bidirectional=True,
                 weight_decay=1e-4, batch_norm=True):
        """
        Initialize the classifier.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            num_layers (int): Number of LSTM layers
            num_classes (int): Number of output classes
            dropout (float): Dropout probability
            bidirectional (bool): Whether to use bidirectional LSTM
            weight_decay (float): L2 regularization strength
            batch_norm (bool): Whether to use batch normalization
        """
        super(ASLSequenceClassifier, self).__init__()
        
        self.lstm_model = ASLTranslationLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=num_classes,
            dropout=dropout,
            bidirectional=bidirectional,
            weight_decay=weight_decay,
            batch_norm=batch_norm
        )
    
    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim)
            mask (torch.Tensor): Mask tensor of shape (batch_size, seq_length)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        return self.lstm_model(x, mask)

class Trainer:
    """
    Trainer class for the ASL translation model.
    """
    def __init__(self, model, device, learning_rate=0.001, weight_decay=1e-4, scheduler_type='cosine'):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): The model to train
            device (torch.device): The device to use for training
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay for regularization
            scheduler_type (str): Type of learning rate scheduler ('cosine', 'step', or None)
        """
        self.model = model
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler_type = scheduler_type
        self.scheduler = None
        
        # Move model to device
        self.model.to(self.device)
    
    def train(self, train_loader, val_loader=None, epochs=100, patience=10, checkpoint_dir='checkpoints'):
        """
        Train the model.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            epochs (int): Number of epochs
            patience (int): Early stopping patience
            checkpoint_dir (str): Directory to save checkpoints
            
        Returns:
            dict: Training history
        """
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Initialize learning rate scheduler
        if self.scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-6
            )
        elif self.scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.5
            )
        
        # Initialize variables
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch in train_pbar:
                # Get batch data
                inputs = batch['smplx_params'].to(self.device)
                masks = batch['mask'].to(self.device)
                labels = batch['label_id'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs, masks)
                loss = self.criterion(outputs, labels)
                
                # Add L2 regularization for additional parameters
                l2_reg = 0.0
                for param in self.model.parameters():
                    l2_reg += torch.norm(param, 2)
                loss += self.model.lstm_model.weight_decay * l2_reg
                
                # Backward pass and optimize
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Update metrics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': train_loss / (train_total if train_total > 0 else 1),
                    'acc': 100 * train_correct / (train_total if train_total > 0 else 1)
                })
            
            # Calculate training metrics
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = 100 * train_correct / len(train_loader.dataset)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                
                # Update history
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['lr'].append(self.optimizer.param_groups[0]['lr'])
                
                # Print epoch results
                print(f'Epoch {epoch+1}/{epochs} - '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - '
                      f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
                
                # Check for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': val_loss,
                    }, os.path.join(checkpoint_dir, 'best_model.pth'))
                else:
                    patience_counter += 1
                    
                # Save checkpoint
                if (epoch + 1) % 10 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': val_loss if val_loader is not None else train_loss,
                    }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
                
                # Early stopping
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}. Best epoch: {best_epoch+1}')
                    break
            else:
                # Update history
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['lr'].append(self.optimizer.param_groups[0]['lr'])
                
                # Print epoch results
                print(f'Epoch {epoch+1}/{epochs} - '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                      f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
                
                # Save checkpoint
                if (epoch + 1) % 10 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': train_loss,
                    }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        return history
    
    def evaluate(self, data_loader):
        """
        Evaluate the model.
        
        Args:
            data_loader (DataLoader): Data loader for evaluation
            
        Returns:
            tuple: Loss and accuracy
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(data_loader, desc='Evaluation')
            for batch in val_pbar:
                # Get batch data
                inputs = batch['smplx_params'].to(self.device)
                masks = batch['mask'].to(self.device)
                labels = batch['label_id'].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs, masks)
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': val_loss / (val_total if val_total > 0 else 1),
                    'acc': 100 * val_correct / (val_total if val_total > 0 else 1)
                })
        
        val_loss = val_loss / len(data_loader.dataset)
        val_acc = 100 * val_correct / len(data_loader.dataset)
        
        return val_loss, val_acc

def load_model(model_path, model_params, device):
    """
    Load a trained model from a checkpoint.
    
    Args:
        model_path (str): Path to the model checkpoint
        model_params (dict): Model parameters
        device (torch.device): Device to load the model on
        
    Returns:
        nn.Module: Loaded model
    """
    # Create model
    model = ASLSequenceClassifier(**model_params)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    return model

def predict(model, inputs, idx_to_label, device):
    """
    Make predictions with the model.
    
    Args:
        model (nn.Module): The trained model
        inputs (numpy.ndarray): Input features
        idx_to_label (dict): Mapping from index to label
        device (torch.device): Device to use for prediction
        
    Returns:
        str: Predicted label
    """
    # Convert inputs to tensor
    inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Create mask for valid positions
    mask = torch.ones((1, inputs.shape[1]), dtype=torch.float32).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(inputs, mask)
        _, predicted = torch.max(outputs, 1)
        
    # Convert to label
    label = idx_to_label[predicted.item()]
    
    return label

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or evaluate the ASL translation model")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Mode: train or eval")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the preprocessed data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save/load checkpoints")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization")
    parser.add_argument("--bidirectional", type=bool, default=True, help="Whether to use bidirectional LSTM")
    parser.add_argument("--batch_norm", type=bool, default=True, help="Whether to use batch normalization")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"], help="LR scheduler type")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    # Implementation depends on your data preprocessing 