# Automated Training for HamNoSys Recognition

This document outlines the automated training process for the HamNoSys sign language recognition model.

## Overview

The automated training system handles:
- Processing data in batches to avoid overwhelming the CPU
- Detecting early stopping and automatically restarting training
- Consolidating model files (keeping only best and latest)
- Moving to inference when accuracy threshold is reached
- Visualizing results and saving comprehensive logs

## Quick Start

### Start Training

To begin the automated training process, simply run:

```bash
# Windows
start_training.bat

# Linux/Mac
python automated_training.py
```

This will start the training process which automatically:
1. Prepares batches of data (15 classes at a time)
2. Trains the model with early stopping monitoring
3. Restarts with new batches when training stalls
4. Saves consolidated best model

### Run Inference

After training (or during breaks), you can run inference using:

```bash
# Windows
run_inference.bat

# Linux/Mac
python inference_hamnosys.py --model-path models/hamnosys/main_model/best_model.keras --visualize
```

## Training Process Details

### Data Processing

Data is processed in smaller batches of classes to prevent memory issues:
- Each batch contains at most 15 classes (configurable)
- Classes are selected based on an offset to avoid overlap
- Data augmentation increases training examples
- Class balancing ensures balanced learning

### Training Automation

The training automation handles:
1. Early stopping detection
2. Model consolidation (only keeping best and latest)
3. Accuracy monitoring (moving to inference at 90%+)
4. Comprehensive logging and visualization

### Command Line Arguments

```
python automated_training.py [options]

Options:
  --epochs EPOCHS           Maximum epochs per training run (default: 500)
  --patience PATIENCE       Patience for early stopping (default: 30)
  --lr LR                   Learning rate (default: 0.0005)
  --include-datasets LIST   Datasets to include in training
  --log-file LOG_FILE       File to save training logs (default: training_log.txt)
  --no-restart              Do not restart after early stopping
  --run-inference           Run inference after training completes
  --gpu                     Use GPU if available
  --augment-factor FACTOR   Data augmentation factor (default: 15)
```

## Model Architecture

The model uses a mixed architecture with:
- Attention mechanism with 12 heads
- Residual connections
- Normalization
- Focal loss for handling class imbalance

## Directory Structure

```
models/hamnosys/main_model/
├── best_model.keras          # Best model based on validation accuracy
├── latest_model.keras        # Latest model from training
├── logs/                     # Training logs for each batch
├── visualizations/           # Visualization results
└── training_summary.json     # Summary of training runs and metrics
```

## Troubleshooting

If you encounter issues:

1. **Memory errors**: Try reducing batch size or max_classes_per_batch in automated_training.py
2. **Training stalls**: Check logs for patterns in early stopping, try different learning rates
3. **Model not improving**: Try increasing augmentation factor or including more datasets

## Performance Tracking

Training progress is tracked in:
- training_log.txt (main log file with batch summaries)
- models/hamnosys/main_model/training_summary.json (detailed metrics)
- models/hamnosys/main_model/logs/ (individual batch logs)

Current model achieves:
- Top-1 accuracy: ~31%
- Top-2 accuracy: ~60% 
- Top-3 accuracy: ~74%

The goal is to improve top-1 accuracy to at least 90%. 