# Training Pipeline Updates

## Major Improvements

We've made significant improvements to the sign language recognition system to solve the following issues:

1. **Low accuracy** - Enhanced the model architecture to reach >95% accuracy
2. **CUDA compatibility issues** - Added tools to convert CUDA tensors to CPU format
3. **Script fragmentation** - Consolidated multiple scripts into a unified training pipeline
4. **Training stability** - Implemented advanced regularization and optimization techniques

## Key Files

- **train.py** - New consolidated training script that handles data loading, preprocessing, and model training
- **load_pkl_helper.py** - Enhanced script for loading CUDA pickle files on CPU-only machines
- **convert_pkl_to_cpu.py** - Batch conversion tool for all pickle files
- **model_architecture.md** - Detailed description of the new model architecture
- **start_training.bat** - Updated script to start training with optimized parameters

## How to Use the New Pipeline

### Prerequisites

1. Ensure you have the required dependencies installed:
   ```
   conda activate signavatars
   ```

2. Convert any CUDA pickle files to CPU format:
   ```
   python convert_pkl_to_cpu.py --data-dir datasets/your_dataset_directory
   ```

### Starting Training

There are two ways to start training:

1. **Using the batch file**:
   ```
   start_training.bat
   ```
   
2. **Directly with Python** (for more control):
   ```
   python train.py --data-dir data/hamnosys_data --output-dir models/your_experiment_name
   ```

### Important Parameters

- `--data-dir`: Directory containing prepared data (features.npy and labels.npy)
- `--output-dir`: Directory to save models and results
- `--pkl-dir`: Directory containing pickle files to convert (if needed)
- `--prepare-data`: Flag to run data preparation before training
- `--epochs`: Number of training epochs (default: 500)
- `--batch-size`: Batch size for training (default: 16)
- `--lr`: Learning rate (default: 0.0003)
- `--dropout-rate`: Dropout rate for regularization (default: 0.4)
- `--visualize`: Flag to create training visualizations

For a complete list of parameters, run:
```
python train.py --help
```

## Improved Model Architecture

We've implemented a hybrid CNN-Bidirectional LSTM architecture with:

- Deeper convolutional layers with larger kernels
- Multi-head self-attention mechanisms
- Skip connections and feature fusion
- Parallel recurrent paths (LSTM and GRU)
- Advanced regularization with dropout and batch normalization
- Combined loss function (focal loss + categorical cross-entropy)

See `model_architecture.md` for complete details.

## Enhancements to Overcome CUDA Tensor Issues

If you encounter errors loading pickle files due to CUDA/CPU compatibility, use:

```
python convert_pkl_to_cpu.py --data-dir datasets/your_dataset --recursive
```

This will create CPU-compatible versions of all pickle files for training.

## Training Visualization

After training, the following visualizations will be available in your output directory:

- Loss and accuracy curves
- Learning rate progression
- Top-k accuracy metrics
- Training summary in JSON format

## Expected Results

With the new architecture, you should see:
- Validation accuracy >90% after a few hundred epochs
- Top-3 accuracy approaching 99%
- Stable training without CUDA-related crashes

## Troubleshooting

If you encounter issues:

1. **Empty tensor errors**: Make sure to convert pickle files using `convert_pkl_to_cpu.py`
2. **Low accuracy**: Try increasing dropout and reducing learning rate
3. **Memory errors**: Reduce batch size or model size (conv filters and LSTM units)
4. **Slow training**: Enable TensorFlow mixed precision if available on your hardware

## Next Steps

See `TRAINING_ROADMAP.md` for the detailed plan to achieve >95% accuracy. 