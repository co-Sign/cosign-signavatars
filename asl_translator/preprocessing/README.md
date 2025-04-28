# ASL Translation Preprocessing Tools

This directory contains tools for preprocessing ASL motion data, specifically SMPL-X parameters, for use in ASL translation models.

## Overview

The preprocessing pipeline converts raw SMPL-X parameters into a format suitable for training ASL translation models. The pipeline:

1. Loads raw data (either PKL files with full SMPL-X parameters or NPY files with motion data)
2. Extracts relevant features (body pose, hand poses, and jaw pose)
3. Saves processed features as NPY files along with metadata

## Key Components

- `data_processor.py`: Main preprocessing module with functions to extract features and process datasets
- `verify_preprocessing.py`: Tool to verify preprocessing pipeline by comparing raw and processed data
- `preprocess_and_verify.py`: End-to-end script that runs preprocessing and verification in one step

## Data Format

### Input Data

The preprocessing pipeline supports two types of input data:

1. **PKL files**: SMPL-X parameter files with the following structure:
   - `smplx`: SMPL-X parameters array with shape (seq_length, 182)
   - Parameter breakdown:
     - root_pose: 0:3 (3 dimensions)
     - body_pose: 3:66 (63 dimensions)
     - left_hand_pose: 66:111 (45 dimensions)
     - right_hand_pose: 111:156 (45 dimensions)
     - jaw_pose: 156:159 (3 dimensions)
     - shape: 159:169 (10 dimensions)
     - expression: 169:179 (10 dimensions)
     - cam_trans: 179:182 (3 dimensions)

2. **NPY files**: NumPy arrays containing SMPL-X parameters with shape (seq_length, 182)
   - Same parameter breakdown as PKL files
   - Optional `_metadata.json` files with:
     - `file_name`: Base name of the file
     - `label`: Class label (e.g., ASL gloss)
     - `seq_length`: Number of frames

### Output Data

The preprocessing pipeline produces:

1. **NPY files**: Containing extracted features with shape (seq_length, 156)
   - Features include body pose (63), left hand pose (45), right hand pose (45), and jaw pose (3)

2. **Metadata JSON files**: Containing:
   - `file_name`: Base name of the file
   - `label`: Class label
   - `seq_length`: Number of frames

## Usage

### Basic Preprocessing

To preprocess a dataset without verification:

```bash
python -m preprocessing.data_processor --data_dir /path/to/raw/data --output_dir /path/to/output
```

Optional arguments:
- `--max_seq_length`: Maximum sequence length (default: 128)

### Preprocessing with Verification

Run preprocessing and verify the results in one step:

```bash
python preprocess_and_verify.py --raw_dir /path/to/raw/data --processed_dir /path/to/output --verification_dir /path/to/verification/results
```

Optional arguments:
- `--num_samples`: Number of samples to verify (default: 5)

### Verification Only

To verify existing preprocessed data:

```bash
python verify_preprocessing.py --raw_dir /path/to/raw/data --processed_dir /path/to/preprocessed/data --output_dir /path/to/verification/results
```

Optional arguments:
- `--max_samples`: Maximum number of samples to verify (default: 5)

## Verification Results

The verification process produces:

1. **Comparison Plots**: Visual comparisons between raw and processed features
2. **Verification Report**: JSON file with verification results including:
   - Summary of pass/fail status
   - Details for each verified sample
   - Processing time and other metrics

## Creating Synthetic Test Data

For testing purposes, you can generate synthetic SMPL-X data:

```bash
python create_test_data.py --output_dir /path/to/output --num_frames 100 --num_samples 10
```

This creates synthetic motion data with realistic patterns that can be used to test the preprocessing pipeline.

## Integration with Training Pipeline

The preprocessing tools are designed to be integrated with the full ASL translation pipeline. To run the complete pipeline from preprocessing to training:

```bash
python run_pipeline.py --raw_data_dir /path/to/raw/data --output_dir /path/to/output
```

This will:
1. Preprocess the raw data
2. Train an ASL translation model
3. Generate a test script for inference 