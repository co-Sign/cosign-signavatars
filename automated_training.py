#!/usr/bin/env python3
"""
Automated Training Script for HamNoSys Recognition Model

This script handles continuous training with automatic restart after early stopping,
processes data in batches to avoid overwhelming the CPU, and progresses to inference
when desired accuracy is reached.
"""
import os
import sys
import json
import time
import subprocess
import argparse
import glob
import re
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path

# Constants
MAIN_MODEL_DIR = 'models/hamnosys/main_model'
BATCH_SIZE = 16
MAX_CLASSES_PER_BATCH = 15
ACCURACY_THRESHOLD = 90.0  # When to move to inference
MAX_ATTEMPTS = 20  # Maximum training attempts
EARLY_STOPPING_PATTERN = r"Epoch \d+: early stopping"
BATCH_TRAINING_COMPLETED_PATTERN = r"Training completed"
ACCURACY_PATTERN = r"Test accuracy \(%\): (\d+\.\d+)%"
TOP2_ACCURACY_PATTERN = r"Test top-2 accuracy: (\d+\.\d+)"
TOP3_ACCURACY_PATTERN = r"Test top-3 accuracy: (\d+\.\d+)"

def parse_args():
    parser = argparse.ArgumentParser(description='Automated Training for HamNoSys Recognition')
    parser.add_argument('--epochs', type=int, default=500,
                      help='Maximum epochs per training run')
    parser.add_argument('--patience', type=int, default=30,
                      help='Patience for early stopping')
    parser.add_argument('--lr', type=float, default=0.0005,
                      help='Learning rate')
    parser.add_argument('--include-datasets', nargs='+', default=['hamnosys2motion', 'wlasl_pkls_cropFalse_defult_shape'],
                      help='Datasets to include in training')
    parser.add_argument('--log-file', type=str, default='training_log.txt',
                      help='File to save training logs')
    parser.add_argument('--no-restart', action='store_true',
                      help='Do not automatically restart training after early stopping')
    parser.add_argument('--run-inference', action='store_true', default=True,
                      help='Run inference after training completes or reaches threshold')
    parser.add_argument('--gpu', action='store_true',
                      help='Use GPU if available')
    parser.add_argument('--augment-factor', type=int, default=15,
                      help='Data augmentation factor')
    
    return parser.parse_args()

def setup_model_directory():
    """Create the main model directory if it doesn't exist"""
    os.makedirs(MAIN_MODEL_DIR, exist_ok=True)
    
    # Create subdirectories for logs and visualizations
    os.makedirs(os.path.join(MAIN_MODEL_DIR, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(MAIN_MODEL_DIR, 'visualizations'), exist_ok=True)
    
    # Create a summary file if it doesn't exist
    summary_file = os.path.join(MAIN_MODEL_DIR, 'training_summary.json')
    if not os.path.exists(summary_file):
        summary_data = {
            'training_runs': [],
            'best_accuracy': 0.0,
            'best_top2_accuracy': 0.0,
            'best_top3_accuracy': 0.0,
            'last_batch': -1,
            'total_training_time': 0.0
        }
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
    
    return summary_file

def get_next_batch(summary_file):
    """Determine the next batch number to process"""
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    return summary['last_batch'] + 1

def prepare_data_batch(batch_num, args):
    """
    Prepare a batch of data for training by selecting a subset of classes
    
    Args:
        batch_num: Batch number
        args: Command line arguments
        
    Returns:
        Command to run data preparation
    """
    # Define which datasets to include for this batch
    datasets = args.include_datasets
    
    # Calculate offset for class selection to avoid overlap
    offset = batch_num * MAX_CLASSES_PER_BATCH
    
    # Base preparation command
    cmd = [
        "python", "prepare_hamnosys_data.py",
        "--data-file", "datasets/hamnosys2motion/data.json",
        "--output-dir", "data/hamnosys_data",
        "--min-length", "5",
        "--min-samples", "6",
        "--max-classes", str(MAX_CLASSES_PER_BATCH),
        "--offset", str(offset),  # Add class offset for batch
        "--augment",
        "--augment-factor", str(args.augment_factor),
        "--balance"
    ]
    
    # Add additional datasets if specified
    if 'wlasl_pkls_cropFalse_defult_shape' in datasets:
        cmd.extend(["--include-wlasl", "--wlasl-dir", "datasets/wlasl_pkls_cropFalse_defult_shape"])
    
    if 'phonex_pkls_cropFalse_shapeFalse' in datasets:
        cmd.extend(["--include-phonex", "--phonex-dir", "datasets/phonex_pkls_cropFalse_shapeFalse"])
    
    if 'how2sign_pkls_cropTrue_shapeTrue' in datasets:
        cmd.extend(["--include-how2sign", "--how2sign-dir", "datasets/how2sign_pkls_cropTrue_shapeTrue"])
    
    return cmd

def run_training(batch_num, args, summary_file):
    """
    Run a training session for the current batch
    
    Args:
        batch_num: Current batch number
        args: Command line arguments
        summary_file: Path to the summary file
        
    Returns:
        Training result data
    """
    # Update summary file with current batch
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    summary['last_batch'] = batch_num
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Prepare data for this batch
    data_cmd = prepare_data_batch(batch_num, args)
    print(f"Preparing data for batch {batch_num}...")
    data_process = subprocess.run(data_cmd, capture_output=True, text=True)
    
    if data_process.returncode != 0:
        print(f"Error preparing data: {data_process.stderr}")
        return None
    
    # Build training command
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(MAIN_MODEL_DIR, 'logs', f'training_batch_{batch_num}_{timestamp}.log')
    
    train_cmd = [
        "python", "train_hamnosys.py",
        "--epochs", str(args.epochs),
        "--output-dir", MAIN_MODEL_DIR,
        "--batch-size", str(BATCH_SIZE),
        "--lr", str(args.lr),
        "--use-attention",
        "--attention-heads", "12",
        "--use-residual",
        "--use-mixed-architecture",
        "--use-class-weights",
        "--normalize-features",
        "--augment",
        "--augment-factor", str(args.augment_factor)
    ]
    
    # Begin training
    print(f"Starting training for batch {batch_num}...")
    start_time = time.time()
    
    # Open log file for writing
    with open(log_file, 'w') as log_f:
        train_process = subprocess.Popen(
            train_cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Variables to track progress
        early_stopping_detected = False
        batch_completed = False
        accuracy = 0.0
        top2_accuracy = 0.0
        top3_accuracy = 0.0
        epoch_counter = 0
        
        # Process output line by line
        for line in train_process.stdout:
            log_f.write(line)
            log_f.flush()
            
            # Extract current epoch
            epoch_match = re.search(r"Epoch (\d+)/", line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                if current_epoch > epoch_counter:
                    epoch_counter = current_epoch
                    # Print simplified progress update
                    print(f"Batch {batch_num}: Epoch {epoch_counter}/{args.epochs}", end='\r')
            
            # Check for early stopping
            if re.search(EARLY_STOPPING_PATTERN, line):
                early_stopping_detected = True
                print(f"\nEarly stopping detected at epoch {epoch_counter}")
            
            # Check for completion
            if re.search(BATCH_TRAINING_COMPLETED_PATTERN, line):
                batch_completed = True
            
            # Extract accuracy metrics
            acc_match = re.search(ACCURACY_PATTERN, line)
            if acc_match:
                accuracy = float(acc_match.group(1))
            
            top2_match = re.search(TOP2_ACCURACY_PATTERN, line)
            if top2_match:
                top2_accuracy = float(top2_match.group(1))
            
            top3_match = re.search(TOP3_ACCURACY_PATTERN, line)
            if top3_match:
                top3_accuracy = float(top3_match.group(1))
        
        # Wait for process to complete
        train_process.wait()
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Print final results
    print(f"\nBatch {batch_num} completed:")
    print(f"  - Training time: {training_time:.2f} seconds")
    print(f"  - Final accuracy: {accuracy:.2f}%")
    print(f"  - Top-2 accuracy: {top2_accuracy:.2f}")
    print(f"  - Top-3 accuracy: {top3_accuracy:.2f}")
    
    # Update summary with results
    result_data = {
        'batch': batch_num,
        'timestamp': timestamp,
        'epochs_completed': epoch_counter,
        'early_stopping': early_stopping_detected,
        'training_time': training_time,
        'accuracy': accuracy,
        'top2_accuracy': top2_accuracy,
        'top3_accuracy': top3_accuracy,
        'log_file': log_file
    }
    
    # Update summary file
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    summary['training_runs'].append(result_data)
    summary['total_training_time'] += training_time
    
    # Update best metrics
    if accuracy > summary['best_accuracy']:
        summary['best_accuracy'] = accuracy
    if top2_accuracy > summary['best_top2_accuracy']:
        summary['best_top2_accuracy'] = top2_accuracy
    if top3_accuracy > summary['best_top3_accuracy']:
        summary['best_top3_accuracy'] = top3_accuracy
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return result_data

def consolidate_models():
    """
    Ensure we only keep the best and latest model files by copying/renaming them
    """
    # Find best and latest model files
    best_models = glob.glob(os.path.join(MAIN_MODEL_DIR, '**/model_best.keras'), recursive=True)
    latest_models = glob.glob(os.path.join(MAIN_MODEL_DIR, '**/model_latest.keras'), recursive=True)
    
    if best_models:
        # Find most recent best model by file timestamp
        best_model = max(best_models, key=os.path.getmtime)
        # Copy to main directory
        shutil.copy2(best_model, os.path.join(MAIN_MODEL_DIR, 'best_model.keras'))
    
    if latest_models:
        # Find most recent latest model by file timestamp
        latest_model = max(latest_models, key=os.path.getmtime)
        # Copy to main directory
        shutil.copy2(latest_model, os.path.join(MAIN_MODEL_DIR, 'latest_model.keras'))

def run_inference():
    """
    Run inference and visualization after training completes
    """
    # Ensure the best model exists
    best_model_path = os.path.join(MAIN_MODEL_DIR, 'best_model.keras')
    if not os.path.exists(best_model_path):
        print("No best model found. Skipping inference.")
        return
    
    output_dir = os.path.join(MAIN_MODEL_DIR, 'visualizations')
    
    # Run inference with best model
    inference_cmd = [
        "python", "inference_hamnosys.py",
        "--model-path", best_model_path,
        "--data-dir", "data/hamnosys_data",
        "--visualize",
        "--output-dir", output_dir
    ]
    
    print("\nRunning inference and visualization...")
    subprocess.run(inference_cmd)
    
    # Run visualization script for training metrics
    visual_cmd = [
        "python", "visualize_training.py",
        "--model-dir", MAIN_MODEL_DIR,
        "--data-dir", "data/hamnosys_data",
        "--output-dir", output_dir,
        "--include-model-analysis"
    ]
    
    print("Generating training visualizations...")
    subprocess.run(visual_cmd)
    
    print(f"Inference and visualization completed. Results in {output_dir}")

def main():
    """
    Main function to coordinate automated training
    """
    args = parse_args()
    summary_file = setup_model_directory()
    attempt = 0
    
    # Append logs to the main log file
    main_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.log_file)
    print(f"Logging to {main_log_file}")
    
    with open(main_log_file, 'a') as log_f:
        log_f.write(f"\n\n--- Automated Training Session Started at {datetime.now()} ---\n")
    
    # Process batches until we reach accuracy threshold or max attempts
    while attempt < MAX_ATTEMPTS:
        batch_num = get_next_batch(summary_file)
        
        # Log batch start
        with open(main_log_file, 'a') as log_f:
            log_f.write(f"\nBatch {batch_num} - Started at {datetime.now()}\n")
        
        # Run training for this batch
        result = run_training(batch_num, args, summary_file)
        
        if result:
            attempt += 1
            
            # Log batch results
            with open(main_log_file, 'a') as log_f:
                log_f.write(f"Batch {batch_num} - Completed at {datetime.now()}\n")
                log_f.write(f"  Accuracy: {result['accuracy']:.2f}%, Top-2: {result['top2_accuracy']:.2f}, Top-3: {result['top3_accuracy']:.2f}\n")
                log_f.write(f"  Training time: {result['training_time']:.2f} seconds\n")
                log_f.write(f"  Early stopping: {result['early_stopping']}\n\n")
            
            # Consolidate models after each batch
            consolidate_models()
            
            # Check if accuracy threshold is reached
            if result['accuracy'] >= ACCURACY_THRESHOLD:
                print(f"Accuracy threshold of {ACCURACY_THRESHOLD}% reached! Moving to inference.")
                break
        else:
            print(f"Training batch {batch_num} failed. Skipping to next batch.")
            
            # Log failure
            with open(main_log_file, 'a') as log_f:
                log_f.write(f"Batch {batch_num} - Failed at {datetime.now()}\n\n")
        
        # If --no-restart flag is set, break after one batch
        if args.no_restart:
            break
    
    # Run inference if enabled
    if args.run_inference:
        run_inference()
    
    # Final summary
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    print("\nTraining Summary:")
    print(f"  - Batches processed: {len(summary['training_runs'])}")
    print(f"  - Best accuracy: {summary['best_accuracy']:.2f}%")
    print(f"  - Best top-2 accuracy: {summary['best_top2_accuracy']:.2f}")
    print(f"  - Best top-3 accuracy: {summary['best_top3_accuracy']:.2f}")
    print(f"  - Total training time: {summary['total_training_time'] / 3600:.2f} hours")
    
    # Log final summary
    with open(main_log_file, 'a') as log_f:
        log_f.write("\nFinal Summary:\n")
        log_f.write(f"  Batches processed: {len(summary['training_runs'])}\n")
        log_f.write(f"  Best accuracy: {summary['best_accuracy']:.2f}%\n")
        log_f.write(f"  Best top-2 accuracy: {summary['best_top2_accuracy']:.2f}\n")
        log_f.write(f"  Best top-3 accuracy: {summary['best_top3_accuracy']:.2f}\n")
        log_f.write(f"  Total training time: {summary['total_training_time'] / 3600:.2f} hours\n")
        log_f.write(f"--- Automated Training Session Ended at {datetime.now()} ---\n\n")
    
    print("\nAutomated training completed! Check logs and visualization results.")

if __name__ == "__main__":
    main() 