#!/usr/bin/env python3
"""
Cleanup script for the ASL recognition pipeline.
Removes generated files and directories.
"""

import os
import shutil
import argparse

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Clean up generated files')
    
    parser.add_argument('--all', action='store_true',
                        help='Remove all generated files and directories')
    parser.add_argument('--models', action='store_true',
                        help='Remove trained models')
    parser.add_argument('--logs', action='store_true',
                        help='Remove log files')
    parser.add_argument('--cached', action='store_true',
                        help='Remove cached files')
    
    return parser.parse_args()

def clean_models(base_dir):
    """
    Remove trained models
    
    Args:
        base_dir: Base directory of the project
    """
    models_dir = os.path.join(base_dir, 'models')
    
    if os.path.exists(models_dir) and os.path.isdir(models_dir):
        print(f"Removing models in {models_dir}...")
        
        # Get all subdirectories in models_dir
        model_dirs = [
            os.path.join(models_dir, d) for d in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, d))
        ]
        
        # Remove each model directory
        for model_dir in model_dirs:
            print(f"  Removing {model_dir}")
            shutil.rmtree(model_dir)
        
        print("Models removed successfully.")
    else:
        print(f"Models directory {models_dir} not found.")

def clean_logs(base_dir):
    """
    Remove log files
    
    Args:
        base_dir: Base directory of the project
    """
    # Define directories that may contain log files
    log_dirs = [
        os.path.join(base_dir, 'logs'),
        os.path.join(base_dir, 'models')
    ]
    
    for log_dir in log_dirs:
        if os.path.exists(log_dir) and os.path.isdir(log_dir):
            print(f"Searching for log files in {log_dir}...")
            
            # Find all log files
            for root, _, files in os.walk(log_dir):
                for file in files:
                    if file.endswith('.log') or 'log' in file.lower():
                        log_file = os.path.join(root, file)
                        print(f"  Removing {log_file}")
                        os.remove(log_file)
    
    print("Log files removed successfully.")

def clean_cached_files(base_dir):
    """
    Remove cached files
    
    Args:
        base_dir: Base directory of the project
    """
    # Define patterns for cached files/directories
    cache_patterns = [
        '__pycache__',
        '.pytest_cache',
        '.ipynb_checkpoints',
        '*.pyc',
        '*.pyo',
        '*.pyd'
    ]
    
    print(f"Removing cached files...")
    
    # Walk through the directory and remove cached files/directories
    for root, dirs, files in os.walk(base_dir):
        # Skip .git directory
        if '.git' in dirs:
            dirs.remove('.git')
        
        # Check directories
        for cache_pattern in cache_patterns:
            if not cache_pattern.startswith('*'):  # Directory pattern
                for d in dirs[:]:  # Use a copy for iteration
                    if d == cache_pattern:
                        cache_dir = os.path.join(root, d)
                        print(f"  Removing {cache_dir}")
                        shutil.rmtree(cache_dir)
                        dirs.remove(d)  # Remove from original list to avoid recursion
        
        # Check files
        for cache_pattern in cache_patterns:
            if cache_pattern.startswith('*'):  # File pattern
                ext = cache_pattern[1:]  # Remove the '*' prefix
                for file in files:
                    if file.endswith(ext):
                        cache_file = os.path.join(root, file)
                        print(f"  Removing {cache_file}")
                        os.remove(cache_file)
    
    print("Cached files removed successfully.")

def main():
    """
    Main function
    """
    args = parse_arguments()
    
    # Get base directory (project root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args.all or args.models:
        clean_models(base_dir)
    
    if args.all or args.logs:
        clean_logs(base_dir)
    
    if args.all or args.cached:
        clean_cached_files(base_dir)
    
    if not (args.all or args.models or args.logs or args.cached):
        print("No cleanup options specified. Use --all, --models, --logs, or --cached.")
        print("Run 'python clean.py --help' for more information.")

if __name__ == '__main__':
    main() 