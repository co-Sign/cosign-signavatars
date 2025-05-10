@echo off
REM Enhanced Training Launcher for Sign Language Recognition
ECHO Starting Sign Language Recognition Training with Enhanced Model...

REM Activate conda environment
CALL conda activate signavatars

REM Skip problematic WLASL pickle files and focus on HamNoSys data
ECHO Skipping WLASL pickle files and focusing on HamNoSys data...

REM Run the new unified training script with optimized parameters
python train.py ^
  --data-dir data/hamnosys_data ^
  --output-dir models/unified ^
  --epochs 500 ^
  --patience 40 ^
  --lr 0.0003 ^
  --batch-size 16 ^
  --dropout-rate 0.4 ^
  --l2-reg 0.0005 ^
  --attention-heads 16 ^
  --augment ^
  --augment-prob 0.5 ^
  --normalize-features ^
  --use-class-weights ^
  --visualize

ECHO Training completed. Check the training summary in models/unified directory.
PAUSE 