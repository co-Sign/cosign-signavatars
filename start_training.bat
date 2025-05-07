@echo off
REM Automated Training Launcher for Sign Language Recognition
ECHO Starting Sign Language Recognition Training...

REM Activate conda environment
CALL conda activate signavatars

REM Run automated training script
python automated_training.py --epochs 500 --patience 30 --lr 0.0005 --augment-factor 15

ECHO Training completed or stopped. Check training logs for details.
PAUSE 