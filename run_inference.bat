@echo off
REM Inference Launcher for Sign Language Recognition
ECHO Starting Sign Language Recognition Inference...

REM Activate conda environment
CALL conda activate signavatars

REM Run inference with best model
python inference_hamnosys.py --model-path models/hamnosys/main_model/best_model.keras --visualize --output-dir models/hamnosys/main_model/visualizations

ECHO Inference completed. Check visualization results.
PAUSE 