You are a machine-learning engineer and full-stack developer working on the “Cosign-SignAvatars” repo (https://github.com/co-Sign/cosign-signavatars.git). I want you to:

1. **Understand the existing codebase**:
   - Inspect `datasets/hamnosys2motion/hamnosys_pkls_default_shape/` which contains pickle files (`.pkl`) with pre-extracted 3D keypoints.
   - Note that we no longer use `.npy` files—load and preprocess directly from these `.pkl` feature dumps.

2. **Create a modular CNN-LSTM hybrid pipeline** for real-time ASL recognition:
   - **Data loader** (`src/preprocessing.py`):  
     • Read each `.pkl` sequence (frames × features) from the dataset folder.  
     • Normalize and pad/truncate sequences to a fixed length (e.g., 50 frames × 1629 features).  
     • Split into train/val/test with stratified sampling based on gloss labels.  

   - **Model definition** (`src/model.py`):  
     • Build a 1D CNN encoder over time for spatial feature extraction (e.g., Conv1D → BatchNorm → ReLU → MaxPool).  
     • Stack 1–2 Bidirectional LSTM layers on CNN outputs to capture temporal dependencies.  
     • Final Dense → Softmax classification into gloss classes.  
     • Make hyperparameters configurable (filters, kernel sizes, LSTM units, dropout).

3. **Training script** (`src/train.py`):  
   - Accept CLI args: `--data-dir`, `--batch-size`, `--epochs`, `--lr`, `--seq-length`, `--output-dir`.  
   - Instantiate data loader, model, loss (`categorical_crossentropy`), optimizer (`Adam` with clipnorm).  
   - Use callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint (save best `.h5`), and TensorBoard.  
   - Log raw & percentage accuracy per epoch.  
   - Save final model in Keras HDF5 (`.h5`) format.

4. **Inference & API** (`src/inference.py` + `src/api.py`):  
   - `inference.py`: load `.h5` model, take a new `.pkl` or live webcam sequence, preprocess, predict gloss, output label & confidence.  
   - `api.py`: Flask server with endpoints:  
     • `POST /predict-sequence` → returns predicted gloss + confidence.  
     • `POST /speech-to-gloss` → chains speech-to-text (e.g., Google STT) → English-to-ASL-gloss API (via OpenAI key env var).  

5. **Avatar integration** (`src/avatar.js` or `/frontend/…`):  
   - Hook your prediction endpoint into the Three.js avatar so that each predicted gloss triggers a corresponding 3D signing animation (e.g. load `.glb` gestures).  
   - Optionally broadcast avatar frames back into a video call via the Stream Video SDK.

6. **Environment & deployment**:  
   - Specify recommended Python 3.9/3.10 with `conda env create -f environment.yml`.  
   - Pin core deps: `tensorflow==2.13.0`, `keras==2.13.0`, `torch` only if needed, `mediapipe` removed (we rely on stored keypoints), plus Flask, python-dotenv, requests.  
   - Document GPU vs CPU install notes in `README.md`.

7. **Documentation & next steps**:  
   - Write `README.md` sections on setup, data preprocessing, training, inference, API usage, avatar integration, and performance troubleshooting.  
   - Suggest future additions: transformer hybrid block, real-time voice feedback, multi-sign language support.

Please generate all necessary code files, folder structure updates, CLI instructions, and example commands so I can run end-to-end training and inference straight away. Start by outlining the project tree and then drop in each file’s contents with comments explaining key parts.
