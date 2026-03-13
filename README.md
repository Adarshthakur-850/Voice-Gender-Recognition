# Real-Time Voice Gender Recognition

Predicts speaker gender (Male/Female) from audio using MFCC features and Deep Learning (CNN).

## Project Structure
- `data/`: Generated synthetic audio.
- `models/`: Trained CNN model (`gender_model.h5`).
- `plots/`: Training curves and Confusion Matrix.
- `src/`: Code modules.

## Features
- **Data Gen**: Creates synthetic Male/Female voices using frequencies.
- **MFCC**: Extracts 40 features from audio.
- **Real-Time**: Captures microphone input for live prediction.

## Installation
```bash
pip install -r requirements.txt
# PortAudio might be needed for sounddevice:
# sudo apt-get install libportaudio2 (Linux)
```

## Usage
1. Train the model:
```bash
python main.py
```
2. Run Live Inference:
```bash
python src/inference.py
```
