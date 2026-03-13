import numpy as np
import scipy.io.wavfile as wav
import os
import random

SR = 22050
DURATION = 3 # seconds

def generate_tone(freq, duration, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    # Add harmonics for realism
    tone += 0.3 * np.sin(2 * np.pi * (freq * 2) * t)
    tone += 0.2 * np.sin(2 * np.pi * (freq * 3) * t)
    return tone

def generate_synthetic_data(data_dir="data"):
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
        return
        
    print("Generating synthetic audio dataset...")
    
    label_map = {'male': (85, 180), 'female': (165, 255)}
    
    for label, (low, high) in label_map.items():
        save_path = os.path.join(data_dir, label)
        os.makedirs(save_path, exist_ok=True)
        
        for i in range(100): # Generate 100 samples per class
            fund_freq = random.randint(low, high)
            audio = generate_tone(fund_freq, DURATION, SR)
            
            # Add noise
            noise = np.random.normal(0, 0.05, audio.shape)
            audio += noise
            
            # Normalize
            audio = audio / np.max(np.abs(audio))
            
            # Save
            filename = os.path.join(save_path, f"{label}_{i}.wav")
            wav.write(filename, SR, audio.astype(np.float32))

    print("Data generation complete.")
