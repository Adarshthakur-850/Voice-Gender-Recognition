import librosa
import numpy as np
import os

def extract_features(file_path, max_len=130):
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio)
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        
        # Pad or truncation
        if mfccs.shape[1] < max_len:
            pad_width = max_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_len]
            
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_dataset(data_dir="data"):
    X = []
    y = []
    labels = ['male', 'female']
    
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        
        if not os.path.exists(path):
            continue
            
        for file in os.listdir(path):
            if file.endswith('.wav'):
                file_path = os.path.join(path, file)
                feats = extract_features(file_path)
                if feats is not None:
                    X.append(feats)
                    y.append(class_num)
                    
    return np.array(X), np.array(y)
