import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import queue
import sys

# Audio configuration
SR = 22050
DURATION = 3 # Seconds per chunk
BLOCK_SIZE = int(SR * DURATION)
audio_queue = queue.Queue()

def extract_live_features(audio, sr=22050, max_len=130):
   
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    
    
    if mfccs.shape[1] < max_len:
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_len]
        
    return mfccs.T # Returns (130, 40)

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Add data to queue to process in main thread
    audio_queue.put(indata.copy())

def real_time_inference(model_path="models/gender_model.h5"):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run 'python main.py' to train the model first.")
        return

    print("\n--------------------------------------------------")
    print(f"Starting Real-Time Recognition (Chunk: {DURATION}s)")
    print("Speak into your microphone...")
    print("Press Ctrl+C to stop.")
    print("--------------------------------------------------\n")
    
    try:
        with sd.InputStream(samplerate=SR, channels=1, blocksize=BLOCK_SIZE, callback=audio_callback):
            while True:
                # Get audio from queue (blocks until data is available)
                audio_data = audio_queue.get()
                
                # Preprocess
                audio_flat = audio_data.flatten()
                
                # Feature Extraction
                try:
                    features = extract_live_features(audio_flat, sr=SR)
                    # Add batch dimension: (1, 130, 40)
                    features = np.expand_dims(features, axis=0)
                    
                    # Predict
                    prediction = model.predict(features, verbose=0)[0][0]
                    
                    # Output
                    label = "FEMALE" if prediction > 0.5 else "MALE"
                    confidence = prediction if prediction > 0.5 else 1 - prediction
                    
                    # Print result with carriage return for "live" feel
                    print(f"Detected: {label} (Conf: {confidence*100:.1f}%)")
                    
                except Exception as e:
                    print(f"Processing error: {e}")
                    
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nMicrophone Error: {e}")
        print("Ensure you have a working microphone and 'portaudio' installed.")

if __name__ == "__main__":
    real_time_inference()
