import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.feature_extraction import load_dataset
from src.model import build_model
import os
import pickle

def train_pipeline(data_dir="data", epochs=10):
    # Load Data
    print("Loading dataset...")
    X, y = load_dataset(data_dir)
    
    if len(X) == 0:
        print("No data found. Please check data generation.")
        return
    
    # Transpose X to (Batch, Time, Features) = (Batch, 130, 40)
    # Current shape likely (Batch, 40, 130).
    X = np.transpose(X, (0, 2, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build Model
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Callbacks
    if not os.path.exists("models"):
        os.makedirs("models")
        
    checkpoint = tf.keras.callbacks.ModelCheckpoint("models/gender_model.h5", save_best_only=True, monitor='val_accuracy')
    
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        callbacks=[checkpoint],
        verbose=1
    )
    
    # Save History
    with open('models/history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
        
    return model, X_test, y_test
