import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization

def build_model(input_shape):
    model = Sequential()
    
    # Input shape: (n_mfcc, time_steps) -> We usually transpose for Conv1D to (time_steps, n_mfcc) 
    # OR we treat (n_mfcc, time) as 2D image.
    # Let's use Conv1D. We need to transpose inputs before feeding, or define input_shape=(n_mfcc, time_steps) and use data_format='channels_first' which is rare.
    # Easier: Transpose in data loading or here.
    # Let's assume input is (40, 130). We will interpret this as 40 features over 130 steps? 
    # Usually it's (Time, Features) for RNNs/1D-CNNs on sequence. 
    # Let's Assume input is (130, 40) for Conv1D. 
    
    model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))
    
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid')) # Binary classification
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
