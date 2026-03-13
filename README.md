# 🎙️ Voice Gender Recognition

A machine learning project that predicts whether a voice belongs to a male or female speaker using audio signal processing and classification models.

---

## 📌 Project Description

Voice Gender Recognition is a supervised machine learning problem where the goal is to classify a speaker's gender based on their voice characteristics.

Human speech contains measurable acoustic properties such as:
- Pitch (fundamental frequency)
- Energy
- Spectral features
- Harmonics
- Formants

These characteristics vary statistically between male and female speakers. This project extracts relevant features from audio signals and trains a classification model to learn these patterns and perform accurate predictions.

The system follows a complete ML pipeline:
1. Audio Data Collection
2. Feature Extraction
3. Data Preprocessing
4. Model Training
5. Evaluation
6. Prediction

---

## 🏗️ System Architecture

Audio Input (.wav)
        ↓
Feature Extraction (MFCC, Spectral Features)
        ↓
Data Preprocessing (Normalization, Label Encoding)
        ↓
Model Training (ML / Deep Learning)
        ↓
Evaluation (Accuracy, Confusion Matrix)
        ↓
Prediction (Male / Female)

---

## 🔬 Technical Approach

### 1️⃣ Audio Preprocessing

Raw audio signals cannot be directly fed into machine learning models. Therefore:

- Audio is loaded using `librosa`
- Sampling rate is standardized
- Noise is optionally reduced
- Audio is converted into numerical feature vectors

---

### 2️⃣ Feature Extraction

The most important part of voice-based ML systems.

Common features used:

#### 🔹 MFCC (Mel-Frequency Cepstral Coefficients)
MFCCs represent short-term power spectrum of sound and are widely used in speech recognition tasks.

#### 🔹 Spectral Centroid
Indicates where the "center of mass" of the spectrum is located.

#### 🔹 Zero Crossing Rate
Measures how often the signal changes sign.

#### 🔹 Chroma Features
Capture harmonic and pitch information.

These features are combined into a structured feature vector for each audio sample.

---

### 3️⃣ Model Training

After feature extraction:

- Features are split into training and testing sets
- Labels are encoded (Male → 0, Female → 1)
- A classification algorithm is trained

Possible models:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Artificial Neural Network (ANN)
- CNN (if using spectrogram images)

---

### 4️⃣ Model Evaluation

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

This helps measure how well the model generalizes to unseen data.

---

## 📊 Example Output

Input: sample_voice.wav  
Output: Predicted Gender → Female  
Confidence Score → 0.92  

---

## 📂 Project Structure

```

Voice-Gender-Recognition/
│
├── data/                 # Audio dataset
├── models/               # Saved trained model
├── plots/                # Training performance graphs
├── src/                  # Core source code
│   ├── feature_extraction.py
│   ├── train.py
│   ├── predict.py
│
├── main.py               # Entry point
├── requirements.txt
└── README.md

````

---

## 🛠️ Installation & Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/Adarshthakur-850/Voice-Gender-Recognition.git
cd Voice-Gender-Recognition
````

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Train Model

```bash
python main.py
```

### Predict Gender

```bash
python predict.py --file sample.wav
```

---

## 📈 Results

The trained model achieves strong classification performance when trained on clean, labeled datasets.

Typical performance:

* Accuracy: 85% – 95% (depends on dataset quality)
* Balanced precision/recall across classes

---

## 📚 Applications

* Voice-based biometric systems
* Call center automation
* Smart assistants
* Customer analytics
* Audio-based demographic studies

---

## ⚠️ Limitations

* Performance depends heavily on audio quality
* Background noise can reduce accuracy
* Not designed for non-binary gender classification
* Requires balanced dataset for fairness

---

## 🔮 Future Enhancements

* Deploy as REST API (FastAPI / Flask)
* Add Web Interface
* Use CNN on spectrogram images
* Integrate real-time microphone prediction
* Expand to multi-class speaker profiling

---

## 🧠 Learning Outcomes

This project demonstrates:

* Digital Signal Processing basics
* Feature engineering for audio data
* Supervised machine learning pipeline
* Model evaluation techniques
* Practical ML deployment workflow

---

## 📌 Author

Adarsh Thakur
Machine Learning Enthusiast
