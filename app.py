import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import soundfile as sf
import os
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained model
model = tf.keras.models.load_model("emotion_model.h5")

# Define emotion labels (Update based on dataset used)
emotion_labels = {
    0: "Neutral",
    1: "Calm",
    2: "Happy",
    3: "Sad",
    4: "Angry",
    5: "Fearful",
    6: "Disgusted",
    7: "Surprised"
}

# Load the scaler used for feature normalization
if os.path.exists("scaler.pkl"):
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
else:
    scaler = StandardScaler()

# Function to extract features from an audio file
def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)

        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)

        # Compute means
        mfcc_mean = np.mean(mfcc, axis=1)
        chroma_mean = np.mean(chroma, axis=1)
        mel_mean = np.mean(mel, axis=1)

        # Combine features into a single array
        features = np.hstack([mfcc_mean, chroma_mean, mel_mean])
        
        # Normalize features
        features = scaler.transform([features])

        # Reshape for model input
        features = features.reshape(1, features.shape[1], 1)
        return features
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# Streamlit UI
st.title("🎵 Emotion Detection from Audio")
st.write("Upload an audio file to detect the emotion.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    # Save uploaded file temporarily
    temp_audio_path = "temp_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract features
    features = extract_features(temp_audio_path)

    if features is not None:
        # Predict emotion
        prediction = model.predict(features)
        predicted_emotion = emotion_labels[np.argmax(prediction)]

        # Display result
        st.success(f"🎭 Detected Emotion: *{predicted_emotion}*")