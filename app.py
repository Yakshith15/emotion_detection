import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile

# Load the trained model
@st.cache_resource
def load_emotion_model():
    model = load_model("emotion_model.h5")  # Replace with your model's actual path
    return model

model = load_emotion_model()

# Emotion labels (update based on your dataset)
emotion_labels = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised']

# Function to predict emotion
def predict_emotion(features):
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    features = np.expand_dims(features, axis=-1)  # Add channel dimension
    predictions = model.predict(features)
    emotion_index = np.argmax(predictions)
    return emotion_labels[emotion_index]

# Audio processing function
def process_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return mfcc_features

# Streamlit UI
st.title("Speech Emotion Detection")
st.write("Upload a voice file or record your voice to detect the emotion.")

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_audio_path = temp_file.name
    
    mfcc_features = process_audio(temp_audio_path)
    emotion = predict_emotion(mfcc_features)
    st.success(f"Predicted Emotion: {emotion}")

# Real-time audio recording
st.write("Or record your voice below:")
if st.button("Start Recording"):
    duration = st.slider("Recording duration (seconds)", 1, 10, 5)
    fs = 22050  # Sampling frequency
    st.info("Recording...")
    recorded_audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()  # Wait until the recording is finished
    st.success("Recording finished!")
    
    # Save recording temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        write(temp_file.name, fs, recorded_audio)
        temp_audio_path = temp_file.name

    mfcc_features = process_audio(temp_audio_path)
    emotion = predict_emotion(mfcc_features)
    st.success(f"Predicted Emotion: {emotion}")
