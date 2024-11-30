import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
from predict import predict_emotion
import librosa

st.title("Speech Emotion Detection")
st.header("Upload a file or record audio to detect emotion.")

option = st.selectbox("Input Method", ["Upload File", "Record Audio"])

if option == "Upload File":
    file = st.file_uploader("Upload a WAV file", type=["wav"])
    if file:
        st.audio(file, format="audio/wav")
        emotion = predict_emotion(file)
        st.write(f"Predicted Emotion: {emotion}")

elif option == "Record Audio":
    duration = st.slider("Recording duration (seconds)", 1, 10, 3)
    if st.button("Record"):
        st.write("Recording...")
        fs = 22050
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        write("recorded.wav", fs, audio)
        audio, _ = librosa.load("recorded.wav", sr=22050)
        emotion = predict_emotion(audio)
        st.write(f"Predicted Emotion: {emotion}")
