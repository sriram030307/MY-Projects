
import streamlit as st
from emotion_model import detect_emotion

st.title("😄 Emotion Detector from Text")
text = st.text_area("Type a message...")

if st.button("Detect Emotion"):
    if text:
        emotion = detect_emotion(text)
        st.success(f"Detected Emotion: {emotion}")
