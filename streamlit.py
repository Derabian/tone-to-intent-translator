import librosa
"""
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import openai
from openai import ChatCompletion   # added import for ChatCompletion
import os
import requests
from dotenv import load_dotenv   # added import for dotenv

# Function to record audio
# Step 1: Record or Upload Audio
st.header("Step 1: Record or Upload Audio")
audio_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])
audio_value = st.audio_input("Record a voice message")


# Function to analyze audio
def get_audio_analysis(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    duration = librosa.get_duration(y=y, sr=sr)
    pitch = librosa.yin(y, fmin=50, fmax=300, sr=sr)
    pitch_arc = f"{int(np.min(pitch))}Hz → {int(np.median(pitch))}Hz → {int(np.max(pitch))}Hz"
    intensity = np.mean(librosa.feature.rms(y=y))
    
    # Display analysis
    st.subheader("📊 Acoustic Summary")
    st.write({
        "MFCCs (mean)": [round(float(mfcc), 4) for mfcc in mfccs_mean],
        "Duration (s)": round(duration, 2),
        "Pitch Arc": pitch_arc,
        "Intensity (RMS avg)": round(float(intensity), 4)
    })
    return { "duration": duration, "pitch_arc": pitch_arc, "intensity": intensity }

# Streamlit UI
st.title("Tone Analysis and Message Refinement")

# Step 2: Analyze Audio Tone
if audio_file:
    st.header("Step 2: Analyze Audio Tone")
    y, sr = librosa.load(audio_file)

    if audio_value:
        y, sr = librosa.load(audio_value)

    if len(y) == 0:
        st.error("The uploaded or recorded audio file is empty. Please try again.")
    else:
        # Analyze audio and display results
        analysis = get_audio_analysis(y, sr)
        duration = analysis["duration"]
        pitch_arc = analysis["pitch_arc"]
        intensity = analysis["intensity"]

        # Step 3: Enter Intended Message
        st.header("Step 3: Enter Your Intended Message")
        user_input = st.text_input("Enter your intended message:")

        if st.button("Refine Message"):
            user_message = user_input  # Assign the intended message to user_message

            # Generate voice description
            voice_description = f"Duration: {round(duration,2)}s. Pitch: {pitch_arc}. Intensity: {round(float(intensity),4)}."

            # Use ChatCompletion create() method from the imported ChatCompletion
            system_prompt = (
                "You are a communication assistant. Based on the acoustic tone (pitch shape, intensity, duration), "
                "rewrite the user's intended message to match the emotional tone and rhythm suggested by the sound. "
                "Keep the structure and message intact, but align it with the expressive curve of the uploaded tone."
            )

            chat_completion = ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Message: '{user_message}'\nTone Profile: {voice_description}"}
                ]
            )

            revised_message = chat_completion.choices[0].message.content.strip()

            # Display revised message
            st.subheader("📝 OpenAI-Rewritten Message")
            st.markdown(revised_message)

            # Generate synthetic voice from ElevenLabs
            st.subheader("🎧 Synthesized Voice Output")
            st.markdown(f"**Original Message:** {user_message}")
            st.markdown(f"**Revised Message:** {revised_message}")
            st.markdown(f"**Voice Description:** {voice_description}")

# New Section: Step 4 - Display Metrics Analysis Upload Use Case
if st.button("Upload Training Metrics"):
    load_dotenv("/home/dastinkartoum/first/start/.env")
    if not os.getenv('OPENAI_API_KEY'):
        st.error("API key not found. Please set the OPENAI_API_KEY environment variable.")
    else:
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
        }
        data = {
            "purpose": "audio_anlysis",
            "filename": "training_examples.jsonl",
            "bytes": 2147483648,
            "mime_type": "text/jsonl"
        }
        response = requests.post("https://api.openai.com/v1/uploads", headers=headers, json=data)
        if response.ok:
            st.success("Upload successful: " + str(response.json()))
        else:
            st.error("Upload failed: " + response.text)
