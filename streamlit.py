import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import openai
from openai import ChatCompletion   # added import for ChatCompletion
import os
import requests
from dotenv import load_dotenv   # added import for dotenv
import asyncio
import sys  # added import for sys
from agents import Agent, Runner

# Disable traceback in Streamlit by overriding sys.excepthook
def streamlit_excepthook(exctype, value, tb):
    st.error(f"Error: {value}")
sys.excepthook = streamlit_excepthook

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
    user_message = user_input  # fix: assign captured input instead of st.text_input
    # Generate voice description
    voice_description = f"Duration: {round(duration,2)}s. Pitch: {pitch_arc}. Intensity: {round(float(intensity),4)}."

async def main():
    agent = Agent(
        name="Assistant",
        instructions="You are a communication assistant. Based on the acoustic tone (pitch shape, intensity, duration), "
                     "rewrite the user's intended message to match the mfccs and audio metrics by strongly aligning it with the expressive curve of the uploaded tone. "
    )
    # Use the correct variable user_message here
    result = await Runner.run(agent, f"Message: '{user_message}'\nTone Profile: {voice_description}")
    st.write(result.final_output)

asyncio.run(main())


