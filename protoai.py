# This script requires Streamlit and Librosa to run in a local or cloud environment
# where audio and UI interactions are supported.

try:
    import streamlit as st
    import librosa
    import numpy as np
    import tempfile
    import openai
    import os
    from dotenv import load_dotenv
    from elevenlabs.client import ElevenLabs, VoiceSettings
except ModuleNotFoundError as e:
    print("❌ Required modules not found:", e)
    print("Please run this script in an environment with Streamlit, Librosa, ElevenLabs SDK, OpenAI Python client, and python-dotenv installed.")
    exit(1)

# === CONFIGURATION ===
load_dotenv()  # Load environment variables from .env file

ElevenLabs.api_key = os.getenv("api_key")
client = ElevenLabs(api_key=ElevenLabs.api_key)

# Securely load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("❌ OPENAI_API_KEY environment variable is not set. Please set it before running the app.")

st.title("🎤 Tone-to-Intent Translator for Speech Empowerment")

st.markdown("""
This app helps individuals with speech impairments express intent through sound.
Upload a vocal sound (e.g., hum, tone), describe the intended message in text, and receive a matching synthesized voice.
""")

# Step 1: Upload audio file
uploaded_file = st.file_uploader("Upload your vocal sound (MP3/WAV)", type=["mp3", "wav"], key="voice_upload")

# Step 2: Enter intended message
user_message = st.text_input("What were you trying to communicate with this sound?")

# Step 3: Analyze audio if both inputs are present
if uploaded_file and user_message:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(audio_path, format='audio/wav')

    # Load audio with librosa
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    pitch = librosa.yin(y, fmin=50, fmax=300, sr=sr)
    pitch_arc = f"{int(np.min(pitch))}Hz → {int(np.median(pitch))}Hz → {int(np.max(pitch))}Hz"
    intensity = np.mean(librosa.feature.rms(y=y))

    # Display analysis
    st.subheader("📊 Acoustic Summary")
    st.write({
        "Duration (s)": round(duration, 2),
        "Pitch Arc": pitch_arc,
        "Intensity (RMS avg)": round(float(intensity), 4)
    })

    # Generate voice description
    voice_description = f"Duration: {round(duration,2)}s. Pitch: {pitch_arc}. Intensity: {round(float(intensity),4)}."

    # Use OpenAI to adapt user message with acoustic shape using the new v1 API
    system_prompt = (
        "You are a communication assistant. Based on the acoustic tone (pitch shape, intensity, duration),"
        " rewrite the user's intended message to match the emotional tone and rhythm suggested by the sound."
        " Keep the structure and message intact, but align it with the expressive curve of the uploaded tone."
    )

    client_openai = openai.OpenAI()
    chat_completion = client_openai.chat.completions.create(
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

    try:
        # Generate audio using ElevenLabs
        audio_generator = client.text_to_speech.convert(
            text=revised_message,
            voice_id="JBFqnCBsd6RMkjVDRZzb",  # Replace with the correct voice ID
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.3,
                similarity_boost=0.6,
                style=0.4
            )
        )
        
        # Write the generator's output to a file
        output_path = "synthesized_response.mp3"
        with open(output_path, "wb") as f:
            for chunk in audio_generator:
                f.write(chunk)
        
        # Play the audio in Streamlit
        st.audio(output_path, format="audio/mp3")
    except Exception as e:
        st.error(f"❌ Failed to generate voice: {e}")

else:
    st.info("Please upload a sound and type your intended message above.")
