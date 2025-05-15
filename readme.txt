🎤 Tone-to-Intent Translator for Speech Empowerment

This app helps individuals with speech impairments express intent through nonverbal vocalizations. By analyzing the acoustic tone of a sound (e.g., hums, melodic cues), the system infers emotional and contextual meaning, rewrites the user’s intended message using OpenAI, and generates a matching voice with ElevenLabs.

✨ Features

Audio Upload: Accept .mp3 or .wav files

Pitch & Intensity Analysis: Using librosa

Tone-Matched Text Rewriting: Powered by OpenAI GPT-4

Voice Generation: Uses ElevenLabs text-to-speech with emotional settings

Natural Language Summary: Acoustic data translated into concise message feedback

On-Device Playback: Play the tone and synthesized voice side-by-side

🚀 Launching the App (Locally)

1. Clone the Repository

git clone https://github.com/Derabian/tone-intent-translator.git
cd tone-intent-translator

2. Install Dependencies

pip install -r requirements.txt

If you don’t have a requirements.txt, install manually:

pip install streamlit librosa openai python-dotenv elevenlabs

3. Set Up Your .env File

Create a file called .env in the project root:

(Your ElevenLabs key is currently hardcoded in the script — consider moving that to the .env as well.)

4. Run the App

streamlit run protoai.py

🛠️ Tech Stack

Python 3.10+

Streamlit – Interactive front-end

Librosa – Audio feature extraction

OpenAI SDK (v1.0+) – Language model integration

ElevenLabs SDK – Voice synthesis

dotenv – Secure API key management

📁 Project Structure

.
├── streamlit.py         # Main Streamlit app
├── .env               # Environment variables (not committed)
├── requirements.txt   # Python dependencies
└── README.md

🧠 Future Enhancements

Multiple voice selections

Emotion classification from pitch contour

Downloadable interaction logs

Side-by-side tone vs. synthetic comparison UI

Web deployment (Streamlit Cloud or Hugging Face)

🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss changes or feature ideas.

📜 License

MIT License – use freely and modify with attribution.

