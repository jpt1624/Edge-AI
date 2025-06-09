# main.py

import os
import time
from audio_transcription import live_transcription
from summarizer import summarize_transcript

# Create a new session folder
session_folder = time.strftime("sessions/session_%Y-%m-%d_%H-%M-%S")
os.makedirs(session_folder, exist_ok=True)
transcript_path = os.path.join(session_folder, "transcript.txt")

try:
    # Run live transcription
    live_transcription(transcript_path)
except KeyboardInterrupt:
    print("\n🛑 Transcription stopped.")
    # Automatically summarize after session ends
    print("🧠 Running summarization...")
    summarize_transcript(session_folder)
