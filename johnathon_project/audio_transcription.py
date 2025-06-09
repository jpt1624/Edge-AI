# live_transcriber.py

import sounddevice as sd
import numpy as np
import torch
from transformers import AutoProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
import webrtcvad
import queue
import time

# Load model and processor once
model = ORTModelForSpeechSeq2Seq.from_pretrained("onnx_models/whisper-small-xenova")
processor = AutoProcessor.from_pretrained("openai/whisper-small")

# Global settings
vad = webrtcvad.Vad(0)
samplerate = 16000
chunk_duration = 0.5
blocksize = int(samplerate * chunk_duration)
audio_queue = queue.Queue()
speech_buffer = []
last_speech_time = 0
min_speech_duration = 2.0
silence_threshold = 1.0

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

def is_speech(audio: np.ndarray, sample_rate: int = 16000, min_frames: int = 3) -> bool:
    pcm = (audio * 32768).astype(np.int16)
    frame_size = int(0.03 * sample_rate)
    speech_frames = 0
    for i in range(0, len(pcm) - frame_size, frame_size):
        frame = pcm[i:i + frame_size].tobytes()
        if vad.is_speech(frame, sample_rate):
            speech_frames += 1
        if speech_frames >= min_frames:
            return True
    return False

def live_transcription(transcript_path: str):
    global last_speech_time, speech_buffer

    with sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback, blocksize=blocksize):
        print("🎙️ Listening... (Press Ctrl+C to stop)")

        while True:
            audio_chunk = audio_queue.get()
            audio_np = np.squeeze(audio_chunk)
            now = time.time()

            if is_speech(audio_np):
                speech_buffer.append(audio_np)
                last_speech_time = now
            elif now - last_speech_time > silence_threshold and speech_buffer:
                total_audio = np.concatenate(speech_buffer)
                total_duration = len(total_audio) / samplerate

                if total_duration >= min_speech_duration:
                    inputs = processor(
                        total_audio,
                        sampling_rate=samplerate,
                        return_tensors="pt",
                        task="transcribe",
                        language="en"
                    )
                    with torch.no_grad():
                        output_ids = model.generate(**inputs, num_beams=1)
                    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

                    if len(text.strip()) >= 5 and any(c.isalnum() for c in text):
                        timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
                        line = f"{timestamp} {text.strip()}\n"
                        print("🗣️", line.strip())

                        with open(transcript_path, "a", encoding="utf-8") as f:
                            f.write(line)
                    else:
                        print("🤖 Skipped low-confidence or noise")
                else:
                    print("⏭️ Skipped: speech too short")

                speech_buffer = []

