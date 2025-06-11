# main.py
import os
import time
from audio_transcription import record_audio, transcribe_with_faster_whisper
from summarizer import summarize_transcript
from object_detection import record_video_with_yolo
from multiprocessing import Process


if __name__ == '__main__':
    # Create a new session folder
    session_folder = time.strftime("sessions/session_%Y-%m-%d_%H-%M-%S")
    os.makedirs(session_folder, exist_ok=True)
    transcript_path = os.path.join(session_folder, "transcript.txt")
    video_path = os.path.join(session_folder, "yolo.mp4")
    audio_path = os.path.join(session_folder, "audio.wav")

    try:
        obj_detection_thread = Process(target=record_video_with_yolo, args=(video_path,))
        record_audio_thread = Process(target=record_audio, args=(audio_path,))
        obj_detection_thread.start()
        record_audio_thread.start()

        # Wait until both processes terminate
        obj_detection_thread.join()
        record_audio_thread.join()

        # Transcribe
        print("[Whisper] 🧠 Running transcription...")
        transcribe_with_faster_whisper(audio_path, session_folder)
        # Summarize
        print("[Bart] 🧠 Running summarization...")
        summarize_transcript(session_folder)
        print(f'Session video, transcription, and summarization saved in the following path: {session_folder}')
    except KeyboardInterrupt:
        print('EXITING')
        

# OLD METHOD FOR REAL TIME, SLOW.------------------
# # main.py
# import threading
# import os
# import time
# from audio_transcription import live_transcription
# from summarizer import summarize_transcript
# from object_detection import obj_detection
# from multiprocessing import Process


# if __name__ == '__main__':
#     # Create a new session folder
#     session_folder = time.strftime("sessions/session_%Y-%m-%d_%H-%M-%S")
#     os.makedirs(session_folder, exist_ok=True)
#     transcript_path = os.path.join(session_folder, "transcript.txt")

#     try:
#         obj_detection_thread = Process(target=obj_detection, args=('live',))
#         transcription_thread = Process(target=live_transcription, args=(transcript_path,))
#         obj_detection_thread.start()
#         transcription_thread.start()
        
#     except KeyboardInterrupt:
#         print("\n🛑 Transcription stopped.")
#         # Automatically summarize after session ends
#         print("🧠 Running summarization...")
#         summarize_transcript(session_folder)
