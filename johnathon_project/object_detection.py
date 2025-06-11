from ultralytics import YOLO
import os

# Load the exported ONNX model
def load_model():
    onnx_model = YOLO("onnx_models/yolo11n.onnx")
    return onnx_model

# Run inference
def obj_detection( mediatype, path=None):
    onnx_model = load_model()
    if mediatype == 'video':
        results = onnx_model(source=path, show=True, conf=0.4, save=False)
    elif mediatype == 'image':
        results = onnx_model(path, task='segment', show=True, save=False)
    elif mediatype == 'live':
        results = onnx_model(source=0, show=True, conf=0.4, save=False)
    # print(results)
    
import cv2
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import threading
import time
import subprocess
import os
from ultralytics import YOLO

def record_video_with_yolo(output_path, duration=30):
    print("[YOLO] Starting real-time detection...")
    model = YOLO("onnx_models/yolo11n.onnx")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera error")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    dirpath = os.path.dirname(output_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or time.time() - start_time > duration:
            break

        results = model.predict(frame, conf=0.4, verbose=False)
        annotated = results[0].plot()
        out.write(annotated)
        cv2.imshow("YOLO Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[YOLO] Video saved to {output_path}")
