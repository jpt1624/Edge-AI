# we need to change to a computer vision model to detect real time objects instead of resnet
# we need to implement pyaudio to use with the audio transcript model
# we need to have a text generation model

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo11n.onnx'