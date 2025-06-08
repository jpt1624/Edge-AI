from ultralytics import YOLO
import os

# Load the exported ONNX model
def load_model():
    onnx_model = YOLO("onnx_models/yolo11n.onnx")
    return onnx_model

# Run inference
def obj_detection(path, onnx_model, mediatype):
    if mediatype == 'video':
        results = onnx_model(source=path, show=True, conf=0.4, save=False)
    elif mediatype == 'image':
        results = onnx_model(path, task='segment', show=True, save=False)
    elif mediatype == 'live':
        results = onnx_model(source=1, show=True, conf=0.4, save=False)
    print(results)

def main():
    onnx_model = load_model()
    # path = "test_data/images/video.mp4"

    for img in os.listdir('test_data/images'):
        path = os.path.join('test_data/images/' + img)
        obj_detection(path, onnx_model, 'image')

main()