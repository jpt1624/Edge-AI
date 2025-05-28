# import torch
# import urllib.request
# from PIL import Image
# from transformers import pipeline

# # Opening the image using PIL
# img = Image.open('JANDAYA_PARAKEET.jpg')

# # Loading the model and preprocessor using Pipeline
# pipe = pipeline("image-classification", model="dennisjooo/Birds-Classifier-EfficientNetB2")

# # Running the inference
# result = pipe(img)[0]

# # Printing the result label
# print(result['label'])

import torch
import timm
from PIL import Image
from torchvision import transforms
import onnx
import onnxruntime as ort
import numpy as np

# ----------- Step 1: Load Image -----------
img_path = "JANDAYA_PARAKEET.jpg"
img = Image.open(img_path).convert("RGB")

# ----------- Step 2: Define Preprocessing (EfficientNetB2 expected 288x288 input) -----------
transform = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

input_tensor = transform(img).unsqueeze(0)  # Shape: [1, 3, 288, 288]

# ----------- Step 3: Load Model from timm (EfficientNet B2) -----------
model = timm.create_model('efficientnet_b2', pretrained=True)
model.eval()

# ----------- Step 4: Test Inference -----------
with torch.no_grad():
    logits = model(input_tensor)
    pred = logits.argmax(-1).item()
    print(f"Predicted class index (PyTorch): {pred}")

# ----------- Step 5: Export to ONNX -----------
torch.onnx.export(
    model,
    input_tensor,
    "efficientnet_b2.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print("Model exported to efficientnet_b2.onnx")

# ----------- Step 6: Inference with ONNX Runtime -----------
ort_session = ort.InferenceSession("efficientnet_b2.onnx")
onnx_outputs = ort_session.run(None, {"input": input_tensor.numpy()})
onnx_pred = np.argmax(onnx_outputs[0], axis=-1)[0]

print(f"Predicted class index (ONNX): {onnx_pred}")

