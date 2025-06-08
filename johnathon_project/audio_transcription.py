from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor
import torchaudio
import torch

model = ORTModelForSpeechSeq2Seq.from_pretrained("onnx_models/whisper-small-xenova")
processor = AutoProcessor.from_pretrained("openai/whisper-small")

waveform, sr = torchaudio.load("test_data/valley.mp3")
if sr != 16000:
    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    output_ids = model.generate(**inputs)

print("🗣️ Transcription:", processor.batch_decode(output_ids, skip_special_tokens=True)[0])
