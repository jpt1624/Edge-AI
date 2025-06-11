# summarizer.py

from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
import os

def summarize_transcript(session_path: str):
    transcript_path = os.path.join(session_path, "transcript.txt")
    summary_path = os.path.join(session_path, "summary.txt")

    # Load transcript
    with open(transcript_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    encoder_sess = ort.InferenceSession("onnx_models/onnx_distilbart/encoder_model.onnx")
    decoder_sess = ort.InferenceSession("onnx_models/onnx_distilbart/decoder_model.onnx")

    # Preprocess input
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=512
    )
    encoder_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }

    # Run encoder
    encoder_outputs = encoder_sess.run(None, encoder_inputs)
    encoder_hidden_states = encoder_outputs[0]

    # Decode step-by-step
    decoder_input_ids = np.array([[tokenizer.bos_token_id]])
    summary_ids = []
    max_length = 64

    for _ in range(max_length):
        decoder_inputs = {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": inputs["attention_mask"]
        }
        decoder_outputs = decoder_sess.run(None, decoder_inputs)
        next_token_logits = decoder_outputs[0][:, -1, :]
        next_token_id = np.argmax(next_token_logits, axis=-1)

        if next_token_id[0] == tokenizer.eos_token_id:
            break

        summary_ids.append(next_token_id[0])
        decoder_input_ids = np.hstack([decoder_input_ids, next_token_id.reshape(1, 1)])

    summary = tokenizer.decode(summary_ids, skip_special_tokens=True)

    # Save summary
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print("🧠 Summary saved to:", summary_path)