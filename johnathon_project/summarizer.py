from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

# Load ONNX sessions
encoder_sess = ort.InferenceSession("onnx_distilbart/encoder_model.onnx")
decoder_sess = ort.InferenceSession("onnx_distilbart/decoder_model.onnx")


## Input text
text = "The disaster occurred while running a test to simulate cooling the reactor during an accident in blackout conditions. The operators carried out the test despite an accidental drop in reactor power, and due to a design issue, attempting to shut down the reactor in those conditions resulted in a dramatic power surge. The reactor components ruptured and lost coolants, and the resulting steam explosions and meltdown destroyed the Reactor building no. 4, followed by a reactor core fire that spread radioactive contaminants across the Soviet Union and Europe.[6] A 10-kilometre (6.2 mi) exclusion zone was established 36 hours after the accident, initially evacuating around 49,000 people. The exclusion zone was later expanded to 30 kilometres (19 mi), resulting in the evacuation of approximately 68,000 more people.[7]"

# Tokenize input
inputs = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=512)
encoder_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}

# Run encoder
encoder_outputs = encoder_sess.run(None, encoder_inputs)
encoder_hidden_states = encoder_outputs[0]

# Decoder setup
decoder_input_ids = np.array([[tokenizer.bos_token_id]])  # Start with <s> token (BART uses this)
max_length = 64
summary_ids = []

for _ in range(max_length):
    decoder_inputs = {
        "input_ids": decoder_input_ids,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": inputs["attention_mask"]
    }
    
    decoder_outputs = decoder_sess.run(None, decoder_inputs)
    next_token_logits = decoder_outputs[0][:, -1, :]  # take last token
    next_token_id = np.argmax(next_token_logits, axis=-1)
    
    if next_token_id[0] == tokenizer.eos_token_id:
        break

    summary_ids.append(next_token_id[0])
    decoder_input_ids = np.hstack([decoder_input_ids, next_token_id.reshape(1, 1)])

# Decode result
summary = tokenizer.decode(summary_ids, skip_special_tokens=True)
