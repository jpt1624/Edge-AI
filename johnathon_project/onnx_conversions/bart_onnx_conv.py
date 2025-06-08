from transformers import AutoTokenizer
from optimum.exporters.onnx import main_export
from pathlib import Path

# Model to export
model_id = "sshleifer/distilbart-cnn-12-6"

# Output folder
output_dir = Path("onnx_distilbart")

# Export ONNX
main_export(
    model_name_or_path="sshleifer/distilbart-cnn-12-6",
    output=Path("onnx_distilbart"),
    task="summarization",
    opset=14  # ← fixed here
)

print(f"Exported to {output_dir.resolve()}")