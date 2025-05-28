# Edge AI Development Environment Setup

This repository contains a setup script that creates a complete Edge AI development environment with Python 3.12, UV package manager, and all necessary dependencies for computer vision, machine learning, and AI model deployment.

## 🚀 Quick Start

```bash
# Clone or download the setup script
chmod +x setup_edge_ai_env.sh
./setup_edge_ai_env.sh
```

## 📋 What This Script Does

The `setup_edge_ai_env.sh` script automatically:

1. **Installs UV Package Manager** - A fast Python package installer and resolver
2. **Creates a Python 3.12 Virtual Environment** - Isolated environment using UV
3. **Installs Comprehensive AI Dependencies** - All packages needed for Edge AI development
4. **Verifies Installation** - Checks that key packages are working correctly
5. **Provides Usage Instructions** - Shows you how to activate and use the environment

## 🔧 Prerequisites

- **Operating System**: Linux, macOS, or Windows (with Git Bash/WSL)
- **Internet Connection**: Required for downloading packages
- **Disk Space**: ~3-5GB for all dependencies
- **Permissions**: Regular user permissions (avoid running with sudo)

## 📦 Installed Packages

### Core AI/ML Frameworks
- **PyTorch** - Deep learning framework
- **TensorFlow** - Google's ML platform
- **Ultralytics** - YOLO object detection
- **Transformers** - Hugging Face NLP models

### Computer Vision
- **OpenCV** - Computer vision library
- **Pillow** - Image processing
- **scikit-image** - Image analysis algorithms

### Model Optimization & Deployment
- **ONNX & ONNX Runtime** - Model interoperability
- **Optimum** - Model optimization tools
- **Neural Compressor** - Intel's optimization toolkit

### Development & Deployment Tools
- **FastAPI** - Modern web API framework
- **Streamlit** - Data app framework
- **Gradio** - ML demo interfaces
- **JupyterLab** - Interactive development environment

### Audio Processing
- **Librosa** - Audio analysis
- **SoundFile** - Audio file I/O

## 🎯 Usage Instructions

### Option 1: Using UV Run (Recommended)
```bash
# Run Python scripts directly
uv run --python .venv python your_script.py

# Start Jupyter Lab
uv run --python .venv jupyter lab

# Launch a Streamlit app
uv run --python .venv streamlit run app.py
```

### Option 2: Manual Environment Activation

**Linux/macOS:**
```bash
source .venv/bin/activate
python your_script.py
deactivate  # when done
```

**Windows (Git Bash/PowerShell):**
```bash
source .venv/Scripts/activate
python your_script.py
deactivate  # when done
```

## 📁 Project Structure

After running the setup script, your directory will contain:

```
your-project/
├── setup_edge_ai_env.sh    # Setup script
├── requirements.txt        # Generated dependencies file
├── .venv/                  # Virtual environment (created by script)
│   ├── bin/               # (Linux/macOS) or Scripts/ (Windows)
│   ├── lib/
│   └── ...
└── README.md              # This file
```

## 🔍 Verification

The script automatically verifies key installations. You should see output like:

```
✅ Key packages verified:
  - Ultralytics: 8.x.x
  - ONNX: 1.x.x
  - ONNX Runtime: 1.x.x
  - PyTorch: 2.x.x
  - TensorFlow: 2.x.x
  - FastAPI: 0.x.x
```

## 💡 Example Projects You Can Build

With this environment, you can immediately start working on:

- **Object Detection**: Using YOLO models with Ultralytics
- **Image Classification**: With PyTorch or TensorFlow
- **NLP Applications**: Using Transformers and Hugging Face models
- **Model Optimization**: Converting models to ONNX format
- **API Deployment**: Creating REST APIs with FastAPI
- **Interactive Demos**: Building Streamlit or Gradio applications
- **Audio Processing**: Speech recognition and audio analysis

## 🛠️ Troubleshooting

### Common Issues

**Permission Errors:**
```bash
# Don't run with sudo - use regular user permissions
# If you see permission errors, try:
chmod +x setup_edge_ai_env.sh
```

**UV Installation Failed:**
```bash
# Manually install UV:
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

**Python 3.12 Not Found:**
The script uses UV's built-in Python management, which automatically downloads Python 3.12 if needed.

**Package Installation Errors:**
```bash
# Clean reinstall:
rm -rf .venv requirements.txt
./setup_edge_ai_env.sh
```

**Import Errors:**
```bash
# Verify you're using the correct environment:
which python  # Should point to .venv/bin/python

# Or use UV run:
uv run --python .venv python -c "import torch; print(torch.__version__)"
```

### Windows-Specific Issues

- Use Git Bash, PowerShell, or WSL for best compatibility
- Ensure Windows Defender isn't blocking script execution
- Some packages may require Microsoft Visual C++ Build Tools

## 🔄 Updating Dependencies

To update packages in your environment:

```bash
# Update all packages
uv pip install --python .venv --upgrade -r requirements.txt

# Update specific package
uv pip install --python .venv --upgrade torch
```

## 🎮 Getting Started Examples

### Quick YOLO Object Detection Test
```python
# test_yolo.py
from ultralytics import YOLO
import cv2

# Load a model
model = YOLO('yolov8n.pt')  # Will download automatically

# Run inference on an image
results = model('https://ultralytics.com/images/bus.jpg')
results[0].show()  # Display results
```

### Simple FastAPI Server
```python
# api_server.py
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Edge AI API is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Run with:
```bash
uv run --python .venv python api_server.py
```

## 📚 Additional Resources

- [UV Documentation](https://docs.astral.sh/uv/)
- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 🤝 Contributing

To modify the environment:

1. Edit `requirements.txt` to add/remove packages
2. Re-run the setup script or use `uv pip install`
3. Test your changes with the verification script

## 📄 License

This setup script is provided as-is for educational and development purposes. Individual packages have their own licenses.

---

**Happy Edge AI Development! 🚀**