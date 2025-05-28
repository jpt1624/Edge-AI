---
marp: true
theme: usa-theme
paginate: true
---

<!-- _class: title -->

# Deploying Edge AI
## 5-Week Master's Level Course
### Dr. Ali Haidous
#### University of South Alabama

---

<!-- _class: section -->

# Course Overview

---

# Course Description

This course introduces Master's students to the principles and practices of deploying advanced AI models at the edge. 

Over 5 intensive weeks, students will learn how to:
- Optimize, convert, and deploy popular generative AI models
- Implement computer vision systems on resource-constrained devices
- Deploy speech recognition capabilities using ONNX and CPUExecutionProvider
- Create practical edge AI applications without specialized hardware

<div class="highlight-box">
Emphasis on practical approaches that leverage standard CPU capabilities found in most computing environments.
</div>

---

# Learning Format

- **Duration**: 5 weeks
- **Weekly Time Commitment**: 5 hours per week
- **Session Structure**: Two 2.5-hour sessions per week
- **Format**: 
  - 75 minutes of instruction 
  - 75 minutes of guided project work

<div class="alert-box">
Each session includes both theoretical instruction and hands-on implementation to ensure continuous progress toward the final project deliverable.
</div>

---

# Prerequisites

<div class="highlight-box">

Students should have:
- Master's level understanding of deep learning principles
- Strong programming experience in Python
- Experience with at least one deep learning framework (PyTorch, TensorFlow, JAX)
- Familiarity with model training and evaluation
- Understanding of performance profiling and optimization concepts
- Previous exposure to computer vision, NLP, or speech processing
- Experience with Python package management and virtual environments
- Basic Git/GitHub knowledge (branching, committing, pull requests)

</div>

---

<!-- _class: section -->

# Weekly Schedule

---

# Week 1: Edge AI Fundamentals & ONNX Runtime

## Session 1: Introduction to Edge AI & ONNX Ecosystem
- What is Edge AI and why it matters
- Challenges of deploying AI at the edge
- Introduction to ONNX as a deployment solution
- ONNX Runtime and its execution providers
- Project kickoff: Select individual project

## Session 2: Converting & Preparing Models for Edge Deployment
- Converting models from various frameworks to ONNX
- Understanding ONNX model structure
- Model profiling and performance benchmarking
- Deployment environment considerations

---

# Week 2: Model Optimization for Edge Deployment

## Session 1: Edge-Optimized Model Architecture
- Model architectures suitable for edge deployment
- Knowledge distillation techniques
- Pruning strategies for neural networks
- Quantization fundamentals
- Measuring accuracy impact of optimizations

## Session 2: Advanced ONNX Optimization Techniques
- ONNX Runtime optimization levels
- Graph-level optimizations in ONNX
- Post-training quantization with ONNX
- Optimizing CPU memory access patterns

---

# Week 3: Deploying Generative AI at the Edge

## Session 1: Edge-Optimized Text Generation Models
- Overview of generative text models deployable at the edge
- Efficient transformers: techniques and architectures
- Optimizing LLMs for edge deployment
- ONNX conversion for generative text models
- KV cache optimization for inference

## Session 2: Edge-Optimized Image Generation Models
- Edge-deployable image generation models
- Converting generative image models to ONNX
- Progressive generation techniques to reduce memory usage
- Memory-efficient attention mechanisms

---

# Week 4: Computer Vision at the Edge

## Session 1: Optimizing Computer Vision Models
- Modern efficient vision architectures (MobileNetV3, EfficientNet)
- Converting vision models from PyTorch/TensorFlow to ONNX
- Efficient image preprocessing pipelines
- Optimizing complex vision tasks (detection, segmentation)

## Session 2: Real-time Computer Vision Applications
- Designing real-time vision applications for edge devices
- Frame processing optimization strategies
- Camera input handling and preprocessing optimization
- Tracking and temporal consistency in edge vision

---

# Week 5: Speech Processing & Multimodal Edge AI

## Session 1: Edge-Optimized Speech Models
- Efficient speech recognition architectures
- Converting speech models to ONNX format
- Optimizing audio preprocessing for edge devices
- Streaming speech recognition approaches

## Session 2: Multimodal Integration & Final Projects
- Combining text, vision, and speech models efficiently
- Resource-aware model orchestration
- Final project integration and testing
- Performance optimization and troubleshooting

---

<!-- _class: section -->

# Sample Projects

---

# Available Project Options

Choose one of the following sample projects:

<div class="columns">
<div>

## Edge AI Assistant
- Desktop-based AI assistant
- Runs entirely on CPU without cloud
- Components:
  - Text Generation
  - Computer Vision
  - Speech Recognition
  - Responsive UI

## Portable Document Intelligence
- Document analysis system
- Offline operation on standard laptops
- Components:
  - Document Detection & OCR
  - Text Understanding
  - Information Extraction
  - On-device Search

</div>
<div>

## Edge Creative Studio
- Content creation tool
- On-device generative AI
- Components:
  - Image Generation
  - Text Assistance
  - Style Transfer
  - Voice Conversion

## Smart Monitoring System
- Privacy-preserving monitoring
- On-device processing
- Components:
  - Person/Activity Detection
  - Audio Analysis
  - Anomaly Detection
  - Incident Reporting

</div>
</div>

---

# Sample Project: Field Research Assistant

<div class="highlight-box">

## Concept
An offline-capable field research tool for scientific data collection and analysis.

## Components
- **Computer Vision**: Species/object identification and counting
- **Image Enhancement**: Lightweight models for improving field photography
- **Speech-to-Text**: Voice note transcription for hands-free documentation
- **Text Generation**: Field reports and data summaries generation

## Learning Focus
- Optimizing classification and detection models for field conditions
- Memory-efficient transcription with limited resources
- Creating battery-aware AI processing pipelines

</div>

---

<!-- _class: section -->

# Project Requirements

---

# The Edge AI Deployment Challenge

<div class="highlight-box">

Students will build a multimodal edge AI application with the following components:

- **Edge Deployment Excellence**: Run efficiently on standard CPU hardware
- **Advanced Model Optimization**: Demonstrate significant size and speed improvements
- **Generative AI Capability**: Implement edge-optimized text or image generation
- **Computer Vision Performance**: Deploy advanced vision capabilities with real-time performance
- **Speech Processing**: Enable efficient speech recognition
- **Resource-Aware Design**: Create intelligent scheduling and resource management
- **Compelling User Experience**: Design interfaces that maintain responsiveness
- **Deployment Versatility**: Ensure consistent performance across various environments
- **Practical Application**: Develop a solution for a real-world use case
- **GitHub Excellence**: Maintain a well-structured, documented repository

</div>

---

# GitHub Repository Requirements

<div class="highlight-box">

## Project Organization
- `/models` - Model conversion and optimization code
- `/benchmark` - Performance testing and comparison utilities
- `/app` - Application code and UI components
- `/notebooks` - Jupyter notebooks documenting experiments
- `/docs` - Comprehensive documentation including setup instructions

## Required Documentation
- Detailed README with project overview and architecture diagram
- Model optimization documentation detailing techniques and results
- Performance benchmarking reports with comparative analysis
- Weekly progress reports documenting challenges and solutions
- Final presentation slides summarizing achievements

</div>

---

# Project Development Journey

<div class="columns">
<div>

## Week 1: Analysis Phase
- Understand edge deployment requirements
- Master ONNX fundamentals
- Establish model conversion workflow
- Set up GitHub repository structure

## Week 2: Optimization Phase
- Apply model architecture modifications
- Implement quantization and pruning
- Optimize neural network operations
- Establish optimization benchmarks

</div>
<div>

## Week 3: Generative Capabilities
- Deploy efficient text/image generation
- Optimize memory usage
- Implement progressive techniques
- Create responsive applications

## Week 4-5: Visual Intelligence & Integration
- Deploy efficient computer vision
- Add optimized speech processing
- Integrate multiple modalities
- Finalize documentation and demo

</div>
</div>

---

# Assessment and Deliverables

<div class="alert-box">

## Evaluation Framework
- **Weekly Milestone Achievements (30%)**: Meeting project milestones
- **Technical Optimization Quality (25%)**: Effectiveness of optimization techniques
- **GitHub Repository Quality (15%)**: Organization and documentation quality
- **System Performance (10%)**: Efficiency on standard CPU hardware
- **Final Demo & Benchmarks (15%)**: Demonstration and performance analysis
- **Documentation & Reproducibility (5%)**: Comprehensive documentation

</div>

---

# Application Domains

Students may choose to specialize in one of these high-impact domains:

<div class="columns">
<div>

- **Healthcare Edge AI**: Portable medical diagnostics and monitoring
- **Field-Deployable AI**: Remote or offline AI solutions
- **Smart Manufacturing**: On-device quality control and monitoring
- **Edge-Based Security**: Privacy-preserving surveillance

</div>
<div>

- **Remote Collaboration**: Edge-enhanced communication tools
- **Mobile Creativity**: On-device generative AI
- **Accessible AI**: Low-resource AI solutions
- **Custom Domain**: (with instructor approval)

</div>
</div>

---

# Resources & Support

<div class="highlight-box">

## Key Documentation
- ONNX Core Documentation: github.com/onnx/onnx
- ONNX Runtime Python API: onnxruntime.ai/docs/api/python/
- ONNX Runtime Optimization: onnxruntime.ai/docs/performance/
- Hugging Face Optimum: huggingface.co/docs/optimum/

## Additional Resources
- Pre-configured Python environments for edge AI development
- Weekly starter templates and optimization examples
- Project structure templates and documentation examples
- Sample project starter repositories

</div>

---

<!-- _class: section -->

# Contact Information

---

# Instructor Contact & Office Hours

<div class="highlight-box">

## Dr. Ali Haidous
- **Email**: ali.haidous@gmail.com
- **Office Hours**: By Appointment
- **Course Website**: southalabama.edu/departments/globalusa/cce/edgeai/

</div>

For technical assistance with course projects, edge AI implementation questions, or additional resources, please visit the [course GitHub repository](https://github.com/ali-haidous/edge-ai) or contact the instructor via email.

---

<!-- _class: title -->

# Thank You!
## Questions?

<div class="alert-box" style="margin-top: 50px; text-align: center;">
University of South Alabama<br>
Deploying Edge AI - Master's Level Course
</div>