---
marp: true
theme: usa-theme
paginate: true
---

<!-- _class: title -->

# Converting & Preparing Models for Edge Deployment
## Week 1 - Session 2
### Deploying Edge AI
#### University of South Alabama

<!-- 
Speaker Notes:
- Welcome to our second session on Deploying Edge AI
- Today we'll focus on the crucial process of taking models from development frameworks and preparing them for efficient edge deployment
- We'll cover conversion workflows, common challenges, and how to evaluate model performance
- By the end of this session, you'll understand how to convert models to ONNX and begin assessing their suitability for edge deployment
-->

---

<!-- _class: section -->

# Converting Models to ONNX Format

<!-- 
Speaker Notes:
- In this first section, we'll explore how to convert models from various frameworks to the ONNX format
- ONNX (Open Neural Network Exchange) provides a unified representation for different frameworks
- This standardization is key for deploying models across diverse hardware and software platforms
- We'll look at the conversion process for PyTorch, TensorFlow, and other popular frameworks
-->

---

# Popular Model Frameworks for Edge AI

<div class="columns">
<div>

## PyTorch
- Widely used in research
- Dynamic computation graph
- Easy to debug and modify
- TorchScript for production

## TensorFlow/Keras
- Production-oriented
- Static computation graph
- TensorFlow Lite for mobile/edge
- SavedModel format

</div>
<div>

## JAX
- Functional programming approach
- XLA compilation support
- Growing popularity in research
- Strong numerical computing performance

## Domain-Specific Frameworks
- Hugging Face Transformers
- Ultralytics YOLO
- MediaPipe
- TensorFlow.js

</div>
</div>

<!-- 
Speaker Notes:
- These are the most common frameworks we'll be converting FROM when preparing models for edge deployment
- PyTorch is exceptionally popular in research contexts, offering flexibility and ease of use
- TensorFlow provides strong production support with tools specifically designed for edge deployment
- JAX is gaining traction for its functional approach and excellent performance characteristics
- Domain-specific frameworks like Hugging Face often have their own export tools with ONNX support
- Understanding the source framework is crucial for successful conversion
- Each framework has its own unique features and limitations that will impact the conversion process
- Ask students which frameworks they're most familiar with to gauge class experience
-->

---

# Basic ONNX Conversion Workflow

```python
# PyTorch to ONNX Example
import torch
import torch.onnx

# 1. Load the trained model
model = YourModelClass()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()  # Set to inference mode

# 2. Create dummy input with the correct shape
dummy_input = torch.randn(1, 3, 224, 224)

# 3. Export to ONNX
torch.onnx.export(
    model,               # model being run
    dummy_input,         # model input
    "model.onnx",        # output file
    export_params=True,  # store the trained parameter weights
    opset_version=12,    # the ONNX version to use
    do_constant_folding=True,  # optimization
    input_names=['input'],     # the model's input names
    output_names=['output'],   # the model's output names
    dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                  'output': {0: 'batch_size'}}
)
```

<!-- 
Speaker Notes:
- This slide shows the basic workflow for converting a PyTorch model to ONNX
- First, we load the trained model and set it to inference mode with model.eval()
- Next, we create a dummy input tensor that matches the expected input shape
- The torch.onnx.export function handles the actual conversion process
- Key parameters include:
  - export_params: ensures weights are included in the ONNX file
  - opset_version: defines which ONNX operations are available (higher is newer)
  - do_constant_folding: an optimization that pre-computes constant expressions
  - input_names/output_names: human-readable names for model inputs/outputs
  - dynamic_axes: allows for flexible dimensions like variable batch sizes
- This basic pattern applies across frameworks, though syntax differs
- Common issues include tensor shape mismatches and unsupported operations
-->

---

# TensorFlow to ONNX Conversion

```python
# TensorFlow to ONNX Example
import tensorflow as tf
import tf2onnx
import onnx

# 1. Load the TensorFlow model
model = tf.keras.models.load_model('model.h5')

# 2. Convert the model to ONNX
model_proto, _ = tf2onnx.convert.from_keras(
    model, 
    opset=12,
    input_signature=(tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),),
    output_path="model.onnx"
)

# 3. Save the ONNX model
onnx.save(model_proto, "model.onnx")

# Alternatively, use the command line:
# python -m tf2onnx.convert --saved-model tensorflow_model_dir --output model.onnx
```

<!-- 
Speaker Notes:
- For TensorFlow models, we typically use the tf2onnx package
- The conversion process starts with loading your TensorFlow/Keras model
- We then use tf2onnx.convert.from_keras to convert the model
- Key parameters include:
  - opset: ONNX version (similar to PyTorch's opset_version)
  - input_signature: defines input tensor shapes and types
  - output_path: where to save the converted model
- Note that TensorFlow uses NHWC format (batch, height, width, channels) while ONNX typically uses NCHW
- The converter handles this transformation automatically in most cases
- The command-line alternative is often simpler for SavedModel format models
- Common challenges include custom operations, control flow, and dynamic shapes
- For TensorFlow Lite models, a different conversion path might be necessary
-->

---

# Converting Hugging Face Models

```python
# Hugging Face Transformers to ONNX Example
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.onnx import export

# 1. Load model and tokenizer
model_id = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Export to ONNX using the built-in export function
export(
    preprocessor=tokenizer,
    model=model,
    opset=12,
    output=f"{model_id}.onnx"
)

# 3. Load and verify the ONNX model
import onnx
onnx_model = onnx.load(f"{model_id}.onnx")
onnx.checker.check_model(onnx_model)
```

<!-- 
Speaker Notes:
- Hugging Face provides direct ONNX export support for its Transformers models
- The transformers.onnx.export function handles the conversion process
- This function manages the complexity of exporting transformer architectures correctly
- We provide both the model and tokenizer to ensure proper input preprocessing
- The export function creates a pipeline that includes both tokenization and inference
- This approach works for a wide range of models: BERT, GPT, T5, etc.
- For transformer models, proper handling of dynamic sequence lengths is crucial
- The resulting ONNX model can be directly loaded into ONNX Runtime
- This simplified workflow hides many of the complexities of transformer architecture
- For more control, you can export just the model component without preprocessing
- Common pitfalls include missing the preprocessing steps or incorrect sequence handling
-->

---

# ONNX Conversion Options

<div class="highlight-box">

## Key Conversion Parameters
- **Opset Version**: ONNX operator set version (e.g., 12, 15, 17)
- **Input/Output Names**: Custom identifiers for model interfaces
- **Input Shapes**: Fixed or dynamic dimensions
- **Precision**: FP32, FP16, or mixed precision
- **Graph Optimizations**: Constant folding, shape inference
- **Target Platform**: CPU, GPU, or specific hardware
- **Quantization Awareness**: Preserving quantization operations
- **Custom Operations**: Handling model-specific operators

</div>

<!-- 
Speaker Notes:
- These parameters significantly impact the quality and compatibility of your converted model
- Opset version determines which ONNX operators are available - newer isn't always better
- Lower opset versions have wider hardware support but fewer operations
- Input/output naming helps with model inspection and debugging
- Dynamic input shapes (using None or -1) allow flexibility but may reduce optimization potential
- Precision selection affects both model size and inference speed
- FP16 can be significantly faster on compatible hardware, but with some accuracy loss
- Graph optimizations like constant folding can greatly improve inference speed
- Target platform may affect how operations are represented
- Quantization awareness is critical if you're working with quantized models
- Custom operations need special handling - either through conversion to standard ops or custom implementations
- Always check the documentation for your specific framework's conversion tool
-->

---

<!-- _class: section -->

# Common Conversion Challenges

<!-- 
Speaker Notes:
- In this section, we'll discuss the common challenges you'll face when converting models to ONNX
- Converting a model is rarely as straightforward as the basic examples suggest
- Modern deep learning models often include operations or structures that don't map cleanly to ONNX
- Understanding these challenges helps you troubleshoot conversion issues
- We'll look at both framework-specific and general conversion problems
- Most importantly, I'll show you strategies for addressing these challenges
-->

---

# Framework-Specific Challenges

<div class="columns">
<div>

## PyTorch Challenges
- Dynamic control flow (if/while)
- Custom C++ extensions
- TorchScript compatibility
- Dynamic shape handling
- Custom autograd functions
- JIT tracing vs scripting
- Hooks and stateful operations

</div>
<div>

## TensorFlow Challenges
- Keras custom layers
- TF custom operations
- Control flow operations
- SavedModel vs Keras formats
- TF-specific optimizations
- GradientTape operations
- Variable batch dimensions

</div>
</div>

<!-- 
Speaker Notes:
- Each framework has its own set of conversion challenges stemming from their architectural differences
- PyTorch challenges often involve its dynamic nature:
  - Control flow within models (if/else statements, loops) may not export cleanly
  - Custom C++ extensions need special handling or reimplementation
  - Dynamic shapes can be particularly problematic without proper tracing
  - PyTorch's flexible autograd doesn't always translate to static graphs
  - JIT tracing captures a single execution path, potentially missing branches
- TensorFlow challenges tend to relate to its abstractions:
  - Custom Keras layers need careful implementation to be convertible
  - TensorFlow-specific optimizations may not have ONNX equivalents
  - The distinction between SavedModel and Keras format affects conversion
  - Variable dimensions require explicit handling during export
- The key is to recognize these issues early in your model development process
- Design your models with export compatibility in mind when targeting edge deployment
-->

---

# General Conversion Challenges

<div class="alert-box">

## Common Issues Across Frameworks
- **Unsupported Operations**: Operations without ONNX equivalents
- **Complex Architectures**: Nested models or ensembles
- **Dynamic Shapes**: Variable-length sequences or dimensions
- **Custom Layers**: Non-standard implementations
- **Large Models**: Memory limitations during conversion
- **Recurrent Structures**: RNNs, LSTMs with complex behaviors
- **Preprocessing Integration**: Input normalization, tokenization
- **Post-processing**: Output decoding, non-maximum suppression

</div>

<!-- 
Speaker Notes:
- These challenges appear regardless of which framework you're converting from
- Unsupported operations are among the most common issues - not everything has an ONNX equivalent
- Complex architectures like model ensembles may need to be exported as separate models
- Dynamic shapes are particularly challenging for edge deployment where static shapes are often preferred
- Custom layers typically need reimplementation or conversion to a sequence of standard operations
- Memory limitations during conversion can be severe - large models may require special handling
- Recurrent structures like LSTMs can have framework-specific behaviors that don't translate cleanly
- Preprocessing and post-processing steps are often overlooked but critical for correct inference
- For example, image normalization or text tokenization should be consistent between training and deployment
- Non-maximum suppression for object detection is particularly problematic to convert
- Understanding these challenges helps you design more portable models from the start
- You'll encounter these issues repeatedly throughout the course, so developing strategies is essential
-->

---

# Handling Unsupported Operations

```python
# Registering a custom ONNX operator
from torch.onnx.symbolic_registry import register_op

# 1. Define how your custom op should be exported
def my_custom_op_exporter(g, input_tensor, attr1, attr2):
    # Create an ONNX node that implements your operation
    return g.op("CustomDomain::MyCustomOp", 
                input_tensor, 
                attr1_i=attr1, 
                attr2_f=attr2)

# 2. Register the exporter for your custom op
register_op("my_custom_op", my_custom_op_exporter, "", 9)

# 3. Alternative: Rewrite model to use only supported ops
def replace_custom_op_with_standard_ops(model):
    # Implementation that modifies the model architecture
    # to use only standard operations
    pass
```

<!-- 
Speaker Notes:
- There are two main approaches to handling unsupported operations
- First approach: Register custom operators with the ONNX conversion system
  - This requires defining how your operation maps to ONNX primitives
  - You need to specify the operation's domain, name, and attributes
  - This is framework-specific - the example shows PyTorch's approach
  - Be aware that custom ops may require special runtime support
- Second approach: Rewrite your model to avoid custom operations
  - Replace custom ops with equivalent sequences of standard operations
  - This might require reimplementing layers or changing model architecture
  - More portable but can be labor-intensive
- A third option (not shown) is to separate your model at the unsupported operation
  - Export what you can to ONNX and handle the rest in your application code
  - This creates a more complex deployment pipeline but might be necessary
- The best strategy depends on your specific use case and deployment constraints
- For edge AI, favoring standard operations usually leads to better portability
-->

---

# Converting Complex Architectures

<div class="columns">
<div>

## Multi-Stage Models
```python
# Export model stages separately
backbone_onnx = export_to_onnx(
    backbone_model, "backbone.onnx")
head_onnx = export_to_onnx(
    head_model, "head.onnx")

# In deployment, run sequentially
def inference(input_data):
    backbone_output = ort_run(
        backbone_onnx, input_data)
    final_output = ort_run(
        head_onnx, backbone_output)
    return final_output
```

</div>
<div>

## Model Ensembles
```python
# Export each model in ensemble
model_paths = []
for i, model in enumerate(ensemble):
    path = f"model_{i}.onnx"
    export_to_onnx(model, path)
    model_paths.append(path)
    
# In deployment, combine outputs
def ensemble_inference(input_data):
    outputs = []
    for path in model_paths:
        output = ort_run(path, input_data)
        outputs.append(output)
    return average_outputs(outputs)
```

</div>
</div>

<!-- 
Speaker Notes:
- Complex architectures often require decomposition into simpler components
- For multi-stage models (like encoder-decoder architectures):
  - Export each stage as a separate ONNX model
  - In your deployment code, connect the stages by passing outputs as inputs
  - This approach simplifies conversion and can improve memory usage
  - You may lose some optimization opportunities across model boundaries
- For ensemble models:
  - Export each ensemble member separately
  - Implement the ensembling logic (averaging, voting, etc.) in your application
  - This gives you flexibility in how you deploy and schedule ensemble inference
- The same principle applies to other complex architectures:
  - GANs (generator and discriminator)
  - Teacher-student models
  - Multi-modal models with separate processing paths
- This approach lets you optimize each component separately
- It also allows for more flexible deployment options (like skipping components when not needed)
- Remember that the connections between components need careful handling for correct results
-->

---

# Working with Dynamic Shapes

<div class="highlight-box">

## Strategies for Handling Dynamic Dimensions

### Fully Dynamic Approach
- Export with dynamic axes (PyTorch: `dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'}}`)
- Allows flexibility but may reduce optimization opportunities

### Fixed Shape Approach
- Export with fixed dimensions for all inputs
- Highest performance but least flexible

### Multiple Fixed Shapes
- Export several versions for common input sizes
- Select appropriate model at runtime

### Shape Optimization
- Determine optimal fixed shapes for your deployment scenario
- Consider batching strategy and typical input distributions

</div>

<!-- 
Speaker Notes:
- Dynamic shapes are particularly challenging for edge deployment
- The fully dynamic approach:
  - Most flexible, accepting varying input dimensions
  - Some optimizations may not be applicable with dynamic shapes
  - Runtime performance can be lower than with fixed shapes
  - Memory allocation may be less efficient
- The fixed shape approach:
  - Highest optimization potential and most predictable performance
  - Input must be resized/padded to match the expected shape
  - This is often preferred for edge deployment with severe constraints
- Multiple fixed shapes strategy:
  - Export several versions optimized for different shapes
  - Select the appropriate model based on incoming data
  - Increases deployment complexity but can balance flexibility and performance
- Shape optimization:
  - Analyze your actual deployment data to identify common shapes
  - Choose fixed shapes that minimize preprocessing overhead
  - Consider batching strategy for throughput vs. latency tradeoffs
- For edge AI, fixed shapes often win on performance, but consider your specific requirements
-->

---

<!-- _class: section -->

# Understanding ONNX Model Structure

<!-- 
Speaker Notes:
- Now that we've covered conversion, let's understand what's inside an ONNX model
- ONNX models follow a specific graph-based structure that's important to understand
- This knowledge helps with debugging conversion issues and optimizing models
- We'll explore the key components that make up an ONNX model file
- Understanding this structure is critical for effective model optimization later
-->

---

# ONNX Model Components

<div class="columns">
<div>

## Key Elements
- **Model Metadata**: Version, domain, description
- **Graph**: The computational network
- **Nodes**: Individual operations (Add, Conv, etc.)
- **Inputs/Outputs**: Model interface definitions
- **Initializers**: Constant values (weights, biases)
- **Value Info**: Tensor shape and type information
- **Attributes**: Operation-specific parameters

</div>
<div>

## Example Node
```python
# A convolutional node in ONNX
Node(
  name="Conv_0",
  op_type="Conv",
  inputs=["input.1", "weight.1", "bias.1"],
  outputs=["5"],
  attributes={
    "dilations": [1, 1],
    "kernel_shape": [3, 3],
    "pads": [1, 1, 1, 1],
    "strides": [1, 1]
  }
)
```

</div>
</div>

<!-- 
Speaker Notes:
- The ONNX format represents models as computational graphs
- Model metadata includes version information and documentation
- The graph is the core component containing the model's computational structure
- Nodes represent individual operations like convolutions, activations, etc.
- Each node has:
  - Inputs: Tensors consumed by the operation
  - Outputs: Tensors produced by the operation
  - Attributes: Parameters that configure the operation
- Initializers are constant values like weights and biases
- Value info provides shape and type information for intermediate tensors
- The example shows a typical convolutional node:
  - Takes input, weight, and bias tensors
  - Specifies kernel size, padding, and stride through attributes
  - Produces a single output tensor
- This graph structure is what makes ONNX models portable across frameworks
- It's also what enables framework-independent optimizations
-->

---

# Inspecting ONNX Models

```python
import onnx
import netron
import onnxruntime as ort

# 1. Basic model inspection
model = onnx.load("model.onnx")
print(f"Model IR version: {model.ir_version}")
print(f"Opset version: {model.opset_import[0].version}")
print(f"Graph inputs: {[i.name for i in model.graph.input]}")
print(f"Graph outputs: {[o.name for o in model.graph.output]}")

# 2. Check model validity
onnx.checker.check_model(model)

# 3. Visualize with Netron (launches browser-based visualization)
netron.start("model.onnx")

# 4. Get basic model metadata with ONNX Runtime
session = ort.InferenceSession("model.onnx")
input_details = session.get_inputs()
output_details = session.get_outputs()
print(f"Input name: {input_details[0].name}, shape: {input_details[0].shape}")
```

<!-- 
Speaker Notes:
- These tools help you understand and debug your converted ONNX models
- The onnx package provides programmatic access to model structure
- You can check IR version, opset version, and input/output configuration
- The onnx.checker.check_model function validates model correctness
- Netron is an essential visual tool for ONNX model inspection
  - It provides an interactive visualization of the model graph
  - You can explore nodes, connections, and tensor shapes
  - Highly recommended for understanding complex models
- ONNX Runtime provides information from the execution perspective
  - Shows what the inference engine actually sees
  - Helps verify input/output specifications
- When debugging conversion issues:
  1. First check model validity with onnx.checker
  2. Visualize with Netron to identify problematic operations
  3. Examine input/output specifications with ONNX Runtime
  4. Trace the error back to the original framework model
- These inspection capabilities are crucial for the optimization work we'll do later
-->

---

# Graph Transformation Tools

```python
import onnx
from onnx import shape_inference
from onnxruntime.transformers import optimizer

# 1. Update shape information throughout the graph
model = onnx.load("model.onnx")
inferred_model = shape_inference.infer_shapes(model)
onnx.save(inferred_model, "model_with_shapes.onnx")

# 2. Optimize model with ONNX Runtime Transformer tools
# Particularly useful for transformer architecture models
opt_model = optimizer.optimize_model(
    "model.onnx",
    model_type="bert",  # or "gpt2", "t5", etc.
    num_heads=12,
    hidden_size=768,
    optimization_level=99  # Maximum optimization
)
opt_model.save_model_to_file("optimized_model.onnx")

# 3. Optimize with onnxoptimizer package
# pip install onnxoptimizer
import onnxoptimizer
optimized_model = onnxoptimizer.optimize(model)
onnx.save(optimized_model, "onnxoptimizer_model.onnx")
```

<!-- 
Speaker Notes:
- ONNX provides tools for graph transformation and optimization
- Shape inference is crucial for optimization:
  - Propagates shape information through the entire graph
  - Helps identify potential issues with tensor shapes
  - Enables downstream optimizations that depend on shape information
- ONNX Runtime Transformer tools offer specialized optimizations:
  - Designed specifically for transformer architectures like BERT, GPT-2
  - Implements optimizations like attention fusion and layer normalization fusion
  - Can dramatically improve performance of transformer models
  - Requires model-specific parameters like hidden size and number of heads
- The onnxoptimizer package provides general graph optimizations:
  - Constant folding: pre-computing constant expressions
  - Node elimination: removing redundant operations
  - Node fusion: combining operations for better performance
- For edge deployment, these optimizations are often essential
- We'll dive deeper into optimization in Week 2, but basic optimizations should be applied during conversion
- Different optimization tools have different strengths - often applying multiple tools in sequence yields best results
-->

---

<!-- _class: section -->

# Model Profiling and Performance Assessment

<!-- 
Speaker Notes:
- Now that we can convert models, we need to assess their performance
- Profiling helps us identify bottlenecks and optimization opportunities
- For edge deployment, understanding resource requirements is critical
- We'll look at tools and techniques for measuring model performance
- This assessment informs both optimization strategies and deployment decisions
- The metrics we gather now will serve as our baseline for measuring optimization success
-->

---

# Key Performance Metrics

<div class="highlight-box">

## What to Measure
- **Inference Time**: Average and percentile latencies (ms)
- **Memory Usage**: Peak and average consumption (MB)
- **Model Size**: Storage requirements (MB)
- **CPU/GPU Utilization**: Percentage of available compute
- **Power Consumption**: Energy usage (relevant for battery-powered devices)
- **Accuracy**: Precision, recall, or other task-specific metrics
- **Throughput**: Inferences per second (for batch processing)
- **Initialization Time**: Startup delay before first inference

</div>

<!-- 
Speaker Notes:
- These metrics help evaluate a model's suitability for edge deployment
- Inference time is usually the primary concern:
  - Measure both average latency and percentiles (p95, p99)
  - Latency variations can be as important as the average
  - Background processes can cause unexpected spikes
- Memory usage dictates hardware requirements:
  - Peak memory determines minimum device specifications
  - Excessive memory usage can cause device throttling or crashes
- Model size affects storage and loading time:
  - Particularly important for embedded systems with limited storage
  - Also affects over-the-air update feasibility
- CPU/GPU utilization shows efficiency:
  - High utilization may indicate good hardware use
  - But can also cause thermal throttling on mobile devices
- Power consumption is critical for battery-powered devices:
  - Directly impacts device runtime
  - Often overlooked but can be a deployment blocker
- Accuracy vs efficiency tradeoff must be quantified:
  - Measure how optimizations affect model performance
  - Establish acceptable accuracy thresholds
- For server-edge hybrid systems, throughput may matter more than latency
- Initialization time can be significant for on-demand models
-->

---

# Performance Benchmarking Tools

```python
import onnxruntime as ort
import numpy as np
import time
import psutil
import os

# Basic latency measurement
def benchmark_model(model_path, input_shape, num_iterations=100):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # Create random input data
    input_data = np.random.random(input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        session.run(None, {input_name: input_data})
    
    # Measure inference time
    latencies = []
    process = psutil.Process(os.getpid())
    memory_usage = []
    
    for _ in range(num_iterations):
        memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
        start = time.time()
        session.run(None, {input_name: input_data})
        latencies.append((time.time() - start) * 1000)  # ms
    
    return {
        "avg_latency_ms": np.mean(latencies),
        "p95_latency_ms": np.percentile(latencies, 95),
        "p99_latency_ms": np.percentile(latencies, 99),
        "avg_memory_mb": np.mean(memory_usage),
        "peak_memory_mb": np.max(memory_usage),
        "model_size_mb": os.path.getsize(model_path) / (1024 * 1024)
    }
```

<!-- 
Speaker Notes:
- This code demonstrates a basic benchmarking approach for ONNX models
- It measures key performance metrics like latency and memory usage
- The benchmarking process includes:
  - Creating a session with ONNX Runtime
  - Generating appropriate synthetic input data
  - Performing warmup iterations to stabilize performance
  - Collecting metrics over multiple runs for statistical validity
- Important benchmarking practices shown here:
  - Always include warmup runs to avoid cold-start penalties
  - Measure multiple iterations to account for variability
  - Track both average and percentile metrics
  - Monitor memory usage throughout inference
  - Include model size as a deployment consideration
- More sophisticated benchmarking would also include:
  - CPU/GPU utilization monitoring
  - Power consumption measurement (device-specific)
  - Thread utilization and thread-count experiments
  - Thermal impact assessment
- For accurate results, benchmarking should be performed on target hardware
- These metrics serve as your baseline for optimization efforts
- Consistent benchmarking methodology is essential for valid comparisons
-->

---

# Advanced Profiling with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np
import json

# Configure session options for profiling
sess_options = ort.SessionOptions()

# Enable profiling
sess_options.enable_profiling = True
sess_options.profile_file_prefix = "onnxruntime_profile"

# Create session with profiling enabled
session = ort.InferenceSession("model.onnx", sess_options)
input_name = session.get_inputs()[0].name
input_data = np.random.random((1, 3, 224, 224)).astype(np.float32)

# Run the model multiple times
for _ in range(100):
    session.run(None, {input_name: input_data})

# Get the profiling file
profiling_file = session.end_profiling()

# Parse and analyze profiling results
with open(profiling_file, "r") as f:
    profiling_data = json.load(f)

# Find the most time-consuming operations
op_times = {}
for item in profiling_data:
    if "op_name" in item:
        op_name = item["op_name"]
        if op_name not in op_times:
            op_times[op_name] = []
        op_times[op_name].append(item["dur"])

# Show top 5 most expensive operations
for op, times in sorted(op_times.items(), key=lambda x: sum(x[1]), reverse=True)[:5]:
    print(f"Operation: {op}, Average time: {sum(times)/len(times):.2f} ms")
```

<!-- 
Speaker Notes:
- ONNX Runtime provides built-in profiling capabilities that go beyond basic timing
- This example demonstrates how to enable and use ONNX Runtime's profiling
- Key components of advanced profiling:
  - Enabling profiling through session options
  - Running the model multiple times to gather statistics
  - Parsing the profiling output to identify bottlenecks
  - Analyzing operation-level performance
- The profiling output contains detailed timing for each operation in the graph
- This helps identify specific bottlenecks rather than just overall latency
- For example, you might discover that:
  - A specific convolution layer is taking 40% of inference time
  - Memory transfers are causing unexpected delays
  - Certain operations aren't being optimized as expected
- This operation-level insight drives targeted optimization:
  - Focus efforts on the most expensive operations
  - Consider restructuring the model to avoid bottlenecks
  - Apply operation-specific optimizations
- The profiling data can be visualized with tools like Chrome Tracing
- For edge deployment, identifying the critical path is essential
- This detailed profiling is a key step before applying the optimizations we'll learn in Week 2
-->

---

# Initial Resource Assessment

<div class="columns">
<div>

## Hardware Targeting
- **CPU Limitations**
  - Core count and frequency
  - SIMD capabilities (AVX, NEON)
  - Cache size constraints
  - Thread scheduling concerns

- **Memory Limitations**
  - Total available RAM
  - Working set restrictions
  - Shared vs dedicated memory
  - Memory bandwidth

</div>
<div>

## Software Considerations
- **OS Constraints**
  - Threading limitations
  - Process priorities
  - Background services
  - Power management

- **Deployment Factors**
  - Startup time requirements
  - Battery impact assessment
  - Thermal constraints
  - Concurrent applications

</div>
</div>

<!-- 
Speaker Notes:
- Resource assessment connects performance metrics to deployment constraints
- CPU limitations impact model structure and optimization:
  - Core count affects parallelization strategy
  - SIMD capabilities enable vectorized operations
  - Cache size determines optimal tensor blocking
  - Thread scheduling can cause performance variability
- Memory limitations define feasible model size:
  - Total RAM limits model size and batch size
  - Working set restrictions may require model splitting
  - Shared memory systems (like mobile GPUs) have unique constraints
  - Memory bandwidth often becomes the bottleneck, not compute
- OS constraints affect deployment architecture:
  - Threading limitations may restrict parallelization
  - Process priorities impact real-world performance
  - Background services can cause unexpected interference
  - Power management may throttle performance
- Other deployment factors to consider:
  - Startup time requirements may affect model loading approach
  - Battery impact determines feasibility for mobile deployment
  - Thermal constraints can cause unexpected throttling
  - Concurrent applications may compete for resources
- This assessment defines your optimization targets for the coming weeks
- The goal is to identify the specific constraints that will guide your optimization efforts
-->

---

# Deployment Environment Considerations

<div class="alert-box">

## Environment Factors Affecting Performance
- **Hardware Diversity**: Target devices may have varying capabilities
- **OS and Runtime Versions**: Different platforms have different ONNX Runtime implementations
- **Form Factor Constraints**: Thermal limitations, power budgets
- **Connectivity Requirements**: Online vs. offline operation
- **User Interaction Patterns**: Response time expectations
- **Concurrent Applications**: Resource competition
- **Deployment Mechanism**: App store, embedded, web-based
- **Update Strategy**: How models will be updated post-deployment
- **Privacy Requirements**: On-device vs. server processing

</div>

<!-- 
Speaker Notes:
- The deployment environment shapes both conversion and optimization strategies
- Hardware diversity requires careful planning:
  - Consider the range of devices you need to support
  - Test on lowest-spec target hardware
  - Plan for graceful degradation or tiered models
- OS and runtime variations affect compatibility:
  - ONNX Runtime versions may have different optimizations
  - Some operations may behave differently across platforms
  - Test on representative environments
- Physical constraints impact sustained performance:
  - Mobile devices throttle under thermal pressure
  - Power considerations limit complexity for battery-powered devices
  - Form factor may restrict memory or cooling
- User experience sets performance requirements:
  - Interactive applications typically need sub-100ms responses
  - Batch processing may prioritize throughput over latency
  - Consider the full pipeline, not just model inference
- Real-world device usage includes:
  - Multiple applications running concurrently
  - Variable resource availability
  - Unpredictable background processes
- Deployment mechanism affects packaging:
  - App store size limits may constrain model size
  - Web deployment has different optimization priorities
  - Embedded systems may have custom runtimes
- Update strategy influences model design:
  - How will you deliver model updates?
  - Is incremental learning required?
  - What's the update frequency?
- Privacy considerations may mandate on-device processing
-->

---

# Week 1 GitHub Project Requirements

<div class="highlight-box">

## Project Milestone - Phase 1
- **Model Selection**: Choose two models relevant to your project
- **Conversion Implementation**: Create scripts that convert models to ONNX format
- **Baseline Profiling**: Measure and document inference time, memory, and model size
- **GitHub Documentation**: 
  - Detailed README with model architecture descriptions
  - Conversion workflow documentation
  - Baseline performance metrics
  - Initial deployment considerations

</div>

<!-- 
Speaker Notes:
- For this week's project milestone, you'll apply what we've learned about conversion and profiling
- Start by selecting two models relevant to your chosen project area:
  - Choose models with different characteristics (size, architecture, etc.)
  - Consider models that represent different aspects of your application
  - Ensure they're suitable for edge deployment (not excessively large)
- Implement conversion pipelines for your selected models:
  - Create reproducible scripts that handle the conversion process
  - Document any challenges and how you addressed them
  - Ensure the conversion preserves model accuracy
- Establish baseline performance metrics:
  - Measure inference time, memory usage, and model size
  - Document the testing environment clearly
  - Create visualizations of key performance characteristics
  - Compare performance between your selected models
- Set up your GitHub repository with proper documentation:
  - Well-structured README with project overview
  - Clear documentation of the conversion workflow
  - Baseline metrics with interpretation
  - Initial thoughts on deployment considerations
- This milestone establishes the foundation for the optimization work in Week 2
- Focus on thorough documentation of both successes and challenges
- Remember that this baseline will be your reference point for measuring optimization impact
-->

---

# Next Steps

<div class="columns">
<div>

## In This Session
- Converting models to ONNX
- Understanding ONNX structure
- Performance profiling
- Resource assessment

</div>
<div>

## Coming in Week 2
- Model architecture optimization
- Knowledge distillation
- Pruning strategies
- Quantization techniques
- ONNX-specific optimizations

</div>
</div>

<!-- 
Speaker Notes:
- Today we learned the fundamental processes for converting models to ONNX format
- We covered:
  - Converting from various frameworks (PyTorch, TensorFlow, Hugging Face)
  - Understanding and addressing common conversion challenges
  - Exploring the structure of ONNX models
  - Profiling models to establish performance baselines
  - Assessing resource requirements for deployment
- These skills form the foundation for our optimization work
- In Week 2, we'll build on this by exploring:
  - How to optimize model architectures specifically for edge deployment
  - Knowledge distillation to create smaller, faster models
  - Pruning strategies to remove unnecessary weights
  - Quantization techniques to reduce precision requirements
  - Advanced ONNX-specific optimizations
- For your project work this week:
  - Apply the conversion techniques we discussed
  - Document any challenges you encounter
  - Establish thorough performance baselines
  - Begin thinking about which optimization strategies might be most effective
- Remember that successful edge deployment starts with proper model conversion and assessment
- Questions about any of the topics we covered today?
-->

---

<!-- _class: title -->

# Questions?
## Week 1 - Session 2
### Deploying Edge AI

<!-- 
Speaker Notes:
- We've covered a lot of ground today in our exploration of model conversion and preparation
- Converting models to ONNX is a critical first step in the edge deployment process
- The techniques we've discussed lay the groundwork for the optimization work to come
- Take time this week to experiment with converting models for your projects
- Document challenges and solutions - this documentation is valuable for both your learning and project evaluation
- Feel free to ask questions now or reach out via email with specific conversion challenges
- In our next session, we'll begin exploring optimization techniques to improve model performance
- For your project milestone, focus on establishing a solid conversion workflow and baseline metrics
- Remember that these baseline measurements will be your reference point throughout the course
-->