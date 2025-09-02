# TinyDiffusion

<div align="center">
  <img src="tinydiffusion_icon.png" alt="tinydiffusion" width="120" height="120">

  <p><em>Benchmarking and Optimized Stable Diffusion for Edge Devices.</em></p>

  <p>
    <a href="https://github.com/DivyenduDutta/TinyDiffusion/blob/master/LICENSE"><img src="https://img.shields.io/github/license/DivyenduDutta/TinyDiffusion?style=flat-square" alt="License"></a>
    <a href="https://github.com/ambv/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black"></a>
  </p>

  <p>
    <a href="#introduction">Introduction</a> •
    <a href="#setup">Setup</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#results">Results</a> •
    <a href="#analysis">Analysis</a>
  </p>
</div>

## Introduction

This repository benchmarks Stable Diffusion UNet inference performance across different runtimes. The focus is on comparing:

- Native PyTorch UNet (GPU) execution
- ONNXRuntime UNet (GPU) execution
- ONNXRuntime UNet (CPU) execution

The goal is to understand the trade-offs in inference speed, CPU/GPU memory usage, and runtime stability when exporting Stable Diffusion components to ONNX and running them with onnxruntime.

### Features

**ONNX Export**: Export the Stable Diffusion UNet model from Hugging Face’s diffusers library into an ONNX graph.

**Flexible Inference**: Run inference on CPU or GPU with onnxruntime or fall back to native PyTorch.

**Benchmarking Suite**: Collect detailed metrics including:
- Average inference time & standard deviation
- CPU memory usage (Resident Set Size)
- GPU memory allocation

**Results Logging**: Save benchmarking results to CSV, with the ability to append new results across runs.

**Visualization**: Generate plots comparing performance across backends for quick insights.

Why ONNX?

[ONNX](https://onnx.ai/) allows exporting deep learning models into a framework-agnostic format. With onnxruntime, models can run on multiple backends (CPU, CUDA, TensorRT, DirectML, etc.) without depending on PyTorch. While this repo shows that ONNX on CPU can be useful for portability, we also observe that PyTorch often outperforms ONNX on GPU for Stable Diffusion UNet inference.

## Setup

Add the project root ie, Folder containing this README to PYTHONPATH whichever way you want. One way would be to create a .env and write the following in it
```
PYTHONPATH=\full\path\to\projectroot
```
And place this .env file in the project root. Works for VS Code.

Another option would be to run `$env:PYTHONPATH = \full\path\to\projectroot` in powershell to set the env variable and then run the scripts.

Install pytorch, torchvision via `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118` - Conda doesnt install GPU version on Windows.

### Sanity

Before committing changes run `pre-commit run --all-files` or `pre-commit run --file <file1>, <file2> ...`
## Quick Start

### Generating Benchmarks

Run
```bash
notebooks/baseline_generation.ipynb
```
to benchmark the Pytorch version of Stable Diffusion's UNet from HuggingFace.


Run
```python
python tinydiffusion/src/onnx_export.py
```
to export the UNet to an ONNX graph.


Then run
```bash
notebooks/onnxruntime_generation.ipynb
```
to benchmark the ONNXRuntime version of UNet on GPU and CPU.

### Visualizing

Once all the benchmarking results populated in `results/benchmarks/benchmark_results.csv`, run
```python
python tinydiffusion/src/benchmark_visualizer.py
```
to generate the visualization plots for comparison.

## Results

<img src="https://github.com/DivyenduDutta/TinyDiffusion/blob/master/results/benchmarks/benchmark_comparison.png" alt="Results">

## Analysis

For analysis see [this](https://github.com/DivyenduDutta/TinyDiffusion/blob/master/Analysis.md)
