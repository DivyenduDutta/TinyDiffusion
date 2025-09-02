### Metrics

see `results/benchmarks/benchmark_results.csv`

GPU - Nvidia Geforce GTX 1070 GPU
CPU - Intel Core i7-8750H

#### Stable Diffusion UNet on GPU

``` bash
INFO - UNet Inference time: 0.52s
INFO - UNet Inference time: 0.15s
INFO - UNet Inference time: 0.26s
INFO - UNet Inference time: 0.26s
INFO - UNet Inference time: 0.26s
INFO - UNet Inference time: 0.26s
INFO - UNet Inference time: 0.24s
INFO - UNet Inference time: 0.28s
INFO - UNet Inference time: 0.24s
INFO - UNet Inference time: 0.27s

Average inference time: 0.27s ± 0.09s

Average CPU memory usage: 889.63MB ± 0.00MB

Average GPU memory usage: 2511.14MB ± 0.00MB
```

---

#### Stable Diffusion UNet via ONNX Runtime on GPU

```bash
INFO - ONNXRuntime Inference time: 1.82s
INFO - ONNXRuntime Inference time: 1.59s
INFO - ONNXRuntime Inference time: 1.59s
INFO - ONNXRuntime Inference time: 1.60s
INFO - ONNXRuntime Inference time: 1.59s
INFO - ONNXRuntime Inference time: 1.59s
INFO - ONNXRuntime Inference time: 1.58s
INFO - ONNXRuntime Inference time: 1.58s
INFO - ONNXRuntime Inference time: 1.60s
INFO - ONNXRuntime Inference time: 1.58s

Average inference time: 1.61s ± 0.07s

Average CPU memory usage: 1390.87MB ± 0.03MB

Average GPU memory usage: 1355.65MB ± 0.00MB

```

---

#### Stable Diffusion UNet via ONNX Runtime on CPU

```bash
INFO - ONNXRuntime Inference time: 8.94s
INFO - ONNXRuntime Inference time: 9.48s
INFO - ONNXRuntime Inference time: 7.13s
INFO - ONNXRuntime Inference time: 7.42s
INFO - ONNXRuntime Inference time: 7.06s
INFO - ONNXRuntime Inference time: 6.98s
INFO - ONNXRuntime Inference time: 6.91s
INFO - ONNXRuntime Inference time: 7.57s
INFO - ONNXRuntime Inference time: 6.90s
INFO - ONNXRuntime Inference time: 6.98s

Average inference time: 7.54s ± 0.92s

Average CPU memory usage: 4968.51MB ± 344.71MB

```
