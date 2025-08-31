##### fp16 vs fp32

While doing `pipe = StableDiffusionPipeline.from_pretrained(model_id, variant="fp16", torch_dtype=torch.float16 if device=="cuda" else torch.float32)` why do we set it to fp16 if its on GPU vs fp32 otherwise?

- Speed
    - Modern NVIDIA GPUs (esp. with Tensor Cores, e.g. RTX, A100, etc.) are highly optimized for half precision (FP16). It can cut memory bandwidth and compute time almost in half.

- Memory savings
    - FP16 uses 2 bytes per number instead of 4 (FP32). Stable Diffusion is huge, so halving VRAM footprint saves OOM issues.

---

### Metrics

#### Stable Diffusion on Nvidia Geforce GTX 1070 GPU

Average inference time: 28.63s ± 3.09s

Average CPU memory usage: 890.11MB ± 59.32MB

Average GPU memory usage: 2486.43MB ± 0.00MB
