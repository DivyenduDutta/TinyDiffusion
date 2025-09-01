"""
This script exports the UNet model of the Stable Diffusion pipeline to ONNX format.
"""

import os
import torch
from diffusers import StableDiffusionPipeline

from tinydiffusion.utils.constants import ModelType, PROMPT
from tinydiffusion.utils.logger import LoggerConfig

LOGGER = LoggerConfig().logger

# ONNX export save path
onnx_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "checkpoints", "onnx", "unet.onnx"
)
os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

# Model cache directory
ROOT_DIR = os.path.dirname(os.getcwd())
model_cache_dir = os.path.join(ROOT_DIR, "checkpoints", "stablediffusion")
os.makedirs(os.path.dirname(model_cache_dir), exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

LOGGER.info(f"Using device: {device}")


def load_sd_pipeline(model_id: str) -> StableDiffusionPipeline:
    """
    Load and return the Stable Diffusion pipeline.

    Args:
        model_id (str): The model ID from HuggingFace to load.

    Returns:
        StableDiffusionPipeline: The loaded Stable Diffusion pipeline.
    """

    if device == "cuda":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=model_cache_dir,
            variant="fp16",
            torch_dtype=torch.float16,
        )
    else:
        # for CPU use fp32 if available
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, cache_dir=model_cache_dir, torch_dtype=torch.float32
        )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.unet.eval()  # Set to eval because we want to do inference and not training

    return pipe


def export_onnx_model(model_id: str) -> None:
    """
    Export the UNet model of the Stable Diffusion pipeline to ONNX format.

    Args:
        model_id (str): The model ID from HuggingFace to export.

    """

    pipe = load_sd_pipeline(model_id)

    batch_size = 1
    # for 512x512 images
    height = 64
    width = 64

    dummy_latents = torch.randn(
        batch_size,
        pipe.unet.config.in_channels,
        height,
        width,
        device=device,
        dtype=pipe.unet.dtype,
    )
    dummy_timestep = torch.tensor(
        [10], device=device, dtype=torch.int64
    )  # arbitrary diffusion step

    # text embeddings
    tokenized = pipe.tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
    text_embeddings = pipe.text_encoder(tokenized)[0]

    if not os.path.exists(onnx_path):
        torch.onnx.export(
            pipe.unet,
            (dummy_latents, dummy_timestep, text_embeddings),
            onnx_path,
            export_params=True,
            opset_version=17,
            input_names=["latents", "timestep", "text_embeddings"],
            output_names=["output"],
            dynamic_axes={
                "latents": {0: "batch", 2: "height", 3: "width"},
                "text_embeddings": {0: "batch"},
                "output": {0: "batch", 2: "height", 3: "width"},
            },
        )
        LOGGER.info(f"Saved ONNX model to {onnx_path}")
    else:
        LOGGER.info(f"ONNX model already exists at {onnx_path}")


if __name__ == "__main__":
    model_id = ModelType.STABLE_DIFFUSION_2_BASE.value

    export_onnx_model(model_id)
