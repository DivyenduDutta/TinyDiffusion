from enum import Enum

PROMPT = (
    "A whale falling through a starry sky beside a floating bowl of petunias, painted in a surreal "
    "cosmic landscape, whimsical and dreamlike, detailed digital art."
)


class ModelType(Enum):
    STABLE_DIFFUSION_2_BASE = (
        "stabilityai/stable-diffusion-2-base"  # This is not LoRA checkpoint
    )
