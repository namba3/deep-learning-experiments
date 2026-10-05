"""Loading and freezing pretrained text and image encoders for training."""

from dataclasses import dataclass
from typing import Any

from image_gen.inference import resolve_vae_latent_scale


@dataclass
class FrozenTrainingComponents:
    """Pretrained tokenizer, encoders, and their resolved dimensions/scales."""

    tokenizer: Any
    text_encoder: Any
    vae: Any
    text_encoder_dim: int
    text_max_length: int
    latent_scale: float


def load_frozen_training_components(args, device, vae_dtype):
    """Load pretrained components, infer text dimensions, and freeze weights."""
    from transformers import AutoModel, AutoTokenizer
    from diffusers import AutoencoderKL

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    text_encoder = AutoModel.from_pretrained(args.text_model).to(device).eval()
    max_positions = getattr(text_encoder.config, "max_position_embeddings", None)
    if max_positions is None and hasattr(text_encoder.config, "text_config"):
        max_positions = getattr(
            text_encoder.config.text_config, "max_position_embeddings", None,
        )
    text_max_length = args.text_max_length
    if max_positions is not None:
        text_max_length = min(text_max_length, max_positions)

    text_encoder_dim = getattr(text_encoder.config, "hidden_size", None)
    if text_encoder_dim is None and hasattr(text_encoder.config, "text_config"):
        text_encoder_dim = text_encoder.config.text_config.hidden_size
    if text_encoder_dim is None:
        raise ValueError("Could not determine Text Encoder hidden size")

    if "qwen" in args.vae_model.lower():
        from diffusers import AutoencoderKLQwenImage

        vae = AutoencoderKLQwenImage.from_pretrained(
            args.vae_model, subfolder="vae", torch_dtype=vae_dtype,
        ).to(device).eval()
    else:
        vae = AutoencoderKL.from_pretrained(
            args.vae_model, subfolder="vae", torch_dtype=vae_dtype,
        ).to(device).eval()

    for module in (text_encoder, vae):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    latent_scale = resolve_vae_latent_scale(vae, args.latent_scale)
    return FrozenTrainingComponents(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        text_encoder_dim=text_encoder_dim,
        text_max_length=text_max_length,
        latent_scale=latent_scale,
    )
