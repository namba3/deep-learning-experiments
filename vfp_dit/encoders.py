"""Frozen model loading and latent normalization for the VFCB-free variant."""

from __future__ import annotations

import torch
from torch import nn


DEFAULT_VAE_MODEL = "Qwen/Qwen-Image"
LATENT_CHANNELS = 16
LATENT_DOWNSAMPLE_FACTOR = 8


def load_qwen_image_vae(
    *,
    model_id: str = DEFAULT_VAE_MODEL,
    device: torch.device | str,
    dtype: torch.dtype,
) -> nn.Module:
    """Load/freeze the original Qwen-Image VAE, not the 2.1 variant."""
    try:
        import diffusers
    except ImportError as error:
        raise RuntimeError(
            "Qwen-Image VAE loading requires Diffusers with AutoencoderKLQwenImage"
        ) from error
    vae_class = getattr(diffusers, "AutoencoderKLQwenImage", None)
    if vae_class is None:
        raise RuntimeError(
            "Installed Diffusers does not export AutoencoderKLQwenImage"
        )
    vae = vae_class.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype,
    ).to(device=device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    if int(getattr(vae.config, "z_dim", -1)) != LATENT_CHANNELS:
        raise ValueError("Qwen-Image VAE config must declare z_dim=16")
    _latent_statistics(vae, device=torch.device(device))
    return vae


def _latent_statistics(
    vae: nn.Module,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    config = vae.config
    mean, std = getattr(config, "latents_mean", None), getattr(config, "latents_std", None)
    if mean is None or std is None:
        raise ValueError("Qwen-Image VAE config must declare latents_mean/std")
    mean = torch.as_tensor(mean, device=device, dtype=torch.float32).reshape(1, -1, 1, 1)
    std = torch.as_tensor(std, device=device, dtype=torch.float32).reshape(1, -1, 1, 1)
    if mean.shape != (1, LATENT_CHANNELS, 1, 1) or std.shape != mean.shape:
        raise ValueError("Qwen-Image latent statistics must have width 16")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Qwen-Image latent statistics must be finite with positive std")
    return mean, std


@torch.no_grad()
def encode_qwen_image_latents(
    vae: nn.Module,
    images: torch.Tensor,
    *,
    sample_mode: str = "mode",
) -> torch.Tensor:
    """Encode RGB images in [-1,1] and normalize ``(z-mean)/std``.

    The original Qwen-Image VAE is a causal video VAE, so still images are
    passed with one frame as ``(B,3,1,H,W)``. The singleton frame is removed
    from its 16-channel latent before returning ``(B,16,h,w)``.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape (B, 3, H, W)")
    if not images.is_floating_point() or not torch.isfinite(images).all():
        raise ValueError("images must be finite floating-point tensors")
    if sample_mode not in {"mode", "sample"}:
        raise ValueError("sample_mode must be 'mode' or 'sample'")
    parameter = next(vae.parameters(), None)
    if parameter is None:
        raise ValueError("VAE must have parameters to determine device and dtype")
    image_video = images.to(device=parameter.device, dtype=parameter.dtype).unsqueeze(2)
    encoded = vae.encode(image_video)
    distribution = getattr(encoded, "latent_dist", None)
    if distribution is None:
        raise TypeError("Qwen-Image VAE encode output must expose latent_dist")
    latent = distribution.mode() if sample_mode == "mode" else distribution.sample()
    if latent.ndim != 5 or latent.shape[2] != 1 or latent.shape[1] != LATENT_CHANNELS:
        raise ValueError(
            "still-image Qwen-Image latent must be (B,16,1,H,W), got "
            + str(tuple(latent.shape))
        )
    latent = latent[:, :, 0].float()
    mean, std = _latent_statistics(vae, device=latent.device)
    normalized = (latent - mean) / std
    if not torch.isfinite(normalized).all():
        raise FloatingPointError("normalized Qwen-Image latent is non-finite")
    return normalized


@torch.no_grad()
def decode_qwen_image_latents(vae: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    """Decode normalized ``(B,16,H,W)`` latents to RGB tensors in ``[-1,1]``."""
    if latents.ndim != 4 or latents.shape[1] != LATENT_CHANNELS:
        raise ValueError("latents must have shape (B, 16, H, W)")
    parameter = next(vae.parameters(), None)
    if parameter is None:
        raise ValueError("VAE must have parameters to determine device and dtype")
    mean, std = _latent_statistics(vae, device=parameter.device)
    raw_latents = latents.to(device=parameter.device, dtype=torch.float32) * std + mean
    decoded = vae.decode(raw_latents.to(dtype=parameter.dtype).unsqueeze(2))
    images = getattr(decoded, "sample", None)
    if images is None:
        raise TypeError("Qwen-Image decode output must expose sample")
    if images.ndim != 5 or images.shape[1] != 3 or images.shape[2] != 1:
        raise ValueError(
            "still-image Qwen-Image decode must return (B, 3, 1, H, W), got "
            + str(tuple(images.shape))
        )
    return images[:, :, 0].float().clamp(-1.0, 1.0)


__all__ = [
    "DEFAULT_VAE_MODEL",
    "LATENT_CHANNELS",
    "LATENT_DOWNSAMPLE_FACTOR",
    "decode_qwen_image_latents",
    "encode_qwen_image_latents",
    "load_qwen_image_vae",
]
