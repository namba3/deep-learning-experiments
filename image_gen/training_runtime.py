"""Device and precision resolution for image generation training."""

from dataclasses import dataclass

import torch

from image_gen.inference import resolve_vae_dtype
from runtime.device import resolve_device


@dataclass
class TrainingRuntime:
    """Resolved device, compute dtypes, and compile setting."""

    device: torch.device
    dtype: torch.dtype
    trainable_dtype: torch.dtype
    vae_dtype: torch.dtype
    compile_enabled: bool


def resolve_training_runtime(args):
    """Resolve requested device and precision settings with CPU fallbacks."""
    device = resolve_device(args.device)
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        args.amp
    ]
    if device.type == "cpu":
        dtype = torch.float32
    requested_trainable_dtype = {
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.trainable_dtype]
    trainable_dtype = requested_trainable_dtype
    if device.type == "cpu" and trainable_dtype == torch.bfloat16:
        trainable_dtype = torch.float32
        print(
            "Trainable BF16 requested but CUDA is unavailable; "
            "using FP32 trainable parameters"
        )
    if (
        device.type == "cuda"
        and trainable_dtype == torch.bfloat16
        and dtype != torch.bfloat16
    ):
        raise ValueError(
            "--trainable-dtype bf16 requires --amp bf16 on CUDA; "
            "use --trainable-dtype fp32 for --amp no/fp16"
        )
    vae_dtype = resolve_vae_dtype(device, args.vae_dtype)
    if args.vae_dtype == "bf16" and vae_dtype != torch.bfloat16:
        print("VAE BF16 requested but CUDA is unavailable; using FP32 VAE")
    compile_enabled = args.compile and device.type == "cuda"
    if args.compile and not compile_enabled:
        print("torch.compile requested but CUDA is unavailable; skipping compile")
    return TrainingRuntime(
        device=device,
        dtype=dtype,
        trainable_dtype=trainable_dtype,
        vae_dtype=vae_dtype,
        compile_enabled=compile_enabled,
    )
