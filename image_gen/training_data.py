"""Dataset and loader setup for image generation training."""

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from image_gen.data import (
    AspectRatioBatchSampler,
    Flickr30KDataset,
    HFDataset,
    RecordsDataset,
    collate,
)
from image_gen.inference import (
    amp_context_for,
    encode_images,
    validate_bucket_shapes_with_vae,
)
from runtime.data import build_dataloader_options


@dataclass
class TrainingData:
    """Data loader and step counts required by the training loop."""

    dataset: object
    loader: DataLoader
    batch_sampler: AspectRatioBatchSampler
    vae_stride: int
    steps_per_epoch: int
    total_optimizer_steps: int
    warmup_steps: int


@dataclass
class LatentProbe:
    """VAE probe tensors and dimensions inferred before model construction."""

    probe: Any
    latent: torch.Tensor
    sample_probe: torch.Tensor
    sample_latent: torch.Tensor
    latent_channels: int
    sample_latent_height: int
    sample_latent_width: int


def build_training_data(args, vae, device):
    """Build the dataset, aspect-ratio loader, and scheduler step counts."""
    if args.records:
        dataset = RecordsDataset(args.records, args.image_size, args.bucket_step)
    elif args.dataset_name.lower() == "nlphuji/flickr30k":
        dataset = Flickr30KDataset(
            args.dataset_split, args.image_size, args.bucket_step, args.cache_dir,
        )
    else:
        dataset = HFDataset(
            args.dataset_name, args.dataset_split, args.image_size,
            args.bucket_step, args.cache_dir,
        )

    vae_stride = validate_bucket_shapes_with_vae(
        vae, dataset.bucket_shapes, args.latent_scale,
    )
    print(f"validated VAE stride={vae_stride} for {len(dataset.bucket_shapes)} buckets")
    batch_sampler = AspectRatioBatchSampler(
        dataset.bucket_ids, args.batch_size, seed=args.seed,
    )
    loader_kwargs = {
        **build_dataloader_options(
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            seed=args.seed,
            stream=0,
        ),
        "collate_fn": collate,
        "batch_sampler": batch_sampler,
    }
    loader = DataLoader(dataset, **loader_kwargs)
    if len(loader) == 0:
        raise ValueError("The dataset must contain at least --batch-size samples")

    steps_per_epoch = math.ceil(len(loader) / args.grad_accumulation)
    total_optimizer_steps = args.epochs * steps_per_epoch
    if args.warmup_steps > total_optimizer_steps:
        raise ValueError(
            f"--warmup-steps ({args.warmup_steps}) cannot exceed total optimizer "
            f"steps ({total_optimizer_steps})"
        )
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps > 0
        else (
            max(1, round(args.warmup_ratio * total_optimizer_steps))
            if args.warmup_ratio > 0.0
            else 0
        )
    )
    if warmup_steps > total_optimizer_steps:
        raise ValueError(
            f"computed warmup steps ({warmup_steps}) cannot exceed total optimizer "
            f"steps ({total_optimizer_steps})"
        )

    return TrainingData(
        dataset=dataset,
        loader=loader,
        batch_sampler=batch_sampler,
        vae_stride=vae_stride,
        steps_per_epoch=steps_per_epoch,
        total_optimizer_steps=total_optimizer_steps,
        warmup_steps=warmup_steps,
    )


def probe_training_latents(args, dataset, vae, device, dtype):
    """Infer VAE latent channels and the model's configured spatial dimensions."""
    with torch.inference_mode():
        with amp_context_for(device, dtype):
            probe_loader_kwargs = {
                "batch_size": 1,
                "shuffle": False,
                "collate_fn": collate,
                **build_dataloader_options(
                    num_workers=args.num_workers,
                    pin_memory=device.type == "cuda",
                    seed=args.seed,
                    stream=1,
                ),
            }
            probe, _ = next(iter(DataLoader(dataset, **probe_loader_kwargs)))
            latent = encode_images(vae, probe.to(device), args.latent_scale)
            latent_channels = latent.shape[1]
            if args.latent_channels is None:
                args.latent_channels = latent_channels
                print(f"using VAE latent channel count: {args.latent_channels}")
            elif latent_channels != args.latent_channels:
                raise ValueError(
                    f"QwenImage VAE produced {latent_channels} latent channels, "
                    f"but --latent-channels is {args.latent_channels}. "
                    "Use --latent-channels to match the VAE, or omit it to infer automatically."
                )
            sample_probe = torch.zeros(
                1, 3, args.image_size, args.image_size, device=device,
            )
            sample_latent = encode_images(vae, sample_probe, args.latent_scale)
            sample_latent_height, sample_latent_width = sample_latent.shape[-2:]

    return LatentProbe(
        probe=probe,
        latent=latent,
        sample_probe=sample_probe,
        sample_latent=sample_latent,
        latent_channels=latent_channels,
        sample_latent_height=sample_latent_height,
        sample_latent_width=sample_latent_width,
    )
