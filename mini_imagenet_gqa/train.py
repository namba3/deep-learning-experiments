"""Train one controlled GQA Transformer variant on timm/mini-imagenet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_imagenet_gqa.buckets import DEFAULT_BUCKET_SIZES, assign_bucket  # noqa: E402
from mini_imagenet_gqa.model import (  # noqa: E402
    MiniImageNetGQAModel, VARIANTS, copy_common_initialization_,
)
from runtime.checkpoint import (  # noqa: E402
    load_training_state,
    make_training_state,
    restore_rng_state,
    save_training_state,
)
from runtime.data import build_dataloader_options  # noqa: E402
from runtime.device import add_device_argument, resolve_device  # noqa: E402
from runtime.progress import RichProgress  # noqa: E402
from runtime.run import RunRecorder  # noqa: E402
from optimizers.schedulefree import AdamWScheduleFree  # noqa: E402
from optimizers.apollo import APOLLO  # noqa: E402
from optimizers.apollo_sf import APOLLOScheduleFree  # noqa: E402
from optimizers.lr_scheduler import (  # noqa: E402
    add_lr_scheduler_arguments,
    build_lr_scheduler,
)

DATASET_ID = "timm/mini-imagenet"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CHECKPOINT_METADATA_KEY = "mini_imagenet_gqa.checkpoint"
CONFIG_METADATA_KEY = "mini_imagenet_gqa.config"
NETWORK_VERSION = 1
APOLLO_SF_OPTIMIZERS = (
    "APOLLO-SF",
    "APOLLO-SF-LRSF",
    "APOLLO-SF-INT8-Z",
    "APOLLO-SF-INT8-Delta",
    "APOLLO-SF-INT4-Z",
    "APOLLO-SF-INT4-Delta",
)
APOLLO_SF_STORAGE = {
    "APOLLO-SF": "bf16_z",
    "APOLLO-SF-LRSF": "low_rank_delta",
    "APOLLO-SF-INT8-Z": "blockwise_int8_z",
    "APOLLO-SF-INT8-Delta": "blockwise_int8_delta",
    "APOLLO-SF-INT4-Z": "blockwise_int4_z",
    "APOLLO-SF-INT4-Delta": "blockwise_int4_delta",
}


class RandomResizedBucketCrop:
    """Area/aspect jitter followed by a shape-preserving crop to one bucket."""

    def __init__(self, bucket_size: tuple[int, int], scale=(0.5, 1.0), ratio=(0.75, 4 / 3)):
        self.height, self.width = bucket_size
        self.scale = scale
        target_ratio = self.width / self.height
        self.ratio = (target_ratio * ratio[0], target_ratio * ratio[1])

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        area = width * height
        crop_width = crop_height = 0
        log_min, log_max = math.log(self.ratio[0]), math.log(self.ratio[1])
        for _ in range(10):
            target_area = area * random.uniform(*self.scale)
            aspect = math.exp(random.uniform(log_min, log_max))
            candidate_width = round(math.sqrt(target_area * aspect))
            candidate_height = round(math.sqrt(target_area / aspect))
            if 0 < candidate_width <= width and 0 < candidate_height <= height:
                crop_width, crop_height = candidate_width, candidate_height
                break
        if crop_width == 0:
            target_ratio = self.width / self.height
            if width / height > target_ratio:
                crop_height = height
                crop_width = max(1, min(width, round(height * target_ratio)))
            else:
                crop_width = width
                crop_height = max(1, min(height, round(width / target_ratio)))
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)
        return TF.resized_crop(
            image, top, left, crop_height, crop_width,
            [self.height, self.width], interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )


class ResizeCoverCenterCrop:
    """Deterministically resize without distortion, then center-crop a bucket."""

    def __init__(self, bucket_size: tuple[int, int]):
        self.height, self.width = bucket_size

    def __call__(self, image: Image.Image) -> Image.Image:
        source_width, source_height = image.size
        scale = max(self.width / source_width, self.height / source_height)
        resized_height = max(self.height, math.ceil(source_height * scale))
        resized_width = max(self.width, math.ceil(source_width * scale))
        image = TF.resize(
            image, [resized_height, resized_width],
            interpolation=InterpolationMode.BILINEAR, antialias=True,
        )
        return TF.center_crop(image, [self.height, self.width])


def build_transforms(
    bucket_size: tuple[int, int], *, degrees: float = 10.0, shear: float = 10.0,
):
    """Build CIFAR-10-order augmentation and deterministic evaluation for one H/W bucket."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.RandomAffine(degrees=degrees, shear=shear),
        transforms.RandomPerspective(distortion_scale=0.1),
        RandomResizedBucketCrop(bucket_size, scale=(0.5, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        ResizeCoverCenterCrop(bucket_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


class HFDatasetView(Dataset):
    def __init__(
        self,
        dataset,
        transforms_by_bucket,
        class_names: tuple[str, ...],
        bucket_sizes: tuple[tuple[int, int], ...],
        split: str,
    ):
        self.dataset = dataset
        self.transforms_by_bucket = transforms_by_bucket
        self.class_names = class_names
        self.class_to_index = {name: index for index, name in enumerate(class_names)}
        self.bucket_sizes = bucket_sizes
        self.bucket_indices_by_bucket: list[list[int]] = [
            [] for _ in bucket_sizes
        ]
        with RichProgress(total=len(dataset), description=f"assign {split} buckets") as progress:
            for index in range(len(dataset)):
                image = dataset[index].get("image")
                if not isinstance(image, Image.Image):
                    raise TypeError(
                        f"image at index {index} was not decoded as a PIL image"
                    )
                bucket_id = assign_bucket(*image.size, bucket_sizes)
                self.bucket_indices_by_bucket[bucket_id].append(index)
                progress.update(1)
        self.bucket_ids = [0] * len(dataset)
        for bucket_id, indices in enumerate(self.bucket_indices_by_bucket):
            for index in indices:
                self.bucket_ids[index] = bucket_id

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        row = self.dataset[index]
        image = row.get("image")
        label = row.get("label")
        if image is None or label is None:
            raise ValueError("dataset rows must contain non-null image and label fields")
        if not isinstance(image, Image.Image):
            raise TypeError(f"image at index {index} was decoded as {type(image).__name__}")
        if isinstance(label, str):
            try:
                label = self.class_to_index[label]
            except KeyError as error:
                raise ValueError(f"unknown class label: {label}") from error
        bucket_id = self.bucket_ids[index]
        bucket_height, bucket_width = self.bucket_sizes[bucket_id]
        image_tensor = self.transforms_by_bucket[bucket_id](image.convert("RGB"))
        if image_tensor.shape[-2:] != (bucket_height, bucket_width):
            raise ValueError(
                f"transform returned {tuple(image_tensor.shape[-2:])}; "
                f"expected bucket {(bucket_height, bucket_width)}"
            )
        # Metadata describes the target grid after bucket selection, not the source file.
        metadata = torch.tensor(
            [math.log(math.sqrt(bucket_width * bucket_height)),
             math.log(bucket_width / bucket_height)],
            dtype=torch.float32,
        )
        return image_tensor, int(label), metadata

    @property
    def bucket_counts(self) -> dict[str, int]:
        return {
            f"{height}x{width}": len(indices)
            for (height, width), indices in zip(self.bucket_sizes, self.bucket_indices_by_bucket)
        }


class BucketBatchSampler(Sampler[list[int]]):
    """Batch equal-shaped images together with deterministic epoch shuffling."""

    def __init__(self, dataset: HFDatasetView, batch_size: int, *, seed: int, shuffle: bool):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self.dataset.bucket_indices_by_bucket
        )

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        batches: list[list[int]] = []
        bucket_order = list(range(len(self.dataset.bucket_indices_by_bucket)))
        if self.shuffle:
            bucket_order = torch.randperm(len(bucket_order), generator=generator).tolist()
        for bucket_id in bucket_order:
            indices = self.dataset.bucket_indices_by_bucket[bucket_id]
            if not indices:
                continue
            if self.shuffle:
                order = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[position] for position in order]
            batches.extend(
                indices[start:start + self.batch_size]
                for start in range(0, len(indices), self.batch_size)
            )
        if self.shuffle and len(batches) > 1:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[position] for position in order]
        yield from batches



def _class_names(dataset) -> tuple[str, ...]:
    label_feature = dataset.features.get("label")
    names = getattr(label_feature, "names", None)
    if names:
        return tuple(str(name) for name in names)
    raise ValueError("dataset label feature must expose a ClassLabel names list")


def load_splits(args):
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("HF dataset loading requires the 'datasets' package") from error
    kwargs = {}
    if args.hf_cache_dir:
        kwargs["cache_dir"] = args.hf_cache_dir
    dataset = load_dataset(args.dataset, **kwargs)
    required = {"train", "validation", "test"}
    if not required.issubset(dataset.keys()):
        raise ValueError(f"dataset must provide train/validation/test splits; got {dataset.keys()}")
    class_names = _class_names(dataset["train"])
    if len(class_names) <= 1:
        raise ValueError("classification requires at least two classes")
    train_transforms = []
    eval_transforms = []
    for bucket_size in args.bucket_sizes:
        train_transform, eval_transform = build_transforms(
            bucket_size, degrees=args.transform_degrees, shear=args.transform_shear,
        )
        train_transforms.append(train_transform)
        eval_transforms.append(eval_transform)
    views = {
        "train": HFDatasetView(
            dataset["train"], train_transforms, class_names, args.bucket_sizes, "train",
        ),
        "validation": HFDatasetView(
            dataset["validation"], eval_transforms, class_names, args.bucket_sizes, "validation",
        ),
        "test": HFDatasetView(
            dataset["test"], eval_transforms, class_names, args.bucket_sizes, "test",
        ),
    }
    return views, class_names


def parse_bucket_sizes(value: str) -> tuple[tuple[int, int], ...]:
    try:
        sizes = tuple(
            tuple(int(axis.strip()) for axis in item.lower().split("x"))
            for item in value.split(",")
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "bucket-sizes must be comma-separated HxW pairs, for example 64x64,56x72"
        ) from error
    if not sizes or any(len(size) != 2 or min(size) <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("bucket-sizes must contain positive HxW pairs")
    if len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("bucket-sizes must be unique")
    return sizes


def parse_widths(value: str) -> tuple[int, ...]:
    try:
        widths = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("widths must be comma-separated integers") from error
    if not widths or min(widths) <= 0:
        raise argparse.ArgumentTypeError("all stage widths must be positive")
    return widths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--variant", choices=VARIANTS, default="naive_gqa")
    parser.add_argument("--image-size", type=int, default=64,
                        help="Reference size used to initialize 2D RoPE tables.")
    parser.add_argument(
        "--bucket-sizes", type=parse_bucket_sizes, default=DEFAULT_BUCKET_SIZES,
        help="Comma-separated target HxW sizes, each divisible by 2**number-of-stages.",
    )
    parser.add_argument("--widths", type=parse_widths, default=(96, 192, 256))
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--blocks-per-stage", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ff-mult", type=float, default=4.0)
    parser.add_argument("--transform-degrees", type=float, default=10.0)
    parser.add_argument("--transform-shear", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer",
        choices=("AdamW", "AdamW-SF", "APOLLO", *APOLLO_SF_OPTIMIZERS),
        default="AdamW",
    )
    parser.add_argument("--apollo-rank", type=int, default=32,
                        help="Low-rank projection size for APOLLO/APOLLO-SF.")
    parser.add_argument(
        "--apollo-sf-quant-block-size", type=int, default=256,
        help="Block size for APOLLO-SF INT8/INT4 state quantization.",
    )
    parser.add_argument(
        "--apollo-sf-delta-refresh", choices=("none", "commit_z", "blend"), default="none",
        help=(
            "APOLLO-SF delta policy at APOLLO projection refresh: none, "
            "commit_z, or blend."
        ),
    )
    parser.add_argument(
        "--apollo-sf-delta-refresh-window", type=int, default=4,
        help="Number of steps used by APOLLO-SF delta blend refresh.",
    )
    parser.add_argument(
        "--apollo-update-proj-gap", type=int, default=200,
        help="APOLLO/APOLLO-SF hard projection refresh interval.",
    )
    parser.add_argument(
        "--apollo-norm-growth-limiter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Limit per-parameter APOLLO update-norm growth to the configured "
            "rate. Enable with --apollo-norm-growth-limiter. Default: off."
        ),
    )
    parser.add_argument(
        "--apollo-fallback", choices=("came", "sgd", "adamw-sf"), default="adamw-sf",
        help="Fallback optimizer for APOLLO 1D parameters. Default: adamw-sf.",
    )
    parser.add_argument(
        "--apollo-matrix-fallback",
        choices=("apollo", "came", "auto", "adamw-sf", "auto-sf"),
        default="auto-sf",
        help=(
            "Fallback for APOLLO matrix parameters. auto compares CAME state; "
            "auto-sf compares AdamW-SF state. Default: auto-sf."
        ),
    )
    add_lr_scheduler_arguments(parser, default="cosine")
    parser.add_argument(
        "--adamw-sf-backend", choices=("torch", "auto", "triton"), default="torch",
        help="AdamW-SF backend; torch is the reference implementation.",
    )
    parser.add_argument(
        "--conditioning-lr-multiplier", type=float, default=1.0,
        help=(
            "Scale LR for metadata embedding and conditioning projections; "
            "1 keeps the single-group default."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--amp", choices=("none", "bf16"), default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="mini_imagenet_gqa/output/bucketed")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--common-init", action="store_true",
                        help="Copy all shape-compatible shared weights from a naive-GQA reference model.")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic PyTorch algorithms for repeatability checks.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--steps-per-epoch", type=int, default=0,
                        help="Optional screening cap; 0 uses all training batches.")
    parser.add_argument("--eval-batches", type=int, default=0,
                        help="Optional screening cap; 0 evaluates the complete split.")
    add_device_argument(parser)
    return parser.parse_args(argv)


def validate_args(args) -> None:
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs/batch-size must be positive and num-workers non-negative")
    if args.heads <= 0 or args.kv_heads <= 0 or args.heads % args.kv_heads:
        raise ValueError("heads must be divisible by positive kv-heads")
    if args.blocks_per_stage <= 0 or args.image_size <= 0:
        raise ValueError("blocks-per-stage and image-size must be positive")
    downsample_factor = 2 ** len(args.widths)
    if args.image_size % downsample_factor:
        raise ValueError("image-size must be divisible by 2**number-of-stages")
    if any(
        height % downsample_factor or width % downsample_factor
        for height, width in args.bucket_sizes
    ):
        raise ValueError("every bucket H/W must be divisible by 2**number-of-stages")
    # A resume restores the complete model and optimizer state after model
    # construction, so retaining --common-init preserves the run protocol and
    # initial-state metadata without changing the resumed weights.
    if args.steps_per_epoch < 0 or args.eval_batches < 0:
        raise ValueError("step/eval caps must be non-negative")
    if (
        args.lr <= 0
        or not math.isfinite(args.conditioning_lr_multiplier)
        or args.conditioning_lr_multiplier <= 0
        or args.weight_decay < 0
        or args.dropout < 0
        or args.dropout >= 1
    ):
        raise ValueError("invalid optimizer, conditioning LR multiplier, or dropout values")
    if args.ff_mult <= 0:
        raise ValueError("ff-mult must be positive")
    if args.apollo_rank <= 0:
        raise ValueError("apollo-rank must be positive")
    if args.apollo_sf_quant_block_size <= 0:
        raise ValueError("apollo-sf-quant-block-size must be positive")
    if args.apollo_update_proj_gap <= 0:
        raise ValueError("apollo-update-proj-gap must be positive")
    if args.apollo_sf_delta_refresh_window <= 0:
        raise ValueError("apollo-sf-delta-refresh-window must be positive")
    if args.warmup_steps is not None and args.warmup_steps < 0:
        raise ValueError("warmup-steps must be non-negative")
    if args.warmup_ratio is not None and not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup-ratio must be in [0, 1)")
    if args.warmup_steps and args.warmup_ratio:
        raise ValueError("use only one of --warmup-steps and --warmup-ratio")
    for width in args.widths:
        if width % args.heads or (width // args.heads) % 4:
            raise ValueError("every width/head_dim must be divisible for 2D RoPE")


def _model(args, num_classes: int) -> MiniImageNetGQAModel:
    return MiniImageNetGQAModel(
        num_classes=num_classes,
        widths=args.widths,
        heads=args.heads,
        kv_heads=args.kv_heads,
        blocks_per_stage=args.blocks_per_stage,
        image_size=args.image_size,
        variant=args.variant,
        dropout=args.dropout,
        ff_mult=args.ff_mult,
    )


def _optimizer_parameters(
    model: nn.Module, lr: float, conditioning_lr_multiplier: float,
):
    """Keep the default single group; optionally isolate metadata/AdaRMS LR."""
    if conditioning_lr_multiplier == 1.0:
        return model.parameters()

    main_parameters = []
    conditioning_parameters = []
    for name, parameter in model.named_parameters():
        is_conditioning = name.startswith("meta_embedding.") or any(
            f".{module_name}." in name
            for module_name in (
                "norm1_scale", "norm2_scale", "norm1_shift", "norm2_shift",
                "q_meta_proj", "kv_meta_proj", "q_only_meta_shift",
                "ffn_meta_proj", "meta_projection",
            )
        )
        (conditioning_parameters if is_conditioning else main_parameters).append(parameter)
    if not conditioning_parameters:
        return model.parameters()
    groups = []
    if main_parameters:
        groups.append({"params": main_parameters, "lr": lr})
    groups.append({
        "params": conditioning_parameters,
        "lr": lr * conditioning_lr_multiplier,
    })
    return groups


def _optimizer_state_summary(optimizer) -> dict:
    """Summarize persistent optimizer tensors and selected APOLLO backends."""
    state_bytes = 0
    parameter_counts: dict[str, int] = {}
    parameter_numel: dict[str, int] = {}
    delta_commit_count = 0
    delta_commit_norm_sum = 0.0
    delta_commit_norm_max = 0.0
    delta_blend_count = 0
    delta_blend_step_count = 0
    delta_blend_norm_sum = 0.0
    for group in optimizer.param_groups:
        delta_commit_count += int(group.get("sf_delta_commit_count", 0))
        delta_commit_norm_sum += float(group.get("sf_delta_commit_norm_sum", 0.0))
        delta_commit_norm_max = max(
            delta_commit_norm_max,
            float(group.get("sf_delta_commit_norm_max", 0.0)),
        )
        delta_blend_count += int(group.get("sf_delta_blend_count", 0))
        delta_blend_step_count += int(group.get("sf_delta_blend_step_count", 0))
        delta_blend_norm_sum += float(group.get("sf_delta_blend_norm_sum", 0.0))
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            backend = str(state.get("backend", type(optimizer).__name__))
            parameter_counts[backend] = parameter_counts.get(backend, 0) + 1
            parameter_numel[backend] = (
                parameter_numel.get(backend, 0) + parameter.numel()
            )
            state_bytes += sum(
                value.numel() * value.element_size()
                for value in state.values()
                if isinstance(value, torch.Tensor)
            )
    return {
        "persistent_state_bytes": state_bytes,
        "parameter_counts_by_backend": parameter_counts,
        "parameter_numel_by_backend": parameter_numel,
        "sf_delta_commit_count": delta_commit_count,
        "sf_delta_commit_norm_sum": delta_commit_norm_sum,
        "sf_delta_commit_norm_max": delta_commit_norm_max,
        "sf_delta_blend_count": delta_blend_count,
        "sf_delta_blend_step_count": delta_blend_step_count,
        "sf_delta_blend_norm_sum": delta_blend_norm_sum,
    }


def _initialize_model(args, num_classes: int, device: torch.device):
    if not args.common_init:
        return _model(args, num_classes).to(device), 0

    # Keep model construction's random draws isolated from the training RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        reference_args = argparse.Namespace(**vars(args))
        reference_args.variant = "naive_gqa"
        reference = _model(reference_args, num_classes)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        model = _model(args, num_classes)
    copied = copy_common_initialization_(model, reference)
    del reference
    return model.to(device), copied


def _model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        cpu_tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(cpu_tensor.dtype).encode("ascii"))
        digest.update(str(tuple(cpu_tensor.shape)).encode("ascii"))
        digest.update(cpu_tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _autocast(device: torch.device, amp: str):
    if amp == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast(device.type, enabled=False)


@torch.inference_mode()
def evaluate(model, loader, device, *, amp: str, max_batches: int = 0) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = total = batches = 0
    for batch in loader:
        images, labels = batch[:2]
        metadata = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _autocast(device, amp):
            logits = model(images, metadata=metadata)
            loss = nn.functional.cross_entropy(logits.float(), labels)
        count = labels.numel()
        loss_sum += float(loss) * count
        correct += int((logits.argmax(-1) == labels).sum())
        total += count
        batches += 1
        if max_batches and batches >= max_batches:
            break
    if total == 0:
        raise ValueError("evaluation loader produced no samples")
    return {"loss": loss_sum / total, "top1": correct / total, "samples": float(total)}


def _checkpoint_payload(
    model, args, *, epoch: int, global_step: int, optimizer, scheduler,
    best_validation_top1: float | None = None,
):
    tensors = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    metadata = {
        CHECKPOINT_METADATA_KEY: json.dumps({
            "network_version": NETWORK_VERSION,
            "epoch": epoch,
            "global_step": global_step,
            "variant": args.variant,
            "best_validation_top1": best_validation_top1,
        }, sort_keys=True),
        CONFIG_METADATA_KEY: json.dumps(vars(args), sort_keys=True),
    }
    return tensors, metadata, make_training_state(
        optimizer=optimizer, epoch=epoch, global_step=global_step, scheduler=scheduler,
    )


def save_checkpoint(
    path: Path, model, args, *, epoch: int, global_step: int, optimizer, scheduler,
    save_resume_state: bool, best_validation_top1: float | None = None,
) -> None:
    tensors, metadata, state = _checkpoint_payload(
        model, args, epoch=epoch, global_step=global_step,
        optimizer=optimizer, scheduler=scheduler,
        best_validation_top1=best_validation_top1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)
    if save_resume_state:
        save_training_state(path, state)


def restore_checkpoint(
    path: Path, model, optimizer, scheduler, device, args, *,
    include_best_validation: bool = False,
) -> tuple[int, int] | tuple[int, int, float]:
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
    record = json.loads(metadata.get(CHECKPOINT_METADATA_KEY, "{}"))
    if record.get("network_version") != NETWORK_VERSION:
        raise ValueError("unsupported classification checkpoint version")
    saved_config_text = metadata.get(CONFIG_METADATA_KEY)
    if saved_config_text is not None:
        saved_config = json.loads(saved_config_text)
        runtime_keys = {
            "device", "dry_run", "hf_cache_dir", "output_dir", "resume",
            "run_name", "num_workers", "validate_only",
        }
        current_config = vars(args)
        legacy_defaults = {
            "conditioning_lr_multiplier": 1.0,
            "optimizer": "AdamW",
            "adamw_sf_backend": "torch",
            "apollo_rank": 32,
            "lr_scheduler": "cosine",
            "warmup_steps": None,
            "warmup_ratio": None,
            "min_lr_ratio": 0.0,
            "lr_step_size": 1000,
            "lr_gamma": 0.1,
            "lr_milestones": [1000, 2000],
            "lr_num_cycles": 1,
            "lr_power": 1.0,
        }
        config_keys = (set(saved_config) | set(current_config)) - runtime_keys
        differences = []
        for key in config_keys:
            saved_value = saved_config.get(key, legacy_defaults.get(key))
            current_value = current_config.get(key, legacy_defaults.get(key))
            if saved_value != json.loads(json.dumps(current_value)):
                differences.append(key)
        if differences:
            raise ValueError("resume settings differ from checkpoint: " + ", ".join(sorted(differences)))
    weights = load_file(str(path), device=str(device))
    model.load_state_dict(weights, strict=True)
    state = load_training_state(path)
    epoch = int(record.get("epoch", 0))
    global_step = int(record.get("global_step", 0))
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None:
            if scheduler is None:
                raise ValueError(
                    "checkpoint has scheduler state but this optimizer does not use a scheduler"
                )
            scheduler.load_state_dict(state["scheduler"])
        restore_rng_state(state["rng"])
        epoch = int(state["epoch"])
        global_step = int(state["global_step"])
    if include_best_validation:
        best_validation = record.get("best_validation_top1")
        if best_validation is None:
            # Recover the historical best when resuming a checkpoint written
            # before the score was stored in its safetensors metadata.
            metrics_path = path.parent.parent / "metrics.jsonl"
            if metrics_path.is_file():
                for line in metrics_path.read_text(encoding="utf-8").splitlines():
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    validation = event.get("validation", {})
                    top1 = validation.get("top1") if event.get("event") == "epoch" else None
                    if isinstance(top1, (int, float)):
                        best_validation = max(float(top1), float(best_validation or float("-inf")))
        return epoch, global_step, (float(best_validation) if best_validation is not None else float("-inf"))
    return epoch, global_step


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    *,
    amp: str,
    steps_cap: int = 0,
    recorder: RunRecorder | None = None,
    epoch: int = 1,
    total_epochs: int = 1,
    global_step_start: int = 0,
    total_steps: int = 0,
    progress_started: float | None = None,
    progress_initial_step: int = 0,
    step_scheduler=None,
) -> dict[str, float]:
    model.train()
    if hasattr(optimizer, "train"):
        optimizer.train()
    loss_sum = 0.0
    correct = total = batches = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    started = perf_counter()
    epoch_batches = min(len(loader), steps_cap) if steps_cap else len(loader)
    with RichProgress(total=epoch_batches, description=f"train epoch {epoch}/{total_epochs}") as progress:
        for batch in loader:
            batch_started = perf_counter()
            images, labels = batch[:2]
            metadata = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with _autocast(device, amp):
                logits = model(images, metadata=metadata)
                loss = nn.functional.cross_entropy(logits.float(), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("training loss became non-finite")
            loss.backward()
            if step_scheduler is not None:
                step_scheduler.step(global_step_start + batches + 1)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            count = labels.numel()
            loss_sum += float(loss.detach()) * count
            correct += int((logits.detach().argmax(-1) == labels).sum())
            total += count
            batches += 1
            progress.update(1, postfix=f"loss={loss.item():.4f} acc={correct / total:.3f}")
            current_global_step = global_step_start + batches
            if recorder is not None:
                elapsed_seconds = (
                    perf_counter() - progress_started
                    if progress_started is not None else perf_counter() - started
                )
                completed_this_run = max(current_global_step - progress_initial_step, 1)
                steps_per_second = completed_this_run / max(elapsed_seconds, 1e-12)
                remaining_steps = max(total_steps - current_global_step, 0)
                recorder.write_progress(
                    current_global_step,
                    status="training",
                    epoch=f"{epoch}/{total_epochs}",
                    batch=f"{batches}/{epoch_batches}",
                    total_steps=total_steps,
                    loss=f"{float(loss.detach()):.6f}",
                    accuracy=f"{correct / total:.5f}",
                    learning_rate=f"{float(optimizer.param_groups[0]['lr']):.6g}",
                    step_time=f"{perf_counter() - batch_started:.3f}s",
                    elapsed=_format_duration(elapsed_seconds),
                    speed=f"{steps_per_second:.3f} optimizer_steps/s",
                    eta=_format_duration(remaining_steps / max(steps_per_second, 1e-12)),
                )
            if steps_cap and batches >= steps_cap:
                break
    if total == 0:
        raise ValueError("training loader produced no samples")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = perf_counter() - started
    metrics = {
        "loss": loss_sum / total,
        "top1": correct / total,
        "samples": float(total),
        "seconds": seconds,
        "samples_per_second": total / max(seconds, 1e-12),
    }
    if device.type == "cuda":
        metrics["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        metrics["peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return metrics


def main(argv=None) -> None:
    args = parse_args(argv)
    validate_args(args)
    if args.deterministic:
        # CUBLAS reads this before CUDA context initialization.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if args.dry_run:
        model, copied = _initialize_model(args, num_classes=100, device=device)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"variant={args.variant} parameters={parameters:,} device={device} "
            f"common_init_copied={copied:,}; no dataset loaded"
        )
        return

    datasets, class_names = load_splits(args)
    model, common_init_copied = _initialize_model(args, len(class_names), device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    initial_state_sha256 = _model_state_sha256(model)
    if args.validate_only:
        sample, label, metadata = datasets["train"][0]
        with torch.no_grad():
            logits = model(sample[None].to(device), metadata=metadata[None].to(device))
        print(
            f"dataset={args.dataset} splits="
            f"{ {name: len(data) for name, data in datasets.items()} } "
            f"classes={len(class_names)} image={tuple(sample.shape)} label={label} "
            f"metadata={metadata.tolist()} logits={tuple(logits.shape)} "
            f"parameters={parameters:,}"
        )
        return
    run = RunRecorder(
        args.output_dir, script="mini_imagenet_gqa.train",
        config={
            **vars(args),
            "lr_schedule_mode": (
                "optimizer_step_warmup_decay" if args.optimizer == "APOLLO"
                else "schedule_free"
                if args.optimizer == "AdamW-SF"
                or args.optimizer in APOLLO_SF_OPTIMIZERS
                else "epoch_cosine"
            ),
            "num_classes": len(class_names),
            "class_names": class_names,
        },
        run_name=args.run_name,
    )
    run.record(
        "model", num_classes=len(class_names), parameters=parameters,
        common_init_copied_parameters=common_init_copied,
        common_init_reference="naive_gqa" if args.common_init else None,
        initial_state_sha256=initial_state_sha256,
        deterministic=args.deterministic,
    )
    optimizer_parameters = _optimizer_parameters(
        model, args.lr, args.conditioning_lr_multiplier,
    )
    train_batch_sampler = BucketBatchSampler(
        datasets["train"], args.batch_size, seed=args.seed, shuffle=True,
    )
    batches_per_epoch = len(train_batch_sampler)
    if args.steps_per_epoch:
        batches_per_epoch = min(batches_per_epoch, args.steps_per_epoch)
    total_optimizer_steps = batches_per_epoch * args.epochs
    if args.optimizer == "AdamW-SF":
        optimizer = AdamWScheduleFree(
            optimizer_parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
            backend=args.adamw_sf_backend,
        )
        scheduler = None
        step_scheduler = None
    elif args.optimizer in APOLLO_SF_OPTIMIZERS:
        optimizer = APOLLOScheduleFree(
            optimizer_parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
            rank=args.apollo_rank,
            sf_state_storage=APOLLO_SF_STORAGE[args.optimizer],
            sf_quant_block_size=args.apollo_sf_quant_block_size,
            sf_delta_refresh=args.apollo_sf_delta_refresh,
            sf_delta_refresh_window=args.apollo_sf_delta_refresh_window,
            update_proj_gap=args.apollo_update_proj_gap,
            fallback={
                "1d": args.apollo_fallback,
                "small_matrix": args.apollo_matrix_fallback,
            },
            seed=args.seed,
            norm_growth_limiter=args.apollo_norm_growth_limiter,
        )
        scheduler = None
        step_scheduler = None
    elif args.optimizer == "APOLLO":
        optimizer = APOLLO(
            optimizer_parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
            rank=args.apollo_rank,
            update_proj_gap=args.apollo_update_proj_gap,
            seed=args.seed,
            norm_growth_limiter=args.apollo_norm_growth_limiter,
            fallback={
                "1d": args.apollo_fallback,
                "small_matrix": args.apollo_matrix_fallback,
            },
        )
        scheduler = build_lr_scheduler(optimizer, args, total_optimizer_steps)
        scheduler.step(0)
        step_scheduler = scheduler
    else:
        optimizer = torch.optim.AdamW(
            optimizer_parameters, lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        step_scheduler = None
    start_epoch = global_step = 0
    best_validation = float("-inf")
    resume_checkpoint = None
    if args.resume:
        checkpoint = Path(args.resume).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        start_epoch, global_step, best_validation = restore_checkpoint(
            checkpoint, model, optimizer, scheduler, device, args,
            include_best_validation=True,
        )
        resume_checkpoint = checkpoint
    train_options = build_dataloader_options(
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        seed=args.seed, stream=0,
    )
    eval_options = build_dataloader_options(
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        seed=args.seed, stream=1,
    )
    train_generator = train_options.pop("generator")
    train_options["persistent_workers"] = False
    eval_options["persistent_workers"] = False
    validation_batch_sampler = BucketBatchSampler(
        datasets["validation"], args.batch_size, seed=args.seed + 1, shuffle=False,
    )
    test_batch_sampler = BucketBatchSampler(
        datasets["test"], args.batch_size, seed=args.seed + 2, shuffle=False,
    )
    validation_loader = DataLoader(
        datasets["validation"], batch_sampler=validation_batch_sampler, **eval_options,
    )
    test_loader = DataLoader(
        datasets["test"], batch_sampler=test_batch_sampler, **eval_options,
    )
    run.record(
        "data", bucket_sizes=args.bucket_sizes,
        bucket_counts={name: view.bucket_counts for name, view in datasets.items()},
    )
    best_path = run.checkpoints_dir / "checkpoint_best.safetensors"
    if resume_checkpoint is not None:
        source_best_path = resume_checkpoint.parent / "checkpoint_best.safetensors"
        if source_best_path.is_file():
            shutil.copy2(source_best_path, best_path)
    progress_started = perf_counter()
    progress_initial_step = global_step
    run.write_progress(
        global_step,
        status="starting",
        epoch=f"{start_epoch + 1}/{args.epochs}",
        batch=f"0/{batches_per_epoch}",
        total_steps=total_optimizer_steps,
        learning_rate=f"{float(optimizer.param_groups[0]['lr']):.6g}",
    )
    for epoch in range(start_epoch, args.epochs):
        train_generator.manual_seed(args.seed + epoch)
        train_batch_sampler.set_epoch(epoch)
        train_loader = DataLoader(
            datasets["train"], batch_sampler=train_batch_sampler,
            generator=train_generator, **train_options,
        )
        epoch_optimizer_steps = min(len(train_loader), args.steps_per_epoch) if args.steps_per_epoch else len(train_loader)
        train_metrics = run_epoch(
            model, train_loader, optimizer, device, amp=args.amp,
            steps_cap=args.steps_per_epoch,
            recorder=run,
            epoch=epoch + 1,
            total_epochs=args.epochs,
            global_step_start=global_step,
            total_steps=total_optimizer_steps,
            progress_started=progress_started,
            progress_initial_step=progress_initial_step,
            step_scheduler=step_scheduler,
        )
        run.write_progress(
            global_step + epoch_optimizer_steps,
            status="validating",
            epoch=f"{epoch + 1}/{args.epochs}",
            batch=f"{batches_per_epoch}/{batches_per_epoch}",
            total_steps=total_optimizer_steps,
            train_loss=f"{train_metrics['loss']:.6f}",
            train_accuracy=f"{train_metrics['top1']:.5f}",
            learning_rate=f"{float(optimizer.param_groups[0]['lr']):.6g}",
        )
        if hasattr(optimizer, "eval"):
            optimizer.eval()
        validation_metrics = evaluate(
            model, validation_loader, device, amp=args.amp, max_batches=args.eval_batches,
        )
        if scheduler is not None and step_scheduler is None:
            scheduler.step()
        global_step += epoch_optimizer_steps
        run.write_progress(
            global_step,
            status="epoch_complete",
            epoch=f"{epoch + 1}/{args.epochs}",
            batch=f"{epoch_optimizer_steps}/{epoch_optimizer_steps}",
            total_steps=total_optimizer_steps,
            train_loss=f"{train_metrics['loss']:.6f}",
            train_accuracy=f"{train_metrics['top1']:.5f}",
            validation_loss=f"{validation_metrics['loss']:.6f}",
            validation_accuracy=f"{validation_metrics['top1']:.5f}",
            learning_rate=f"{float(optimizer.param_groups[0]['lr']):.6g}",
        )
        run.record(
            "epoch", epoch=epoch + 1, global_step=global_step,
            lr=optimizer.param_groups[0]["lr"], train=train_metrics,
            validation=validation_metrics,
        )
        print(
            f"epoch={epoch + 1}/{args.epochs} train_loss={train_metrics['loss']:.4f} "
            f"train_top1={train_metrics['top1']:.4f} val_loss={validation_metrics['loss']:.4f} "
            f"val_top1={validation_metrics['top1']:.4f}"
        )
        is_best = validation_metrics["top1"] > best_validation
        if is_best:
            best_validation = validation_metrics["top1"]
            save_checkpoint(
                best_path, model, args, epoch=epoch + 1, global_step=global_step,
                optimizer=optimizer, scheduler=scheduler, save_resume_state=False,
                best_validation_top1=best_validation,
            )
        latest_path = run.checkpoints_dir / "checkpoint_latest.safetensors"
        save_checkpoint(
            latest_path, model, args, epoch=epoch + 1, global_step=global_step,
            optimizer=optimizer, scheduler=scheduler, save_resume_state=True,
            best_validation_top1=best_validation,
        )
    if best_path.is_file():
        model.load_state_dict(load_file(str(best_path), device=str(device)), strict=True)
    test_metrics = evaluate(model, test_loader, device, amp=args.amp, max_batches=args.eval_batches)
    run.record("test", checkpoint=str(best_path), metrics=test_metrics)
    run.record("optimizer_state", **_optimizer_state_summary(optimizer))
    run.write_progress(
        global_step,
        status="completed",
        total_steps=total_optimizer_steps,
        test_loss=f"{test_metrics['loss']:.6f}",
        test_accuracy=f"{test_metrics['top1']:.5f}",
    )
    run.finish(status="completed", global_step=global_step, best_validation_top1=best_validation)
    print(f"test_top1={test_metrics['top1']:.4f} test_loss={test_metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
