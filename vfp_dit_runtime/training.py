"""Shared helpers for the VFP-DiT training entrypoints."""

from __future__ import annotations

import copy
import json
import math
import random
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Subset

from runtime.checkpoint import (
    load_training_state,
    make_training_state,
    restore_rng_state,
    save_training_state,
)
from runtime.data import build_dataloader_options
from runtime.device import resolve_device
from runtime.memory import maybe_collect_memory
from runtime.profiling import component_timer
from runtime.progress import RichProgress
from runtime.run import RunRecorder
from runtime.sampler import ResumableRandomSampler, ResumableWeightedRandomSampler
from runtime.signal import GracefulStop
from optimizers.factory import build_optimizer
from optimizers.lr_scheduler import add_lr_scheduler_arguments, build_lr_scheduler

from .data import TensorManifestDataset, collate_tensor_records, move_batch_to_device


def _checkpoint_stage_matches(actual: str | None, expected: str) -> bool:
    """Allow the former Simple stage id while keeping legacy VFCB stages rejected."""
    accepted = {expected}
    if expected == "vfp_dit.train":
        accepted.add("vfp_dit_simple.train")
    return actual in accepted


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_loss_components(total: float, metrics: dict[str, float]) -> str:
    if all(key in metrics for key in ("mse", "direction", "relational")):
        components = (
            ("mse", metrics["mse"], metrics["mse_weight"]),
            ("direction", metrics["direction"], metrics["direction_weight"]),
            ("relational", metrics["relational"], metrics["relational_weight"]),
        )
    elif "flow_mse" in metrics:
        components = (("flow_mse", metrics["flow_mse"], 1.0),)
    else:
        return "-"

    denominator = max(abs(total), 1e-12)
    return " ".join(
        f"{name}={value:.6f} ({100.0 * value * weight / denominator:.1f}%)"
        for name, value, weight in components
    )


def validate_common_training_args(args) -> None:
    if args.epochs <= 0 or args.batch_size <= 0 or args.grad_accumulation <= 0:
        raise ValueError("epochs, batch-size, and grad-accumulation must be positive")
    if (args.num_workers < 0 or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0
            or not math.isfinite(args.grad_clip) or args.grad_clip < 0):
        raise ValueError("num-workers, lr, weight-decay, or grad-clip has an invalid value")
    condition_dropout = float(getattr(args, "condition_dropout", 0.0))
    if not math.isfinite(condition_dropout) or not 0.0 <= condition_dropout < 1.0:
        raise ValueError("--condition-dropout must be in [0, 1)")
    samples_per_epoch = getattr(args, "samples_per_epoch", None)
    if samples_per_epoch is not None and samples_per_epoch <= 0:
        raise ValueError("--samples-per-epoch must be positive when provided")
    if getattr(args, "full_data_epoch", False) and samples_per_epoch is not None:
        raise ValueError("--full-data-epoch cannot be combined with --samples-per-epoch")
    validation_fraction = float(getattr(args, "validation_fraction", 0.0))
    if not math.isfinite(validation_fraction) or not 0.0 <= validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in [0, 1)")
    validation_samples = getattr(args, "validation_samples", 256)
    if validation_samples <= 0:
        raise ValueError("--validation-samples must be positive")
    if args.optimizer == "APOLLO" and args.apollo_rank <= 0:
        raise ValueError("--apollo-rank must be positive")
    anomaly_batch = int(getattr(args, "anomaly_detection_batch", 0))
    if anomaly_batch < 0:
        raise ValueError("--anomaly-detection-batch must be 0 or a positive batch index")
    if args.gc_interval < 0 or args.empty_cache_interval < 0:
        raise ValueError("--gc-interval and --empty-cache-interval must be >= 0")

def add_common_arguments(parser, *, default_output: str, default_optimizer: str = "AdamW") -> None:
    parser.add_argument("--data-mode", choices=("hf", "tensor"), default="hf")
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--multi-edit-data-root", default="data/MultiEdit")
    parser.add_argument("--coco-split", default="val")
    parser.add_argument("--edit-split", default="train")
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--coco-weight", type=float, default=1.0)
    parser.add_argument("--ti2i-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--samples-per-epoch", type=int, default=None,
        help="Optional number of replacement-sampled training examples per epoch",
    )
    parser.add_argument(
        "--full-data-epoch", action="store_true",
        help=(
            "Visit every training row exactly once per epoch; for HF data, rows "
            "are drawn without replacement in weighted random order"
        ),
    )
    parser.add_argument(
        "--validation-fraction", type=float, default=0.0,
        help="Optional per-source holdout fraction for epoch-end validation (0 disables validation)",
    )
    parser.add_argument(
        "--validation-samples", type=int, default=256,
        help="Validation examples evaluated per epoch; HF sampling uses source-level weights",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--gc-interval", type=int, default=10,
        help="Run Python garbage collection every N consumed batches (default: 10); 0 disables it",
    )
    parser.add_argument(
        "--empty-cache-interval", type=int, default=10,
        help=(
            "Release unused CUDA allocator cache every N consumed batches (default: 10); 0 disables it. "
            "This does not free live tensors and can reduce throughput"
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--condition-dropout", type=float, default=0.1,
                        help="Per-example probability of replacing text/reference conditioning with a null token")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=("no", "bf16"), default="bf16")
    parser.add_argument(
        "--optimizer", choices=("AdamW", "AdamW-SF", "APOLLO"),
        default=default_optimizer,
    )
    parser.add_argument("--apollo-rank", type=int, default=256)
    add_lr_scheduler_arguments(parser, default="cosine")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--check-finite-updates", action="store_true",
        help=(
            "Check gradients before and trainable parameters after every optimizer update"
        ),
    )
    parser.add_argument(
        "--anomaly-detection-batch", type=int, default=0,
        help=(
            "Enable PyTorch forward/backward anomaly tracing for this 1-based batch only; "
            "0 disables it"
        ),
    )


def apply_condition_dropout(
    condition_hidden: torch.Tensor,
    condition_mask: torch.Tensor,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace selected examples with one valid zero-input null-condition token."""
    if condition_hidden.ndim != 3 or condition_mask.shape != condition_hidden.shape[:2]:
        raise ValueError("condition hidden/mask must have shapes (B,L,D) and (B,L)")
    if condition_hidden.shape[1] <= 0 or not condition_mask.any(dim=-1).all():
        raise ValueError("Each condition sequence must contain a valid token")
    if not 0.0 <= probability < 1.0:
        raise ValueError("condition dropout probability must be in [0, 1)")
    if probability == 0.0:
        return condition_hidden, condition_mask, condition_hidden.new_zeros((), dtype=torch.float32)
    dropped = torch.rand(condition_hidden.shape[0], device=condition_hidden.device) < probability
    condition = condition_hidden.clone()
    mask = condition_mask.clone()
    condition[dropped] = 0
    mask[dropped] = False
    mask[dropped, 0] = True
    return condition, mask, dropped.float().mean()

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_dataset(dataset, validation_fraction: float, seed: int):
    """Create a deterministic row-level holdout without assuming named splits."""
    if validation_fraction == 0.0:
        return dataset, None
    if len(dataset) < 2:
        raise ValueError("validation holdout requires at least two rows")
    validation_size = max(1, round(len(dataset) * validation_fraction))
    if validation_size >= len(dataset):
        raise ValueError(
            f"validation fraction {validation_fraction} leaves no training rows "
            f"for a source of length {len(dataset)}"
        )
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed),
    ).tolist()
    validation_indices = sorted(indices[:validation_size])
    train_indices = sorted(indices[validation_size:])
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def _cap_validation_dataset(dataset, limit: int, seed: int):
    if len(dataset) <= limit:
        return dataset
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed),
    )[:limit].tolist()
    return Subset(dataset, sorted(indices))


def make_loader(args, required_keys: tuple[str, ...]):
    def loader_options(stream: int):
        options = build_dataloader_options(
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available() and args.data_mode == "tensor",
            seed=args.seed,
            stream=stream,
        )
        # Workers may start only after the model has initialized CUDA. Select
        # spawn based on the requested device, including CPU-encoder offload runs.
        if args.num_workers > 0 and resolve_device(args.device).type == "cuda":
            options["multiprocessing_context"] = "spawn"
        return options

    validation_fraction = float(getattr(args, "validation_fraction", 0.0))
    validation_samples = int(getattr(args, "validation_samples", 256))
    samples_per_epoch = getattr(args, "samples_per_epoch", None)
    full_data_epoch = bool(getattr(args, "full_data_epoch", False))
    validation_dataset = None
    validation_loader = None

    if full_data_epoch and args.data_mode == "hf" and args.batch_size != 1:
        raise ValueError("--full-data-epoch currently requires --batch-size 1 for HF data")

    if args.data_mode == "tensor":
        if not args.train_manifest:
            raise ValueError("--train-manifest is required when --data-mode=tensor")
        full_dataset = TensorManifestDataset(args.train_manifest, required_keys=required_keys)
        dataset, validation_dataset = _split_dataset(
            full_dataset, validation_fraction, args.seed,
        )
        sampler = None
        shuffle = True
        if full_data_epoch:
            sampler = ResumableRandomSampler(dataset, seed=args.seed)
            shuffle = False
        elif samples_per_epoch is not None:
            sampler = ResumableWeightedRandomSampler(
                torch.ones(len(dataset), dtype=torch.double),
                num_samples=samples_per_epoch,
                replacement=True,
                seed=args.seed,
            )
            shuffle = False
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=False,
            collate_fn=collate_tensor_records,
            **loader_options(stream=0),
        )
        if validation_dataset is not None:
            validation_eval_dataset = _cap_validation_dataset(
                validation_dataset, validation_samples, args.seed + 1_000_003,
            )
            validation_loader = DataLoader(
                validation_eval_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_tensor_records,
                **loader_options(stream=1),
            )
        return dataset, loader, validation_dataset, validation_loader

    from .hf_data import (
        AspectResolutionBatchSampler, AspectResolutionDataset,
        collate_hf_image_examples, load_hf_image_sources, make_mixed_sampler,
    )

    if args.resolution <= 0 or args.resolution % 16:
        raise ValueError("resolution must be a positive multiple of 16")
    coco, ti2i = load_hf_image_sources(
        multi_edit_data_root=args.multi_edit_data_root,
        edit_split=args.edit_split,
        coco_split=args.coco_split,
        cache_dir=args.hf_cache_dir,
    )
    train_coco, validation_coco = _split_dataset(
        coco, validation_fraction, args.seed,
    )
    train_ti2i, validation_ti2i = _split_dataset(
        ti2i, validation_fraction, args.seed + 1,
    )
    dataset, sampler = make_mixed_sampler(
        train_coco, train_ti2i,
        coco_weight=args.coco_weight,
        ti2i_weight=args.ti2i_weight,
        seed=args.seed,
        num_samples=samples_per_epoch,
        replacement=not full_data_epoch,
    )
    resolution_levels = getattr(args, "resolution_levels", None)
    aspect_ratios = getattr(args, "aspect_ratios", None)
    alignment = int(getattr(args, "bucket_alignment", 32))
    collate_fn = partial(
        collate_hf_image_examples,
        resolution=args.resolution,
        resolution_levels=resolution_levels,
        aspect_ratios=aspect_ratios or (0.5, 0.5625, 2 / 3, 0.75, 1.0, 4 / 3, 1.5, 16 / 9, 2.0),
        alignment=alignment,
    )
    if resolution_levels and args.batch_size > 1:
        with RichProgress(
            total=len(dataset), description="indexing image aspect buckets",
        ) as index_progress:
            mixed = AspectResolutionDataset(
                dataset,
                resolution_levels=resolution_levels,
                aspect_ratios=aspect_ratios,
                alignment=alignment,
                progress_callback=lambda advance: index_progress.update(advance),
            )
        sampler = AspectResolutionBatchSampler(
            mixed, sampler.weights,
            num_samples=sampler.num_samples,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        dataset = mixed
        loader = DataLoader(
            dataset, batch_sampler=sampler,
            collate_fn=collate_fn, **loader_options(stream=0),
        )
    else:
        if not (resolution_levels and args.batch_size > 1):
            sampler = ResumableWeightedRandomSampler(
                sampler.weights,
                num_samples=sampler.num_samples,
                replacement=sampler.replacement,
                seed=args.seed,
            )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            drop_last=False,
            collate_fn=collate_fn,
            **loader_options(stream=0),
        )
    if validation_coco is not None and validation_ti2i is not None:
        validation_dataset, validation_sampler = make_mixed_sampler(
            validation_coco, validation_ti2i,
            coco_weight=args.coco_weight,
            ti2i_weight=args.ti2i_weight,
            seed=args.seed + 1_000_003,
            num_samples=validation_samples,
        )
        validation_options = loader_options(stream=1)
        # Recreate workers each epoch so their Python RNG follows the seeded
        # validation generator and caption selection is reproducible.
        validation_options["persistent_workers"] = False
        if resolution_levels and args.batch_size > 1:
            with RichProgress(
                total=len(validation_dataset),
                description="indexing validation aspect buckets",
            ) as index_progress:
                validation_dataset = AspectResolutionDataset(
                    validation_dataset,
                    resolution_levels=resolution_levels,
                    aspect_ratios=aspect_ratios,
                    alignment=alignment,
                    progress_callback=lambda advance: index_progress.update(advance),
                )
            validation_batch_sampler = AspectResolutionBatchSampler(
                validation_dataset, validation_sampler.weights,
                num_samples=validation_sampler.num_samples,
                batch_size=args.batch_size,
                seed=args.seed + 1_000_003,
            )
            validation_loader = DataLoader(
                validation_dataset, batch_sampler=validation_batch_sampler,
                collate_fn=collate_fn, **validation_options,
            )
        else:
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=args.batch_size,
                sampler=validation_sampler,
                drop_last=False,
                collate_fn=collate_fn,
                **validation_options,
            )
    return dataset, loader, validation_dataset, validation_loader


def autocast_context(device: torch.device, amp: str):
    if amp == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def sampled_relational_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    pairs_per_bucket: int = 64,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compare cosine geometry using bounded, sampled spatial token pairs."""
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be 'mean' or 'none'")
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("relational features must have matching (B,C,H,W) shapes")
    batch, _channels, height, width = prediction.shape
    tokens = height * width
    if tokens < 2:
        if reduction == "none":
            return prediction.new_zeros((batch,))
        return prediction.new_zeros(())
    pred = F_normalize_tokens(prediction.flatten(2).transpose(1, 2))
    truth = F_normalize_tokens(target.flatten(2).transpose(1, 2))
    device = prediction.device
    pairs = min(pairs_per_bucket, tokens)
    losses = []

    # Local and intermediate buckets enumerate only a small offset window, not
    # the full token-by-token relation matrix.
    offset_sets = (
        [(dy, dx) for dy in range(-1, 2) for dx in range(-1, 2)
         if 0 < dy * dy + dx * dx < 4],
        [(dy, dx) for dy in range(-8, 9) for dx in range(-8, 9)
         if 4 <= dy * dy + dx * dx < 64],
    )
    for offsets in offset_sets:
        offsets = [(dy, dx) for dy, dx in offsets if abs(dy) < height and abs(dx) < width]
        if not offsets:
            continue
        offset_tensor = torch.tensor(offsets, device=device, dtype=torch.long)
        candidate_count = pairs * 8
        anchors = torch.randint(tokens, (candidate_count,), device=device)
        chosen = offset_tensor[torch.randint(len(offsets), (candidate_count,), device=device)]
        anchor_y = torch.div(anchors, width, rounding_mode="floor")
        anchor_x = anchors.remainder(width)
        other_y = anchor_y + chosen[:, 0]
        other_x = anchor_x + chosen[:, 1]
        valid = (other_y >= 0) & (other_y < height) & (other_x >= 0) & (other_x < width)
        left = anchors[valid][:pairs]
        right = (other_y[valid] * width + other_x[valid])[:pairs]
        if left.numel() == 0:
            continue
        if left.numel() < pairs:
            repeat = torch.randint(left.numel(), (pairs,), device=device)
            left, right = left[repeat], right[repeat]
        pred_relation = (pred[:, left] * pred[:, right]).sum(-1)
        target_relation = (truth[:, left] * truth[:, right]).sum(-1)
        squared_error = (pred_relation.float() - target_relation.float()).square()
        losses.append(squared_error.mean() if reduction == "mean" else squared_error.mean(-1))

    # Global pairs are sampled directly and filtered by spatial distance.
    if max(height, width) > 8:
        candidate_count = max(pairs * 16, 256)
        left = torch.randint(tokens, (candidate_count,), device=device)
        right = torch.randint(tokens, (candidate_count,), device=device)
        left_y = torch.div(left, width, rounding_mode="floor")
        left_x = left.remainder(width)
        right_y = torch.div(right, width, rounding_mode="floor")
        right_x = right.remainder(width)
        far = ((left_y - right_y).square() + (left_x - right_x).square()) >= 64
        left, right = left[far][:pairs], right[far][:pairs]
        if left.numel() > 0:
            if left.numel() < pairs:
                repeat = torch.randint(left.numel(), (pairs,), device=device)
                left, right = left[repeat], right[repeat]
            pred_relation = (pred[:, left] * pred[:, right]).sum(-1)
            target_relation = (truth[:, left] * truth[:, right]).sum(-1)
            squared_error = (pred_relation.float() - target_relation.float()).square()
            losses.append(squared_error.mean() if reduction == "mean" else squared_error.mean(-1))

    if not losses:
        left = torch.arange(tokens - 1, device=device)
        right = left + 1
        squared_error = (
            (pred[:, left] * pred[:, right]).sum(-1)
            - (truth[:, left] * truth[:, right]).sum(-1)
        ).float().square()
        losses.append(squared_error.mean() if reduction == "mean" else squared_error.mean(-1))
    return torch.stack(losses).mean(0).to(dtype=prediction.dtype)


def F_normalize_tokens(value: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(value.float(), dim=-1, eps=1e-6).to(value.dtype)


def vfcb_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    timestep: torch.Tensor,
    *,
    mse_min: float = 0.25,
    mse_max: float = 1.0,
    direction_min: float = 0.25,
    direction_max: float = 1.0,
    relational_min: float = 0.1,
    relational_max: float = 0.5,
    pairs_per_bucket: int = 64,
    reduction: str = "mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be 'mean' or 'none'")
    if prediction.shape != target.shape:
        raise ValueError(
            f"VFCB output/teacher shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    per_example = reduction == "none"
    squared_error = (prediction.float() - target.float()).square()
    mse = squared_error.flatten(1).mean(-1) if per_example else squared_error.mean()
    pred_tokens = F_normalize_tokens(prediction.flatten(2).transpose(1, 2))
    target_tokens = F_normalize_tokens(target.flatten(2).transpose(1, 2))
    cosine = (pred_tokens.float() * target_tokens.float()).sum(-1)
    direction = 1.0 - (cosine.mean(-1) if per_example else cosine.mean())
    relational = sampled_relational_loss(
        prediction, target, pairs_per_bucket=pairs_per_bucket, reduction=reduction,
    )
    recoverability = (1.0 - timestep.float().reshape(-1)).clamp(0, 1)
    recoverable = recoverability if per_example else recoverability.mean()
    mse_weight = mse_min + (mse_max - mse_min) * recoverable
    direction_weight = direction_min + (direction_max - direction_min) * (1.0 - recoverable)
    relational_weight = relational_min + (relational_max - relational_min) * (1.0 - recoverable)
    loss = mse_weight * mse + direction_weight * direction + relational_weight * relational
    return loss, {
        "mse": mse,
        "direction": direction,
        "relational": relational,
        "mse_weight": mse_weight,
        "direction_weight": direction_weight,
        "relational_weight": relational_weight,
    }


def save_model_checkpoint(
    model: torch.nn.Module,
    path: Path,
    *,
    metadata: dict[str, Any],
    optimizer,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    scheduler=None,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(tensors, str(path), metadata={
        "vfp_dit.checkpoint": json.dumps({"network_version": 1, **metadata}, sort_keys=True),
        "vfp_dit.config": json.dumps(config, sort_keys=True),
        "epoch": str(epoch),
        "global_step": str(global_step),
    })
    resume_state = make_training_state(
        optimizer=optimizer,
        epoch=epoch,
        global_step=global_step,
        scheduler=scheduler,
        extra=extra,
    )
    save_training_state(path, resume_state)


RESUME_RUNTIME_CONFIG_KEYS = {
    "device", "dry_run", "hf_cache_dir", "init_checkpoint", "output_dir",
    "resume", "run_name", "validate_only", "profile_components",
    "gradient_checkpointing",
    "encoder_device", "vae_device", "encoder_prefetch_batches",
    "observe_interval", "sample_steps", "sample_prompt", "sample_guidance_scale",
    "no_observe_samples", "log_metadata_diagnostics", "adapter_heads",
    "check_finite_updates", "anomaly_detection_batch",
    "gc_interval", "empty_cache_interval",
}


def _normalize_resume_config_value(value: Any) -> Any:
    """Normalize JSON lists and Python tuples to the same config value."""
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_resume_config_value(item) for item in value)
    if isinstance(value, dict):
        return {
            key: _normalize_resume_config_value(item)
            for key, item in value.items()
        }
    return value


def _debug_batch_identity(batch: Any) -> str:
    """Return compact source/sample identifiers for finite-check errors."""
    if not isinstance(batch, dict):
        return "batch_identity=unavailable"
    fields = []
    for key in ("conditioning_types", "sample_ids", "sources"):
        value = batch.get(key)
        if value is not None:
            if isinstance(value, (list, tuple)):
                value = list(value)
            fields.append(f"{key}={value!r}")
    return " ".join(fields) if fields else "batch_identity=unavailable"


def _nonfinite_gradient_names(model: torch.nn.Module) -> list[str]:
    return [
        name for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item()
    ]


def _nonfinite_parameter_names(model: torch.nn.Module) -> list[str]:
    return [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and not torch.isfinite(parameter).all().item()
    ]


def _all_finite_gradients(model: torch.nn.Module) -> bool:
    checks = [
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return not checks or bool(torch.stack(checks).all().item())


def _all_finite_parameters(model: torch.nn.Module) -> bool:
    checks = [
        torch.isfinite(parameter).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return not checks or bool(torch.stack(checks).all().item())


def validate_resume_config(saved: dict[str, Any], current: dict[str, Any]) -> None:
    """Reject silent changes to the data, model, and optimization contract."""
    if not saved:
        print("Warning: resume checkpoint has no saved config metadata; config drift cannot be checked.")
        return
    legacy_defaults = {
        "condition_dropout": 0.0,
        "timesteps_per_image": 1,
        "vfcb_ff_mult": 3.0,
        "validation_fraction": 0.0,
        "validation_samples": 256,
        # Added in VFP-DiT Simple: absent fields identify the pre-change block.
        "attention_head_gate": "timestep_sigmoid",
        "metadata_ffn_residual_gate": False,
        "metadata_ffn_gate_mapping": "linear",
        "fuse_same_input_projections": False,
        # Legacy checkpoints used a direct latent-channel prediction head.
        "output_refinement_depth": 0,
        "output_skip_fusion_mode": "concat_linear",
        # Older output heads were linear and had no metadata-scaled correction.
        "output_head_ada_scale": False,
        # Earlier TI2I checkpoints fused a strided latent into the Qwen grid.
        "reference_latent_fusion_mode": "legacy_latent_to_qwen",
    }
    differences = []
    for key in sorted(set(saved) | set(current)):
        if key in RESUME_RUNTIME_CONFIG_KEYS:
            continue
        if key == "target_latent_downsample_factor":
            # Older checkpoints used latent_downsample_factor for both the
            # target path and reference-latent path.
            saved_value = saved.get(
                key, saved.get("latent_downsample_factor", legacy_defaults.get(key, 1)),
            )
        else:
            saved_value = saved.get(key, legacy_defaults.get(key))
        current_value = current.get(key, legacy_defaults.get(key))
        if (
            _normalize_resume_config_value(saved_value)
            != _normalize_resume_config_value(current_value)
        ):
            differences.append(f"{key}: checkpoint={saved_value!r}, requested={current_value!r}")
    if differences:
        raise ValueError(
            "Resume config differs from the checkpoint: " + "; ".join(differences)
            + ". Keep the saved training settings or use --init-checkpoint for an intentional change.",
        )

def restore_training(
    args,
    model: torch.nn.Module,
    optimizer,
    device: torch.device,
    scheduler=None,
    *,
    expected_stage: str,
) -> tuple[int, int]:
    if not args.resume:
        return 0, 0
    path = Path(args.resume).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        checkpoint_metadata = checkpoint.metadata() or {}
    metadata = json.loads(checkpoint_metadata.get("vfp_dit.checkpoint", "{}"))
    saved_config_text = checkpoint_metadata.get("vfp_dit.config")
    saved_config = None
    if saved_config_text is None:
        print(f"Warning: checkpoint has no vfp_dit.config metadata: {path}")
    else:
        saved_config = json.loads(saved_config_text)
        validate_resume_config(saved_config, vars(args))
    if metadata.get("network_version") != 1:
        raise ValueError(f"Unsupported VFP-DiT checkpoint version: {metadata.get('network_version')!r}")
    if not _checkpoint_stage_matches(metadata.get("stage"), expected_stage):
        raise ValueError(f"Expected a {expected_stage} checkpoint, got {metadata.get('stage')!r}")
    weights = load_file(str(path), device=str(device))
    compatible_loader = getattr(model, "load_checkpoint_state_dict", None)
    if compatible_loader is None:
        model.load_state_dict(weights, strict=True)
    else:
        dropped_adapter_attention = compatible_loader(
            weights,
            adapter_type=(saved_config or {}).get("adapter_type"),
        )
        if dropped_adapter_attention:
            raise ValueError(
                "This checkpoint contains retired adapter attention tensors. Its optimizer "
                "state cannot be resumed after the adapter simplification; use "
                "--init-checkpoint to start a fresh optimizer from its compatible weights."
            )
    state = load_training_state(path)
    if state is None:
        args._resume_epoch_in_progress = False
        args._resume_sampler_state = None
        print(f"Loaded model weights only (no resume sidecar): {path}")
        return 0, 0
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    restore_rng_state(state["rng"])
    extra = state.get("extra", {})
    args._resume_epoch_in_progress = bool(extra.get("epoch_in_progress", False))
    args._resume_sampler_state = extra.get("sampler_state")
    print(f"Restored optimizer and RNG state: {path}")
    return int(state["epoch"]), int(state["global_step"])


def initialize_matching_weights(model: torch.nn.Module, path_value: str, *, expected_stage: str, device: torch.device) -> None:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Initialization checkpoint not found: {path}")
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        checkpoint_metadata = checkpoint.metadata() or {}
        metadata = json.loads(checkpoint_metadata.get("vfp_dit.checkpoint", "{}"))
        source_config_text = checkpoint_metadata.get("vfp_dit.config")
    source_config = json.loads(source_config_text) if source_config_text else None
    if not _checkpoint_stage_matches(metadata.get("stage"), expected_stage):
        raise ValueError(f"Expected a {expected_stage} initialization checkpoint")
    # Keep the source checkpoint off the accelerator while folding or matching
    # tensors; model.load_state_dict copies compatible weights to its device.
    source = load_file(str(path), device="cpu")
    compatible_loader = getattr(model, "load_initialization_state_dict", None)
    if compatible_loader is not None:
        loaded, missing = compatible_loader(source, source_config=source_config)
        print(f"Initialized {loaded} tensors; {missing} current tensors remain initialized")
        return
    destination = model.state_dict()
    compatible = {
        key: value for key, value in source.items()
        if key in destination and tuple(value.shape) == tuple(destination[key].shape)
    }
    if not compatible:
        raise ValueError(f"No matching parameter shapes in initialization checkpoint: {path}")
    missing, _unexpected = model.load_state_dict(compatible, strict=False)
    print(f"Initialized {len(compatible)} tensors; {len(missing)} current tensors remain initialized")


def _update_feature_statistics(
    state: dict[str, Any] | None,
    prediction: torch.Tensor,
    teacher: torch.Tensor,
) -> dict[str, Any]:
    """Accumulate float64 sufficient statistics for sampled spatial features."""
    if prediction.shape != teacher.shape or prediction.ndim != 3:
        raise ValueError("feature samples must have matching (B,S,C) shapes")
    prediction = prediction.detach().reshape(-1, prediction.shape[-1]).to(device="cpu", dtype=torch.float64)
    teacher = teacher.detach().reshape(-1, teacher.shape[-1]).to(device="cpu", dtype=torch.float64)
    if state is None:
        channels = prediction.shape[-1]
        state = {
            "count": 0,
            "element_count": 0,
            "prediction_sum": torch.zeros(channels, dtype=torch.float64),
            "teacher_sum": torch.zeros(channels, dtype=torch.float64),
            "prediction_cross": torch.zeros((channels, channels), dtype=torch.float64),
            "teacher_cross": torch.zeros((channels, channels), dtype=torch.float64),
            "cross": torch.zeros((channels, channels), dtype=torch.float64),
            "prediction_square_sum": 0.0,
            "teacher_square_sum": 0.0,
        }
    if prediction.shape[-1] != state["prediction_sum"].numel():
        raise ValueError("feature channel count changed during validation")
    state["count"] += prediction.shape[0]
    state["element_count"] += prediction.numel()
    state["prediction_sum"] += prediction.sum(0)
    state["teacher_sum"] += teacher.sum(0)
    state["prediction_cross"] += prediction.T @ prediction
    state["teacher_cross"] += teacher.T @ teacher
    state["cross"] += prediction.T @ teacher
    state["prediction_square_sum"] += prediction.square().sum().item()
    state["teacher_square_sum"] += teacher.square().sum().item()
    return state


def _finish_feature_statistics(state: dict[str, Any]) -> dict[str, float]:
    count = state["count"]
    if count < 2:
        raise ValueError("at least two sampled feature tokens are required for covariance diagnostics")
    prediction_cov = state["prediction_cross"] - torch.outer(
        state["prediction_sum"], state["prediction_sum"],
    ) / count
    teacher_cov = state["teacher_cross"] - torch.outer(
        state["teacher_sum"], state["teacher_sum"],
    ) / count
    cross_cov = state["cross"] - torch.outer(
        state["prediction_sum"], state["teacher_sum"],
    ) / count
    prediction_cov = (prediction_cov + prediction_cov.T) * (0.5 / (count - 1))
    teacher_cov = (teacher_cov + teacher_cov.T) * (0.5 / (count - 1))
    cross_cov = cross_cov / (count - 1)

    prediction_eigenvalues = torch.linalg.eigvalsh(prediction_cov).flip(0).clamp_min(0)
    teacher_eigenvalues = torch.linalg.eigvalsh(teacher_cov).flip(0).clamp_min(0)
    prediction_variance = prediction_eigenvalues.sum()
    teacher_variance = teacher_eigenvalues.sum()
    tiny = torch.finfo(torch.float64).tiny
    prediction_prob = prediction_eigenvalues / prediction_variance.clamp_min(tiny)
    teacher_prob = teacher_eigenvalues / teacher_variance.clamp_min(tiny)
    prediction_effective_rank = torch.where(
        prediction_variance > tiny,
        torch.exp(-(prediction_prob * prediction_prob.clamp_min(1e-300).log()).sum()),
        prediction_variance.new_zeros(()),
    )
    teacher_effective_rank = torch.where(
        teacher_variance > tiny,
        torch.exp(-(teacher_prob * teacher_prob.clamp_min(1e-300).log()).sum()),
        teacher_variance.new_zeros(()),
    )
    cka = cross_cov.square().sum() / torch.sqrt(
        prediction_cov.square().sum() * teacher_cov.square().sum()
    ).clamp_min(tiny)
    prediction_rms = (state["prediction_square_sum"] / state["element_count"]) ** 0.5
    teacher_rms = (state["teacher_square_sum"] / state["element_count"]) ** 0.5

    metrics = {
        "feature_token_count": float(count),
        "feature_rms_ratio": prediction_rms / max(teacher_rms, 1e-30),
        "feature_linear_cka": float(cka.item()),
        "prediction_covariance_trace": float(prediction_variance.item()),
        "teacher_covariance_trace": float(teacher_variance.item()),
        "prediction_effective_rank": float(prediction_effective_rank.item()),
        "teacher_effective_rank": float(teacher_effective_rank.item()),
    }
    for index in range(min(16, prediction_eigenvalues.numel())):
        metrics[f"prediction_spectrum_{index + 1:02d}"] = float(
            (prediction_prob[index]).item()
        )
        metrics[f"teacher_spectrum_{index + 1:02d}"] = float(teacher_prob[index].item())
    return metrics


def _start_performance_window(device: torch.device) -> tuple[float, float | None, float | None]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start_allocated = torch.cuda.memory_allocated(device)
        start_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        start_allocated = None
        start_reserved = None
    return perf_counter(), start_allocated, start_reserved


def _finish_performance_window(
    device: torch.device,
    started: float,
    start_allocated: float | None,
    start_reserved: float | None,
) -> tuple[float, dict[str, float]]:
    if device.type != "cuda":
        return perf_counter() - started, {}
    torch.cuda.synchronize(device)
    bytes_per_gib = 1024 ** 3
    memory = {
        "start_allocated_gib": float(start_allocated or 0) / bytes_per_gib,
        "start_reserved_gib": float(start_reserved or 0) / bytes_per_gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / bytes_per_gib,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / bytes_per_gib,
        "end_allocated_gib": torch.cuda.memory_allocated(device) / bytes_per_gib,
        "end_reserved_gib": torch.cuda.memory_reserved(device) / bytes_per_gib,
    }
    return perf_counter() - started, memory


def _iter_nested_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_nested_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_nested_tensors(item)


def _unique_device_storage_bytes(tensors, device: torch.device) -> int:
    """Count unique tensor storages on a device, avoiding view double counts."""
    target_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    seen: set[tuple[str, int, int]] = set()
    total = 0
    for tensor in tensors:
        if (
            tensor.device.type != device.type
            or tensor.device.index != target_index
            or tensor.numel() == 0
        ):
            continue
        storage = tensor.untyped_storage()
        size = storage.nbytes()
        pointer = storage.data_ptr()
        key = (str(tensor.device), pointer, size)
        if size and key not in seen:
            seen.add(key)
            total += size
    return total


def _training_memory_breakdown(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    peak_allocated_gib: float,
) -> dict[str, float]:
    """Report persistent GPU storages and the unclassified peak remainder.

    The peak remainder is not pure activation memory: it also includes live
    gradients, kernel workspaces, and other temporary tensors.
    """
    if device.type != "cuda":
        return {}
    parameters = list(model.parameters())
    parameter_bytes = _unique_device_storage_bytes(parameters, device)
    buffer_bytes = _unique_device_storage_bytes(model.buffers(), device)
    optimizer_state_bytes = _unique_device_storage_bytes(
        (
            tensor
            for state in optimizer.state.values()
            for tensor in _iter_nested_tensors(state)
        ),
        device,
    )
    gradient_bytes = _unique_device_storage_bytes(
        (parameter.grad for parameter in parameters if parameter.grad is not None),
        device,
    )
    persistent_bytes = (
        parameter_bytes + buffer_bytes + optimizer_state_bytes + gradient_bytes
    )
    bytes_per_gib = 1024 ** 3
    peak_bytes = int(peak_allocated_gib * bytes_per_gib)
    current_allocated_bytes = torch.cuda.memory_allocated(device)
    return {
        "model_parameters_gib": parameter_bytes / bytes_per_gib,
        "model_buffers_gib": buffer_bytes / bytes_per_gib,
        "optimizer_state_gib": optimizer_state_bytes / bytes_per_gib,
        "gradients_at_epoch_end_gib": gradient_bytes / bytes_per_gib,
        "peak_overhead_over_persistent_gib": max(peak_bytes - persistent_bytes, 0) / bytes_per_gib,
        "end_unclassified_allocated_gib": max(
            current_allocated_bytes - persistent_bytes, 0,
        ) / bytes_per_gib,
    }


def evaluate_epoch(model, loader, step_fn, args, device, *, epoch: int) -> dict[str, float]:
    """Evaluate a fixed seeded validation stream with conditioning dropout disabled."""
    eval_args = copy.copy(args)
    if hasattr(eval_args, "condition_dropout"):
        eval_args.condition_dropout = 0.0

    python_rng = random.getstate()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    try:
        validation_seed = args.seed + 1_000_003
        seed_everything(validation_seed)
        if loader.generator is not None:
            loader.generator.manual_seed(validation_seed)
        sampler_generator = getattr(loader.sampler, "generator", None)
        if sampler_generator is not None:
            sampler_generator.manual_seed(validation_seed)
        batch_sampler_set_epoch = getattr(loader.batch_sampler, "set_epoch", None)
        if callable(batch_sampler_set_epoch):
            batch_sampler_set_epoch(validation_seed)

        model.eval()
        totals: dict[str, float] = {}
        condition_loss_totals: dict[str, float] = {}
        condition_loss_counts: dict[str, int] = {}
        binned_samples: dict[str, list[torch.Tensor]] = {}
        feature_statistics = None
        sample_count = 0
        performance_started, start_allocated, start_reserved = _start_performance_window(device)
        with RichProgress(
            total=len(loader), description=f"validation epoch {epoch + 1}/{args.epochs}",
        ) as progress:
            for raw_batch in loader:
                batch = (raw_batch if args.data_mode == "hf"
                         else move_batch_to_device(raw_batch, device))
                batch_size = (
                    len(raw_batch["prompts"]) if args.data_mode == "hf"
                    else int(raw_batch["clean_latent"].shape[0])
                )
                with torch.no_grad(), autocast_context(device, args.amp):
                    _loss, metrics = step_fn(model, batch, eval_args, device)
                if not torch.isfinite(_loss):
                    raise FloatingPointError(
                        f"Non-finite validation loss at epoch={epoch + 1}"
                    )
                if "prediction_feature_samples" in metrics:
                    if "teacher_feature_samples" not in metrics:
                        raise ValueError("teacher feature samples are required with prediction samples")
                    feature_statistics = _update_feature_statistics(
                        feature_statistics,
                        metrics["prediction_feature_samples"],
                        metrics["teacher_feature_samples"],
                    )
                for key, value in metrics.items():
                    if key in {"prediction_feature_samples", "teacher_feature_samples"}:
                        continue
                    if key == "per_example_loss":
                        per_example = value.detach().float().reshape(-1).cpu()
                        condition_types = raw_batch.get("conditioning_types")
                        if per_example.numel() != batch_size:
                            raise ValueError(
                                "per_example_loss must contain one value per validation example"
                            )
                        if condition_types is not None:
                            if len(condition_types) != batch_size:
                                raise ValueError(
                                    "conditioning_types must match validation batch size"
                                )
                            for sample_loss, condition_type in zip(
                                per_example.tolist(), condition_types, strict=True,
                            ):
                                if not isinstance(condition_type, str):
                                    raise TypeError("conditioning_types entries must be strings")
                                label = condition_type.lower()
                                if not label.replace("_", "").isalnum():
                                    raise ValueError(
                                        f"invalid conditioning type for metric name: {condition_type!r}"
                                    )
                                condition_loss_totals[label] = (
                                    condition_loss_totals.get(label, 0.0) + sample_loss
                                )
                                condition_loss_counts[label] = (
                                    condition_loss_counts.get(label, 0) + 1
                                )
                        continue
                    if key.endswith("_samples"):
                        binned_samples.setdefault(key.removesuffix("_samples"), []).append(
                            value.detach().float().reshape(-1).cpu()
                        )
                        continue
                    metric_value = float(value.detach().float().mean().item())
                    totals[key] = totals.get(key, 0.0) + metric_value * batch_size
                sample_count += batch_size
                progress.update(advance=1, postfix=f"val_loss={float(_loss):.6f}")
        if sample_count == 0:
            raise ValueError("Validation loader produced no examples")
        result = {key: value / sample_count for key, value in totals.items()}
        for condition_type, total in condition_loss_totals.items():
            count = condition_loss_counts[condition_type]
            result[f"loss_{condition_type}"] = total / count
            result[f"loss_{condition_type}_samples"] = float(count)
        if feature_statistics is not None:
            result.update(_finish_feature_statistics(feature_statistics))
        dropout_conditional = binned_samples.get("dropout_conditional_loss")
        dropout_unconditional = binned_samples.get("dropout_unconditional_loss")
        if dropout_conditional and dropout_unconditional:
            conditional_values = torch.cat(dropout_conditional)
            unconditional_values = torch.cat(dropout_unconditional)
            result["dropout_conditional_loss"] = float(conditional_values.mean().item())
            result["dropout_unconditional_loss"] = float(unconditional_values.mean().item())
            result["dropout_loss_delta"] = float(
                (unconditional_values - conditional_values).mean().item()
            )
            for component in ("mse", "direction", "relational"):
                component_delta = binned_samples.get(f"dropout_{component}_delta")
                if component_delta:
                    result[f"dropout_{component}_delta"] = float(
                        torch.cat(component_delta).mean().item()
                    )
        if "timestep" in binned_samples:
            timestep = torch.cat(binned_samples["timestep"])
            timestep_edges = torch.tensor([0.2, 0.4, 0.6, 0.8])
            timestep_bin = torch.bucketize(timestep, timestep_edges, right=True)
            timestep_labels = ("00_20", "20_40", "40_60", "60_80", "80_100")
            component_names = (
                "mse", "direction", "relational",
                "dropout_mse_delta", "dropout_direction_delta", "dropout_relational_delta",
            )
            for bin_index, label in enumerate(timestep_labels):
                selected = timestep_bin == bin_index
                result[f"tbin_{label}_count"] = float(selected.sum().item())
                if selected.any():
                    for component in component_names:
                        component_batches = binned_samples.get(component)
                        if component_batches:
                            values = torch.cat(component_batches)
                            result[f"tbin_{label}_{component}"] = float(values[selected].mean().item())

            # For x_t=(1-t)x_0+t*eps, coefficient log-SNR is
            # log((1-t)^2/t^2) = 2*(log1p(-t)-log(t)).
            log_snr = 2.0 * (torch.log1p(-timestep) - torch.log(timestep))
            log_snr_edges = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0])
            log_snr_bin = torch.bucketize(log_snr, log_snr_edges, right=True)
            log_snr_labels = ("le_m4", "m4_m2", "m2_0", "0_2", "2_4", "ge_4")
            for bin_index, label in enumerate(log_snr_labels):
                selected = log_snr_bin == bin_index
                result[f"snrbin_{label}_count"] = float(selected.sum().item())
                if selected.any():
                    for component in component_names:
                        component_batches = binned_samples.get(component)
                        if component_batches:
                            values = torch.cat(component_batches)
                            result[f"snrbin_{label}_{component}"] = float(values[selected].mean().item())
        validation_seconds, validation_memory = _finish_performance_window(
            device, performance_started, start_allocated, start_reserved,
        )
        result["seconds"] = validation_seconds
        result["samples"] = float(sample_count)
        result["samples_per_second"] = sample_count / max(validation_seconds, 1e-12)
        result.update(validation_memory)
        return result
    finally:
        model.train(was_training)
        random.setstate(python_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _iter_prepared_batches(loader, *, prepare_batch_fn, prefetch_batches: int):
    """Encode batches on one background thread with a strict bounded queue.

    The sampler cursor is advanced only by the training loop after a yielded
    batch is consumed. Raw rows fetched ahead by this iterator therefore do
    not advance the checkpoint position.
    """
    iterator = iter(loader)
    if prepare_batch_fn is None:
        for raw_batch in iterator:
            yield raw_batch, raw_batch
        return
    if prefetch_batches <= 0:
        for raw_batch in iterator:
            yield raw_batch, prepare_batch_fn(raw_batch)
        return

    pending: deque[tuple[Any, Future]] = deque()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vfp-encode")

    def submit_one() -> bool:
        try:
            raw = next(iterator)
        except StopIteration:
            return False
        pending.append((raw, executor.submit(prepare_batch_fn, raw)))
        return True

    try:
        for _ in range(prefetch_batches):
            if not submit_one():
                break
        while pending:
            raw_batch, future = pending.popleft()
            # Refill before waiting so CPU encoding overlaps the current step.
            submit_one()
            yield raw_batch, future.result()
    finally:
        for _raw, future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def run_training(
    args,
    *,
    script_name: str,
    required_keys: tuple[str, ...],
    model: torch.nn.Module,
    step_fn,
    observe_fn=None,
    prepare_batch_fn=None,
    prefetch_batches: int = 0,
) -> None:
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint cannot be used together")
    validate_common_training_args(args)
    if prefetch_batches < 0:
        raise ValueError("prefetch_batches must be non-negative")
    if prepare_batch_fn is None and prefetch_batches:
        raise ValueError("prefetch_batches requires prepare_batch_fn")
    device = resolve_device(args.device)
    seed_everything(args.seed)
    if args.dry_run:
        print(f"{script_name}: device={device}; model parameters={sum(p.numel() for p in model.parameters()):,}")
        return

    dataset, loader, validation_dataset, validation_loader = make_loader(args, required_keys)
    first = dataset[0]
    recorder = RunRecorder(
        args.output_dir,
        script=script_name,
        run_name=args.run_name,
        config=vars(args),
    )
    if args.validate_only:
        validation_count = 0 if validation_dataset is None else len(validation_dataset)
        print(
            f"Validated {len(dataset)} training samples and {validation_count} held-out rows; "
            "first training sample fields/shapes:"
        )
        for key, value in first.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {tuple(value.shape)} {value.dtype}")
            elif isinstance(value, (list, tuple)):
                print(f"  {key}: list[{len(value)}] ({type(value[0]).__name__ if value else 'empty'})")
            else:
                print(f"  {key}: {type(value).__name__}")
        recorder.finish(
            status="validated", samples=len(dataset),
            validation_rows=validation_count,
        )
        return

    model = model.to(device)
    if args.init_checkpoint:
        initialize_matching_weights(
            model, args.init_checkpoint, expected_stage=script_name, device=device,
        )
    optimizer = build_optimizer(
        args.optimizer, model.parameters(), args, lr=args.lr,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(loader) / args.grad_accumulation)
    scheduler = build_lr_scheduler(
        optimizer, args, total_steps=max(1, updates_per_epoch * args.epochs),
    )
    start_epoch, global_step = restore_training(
        args, model, optimizer, device, scheduler, expected_stage=script_name,
    )
    if start_epoch > args.epochs:
        raise ValueError(f"Checkpoint epoch {start_epoch} exceeds requested --epochs {args.epochs}")
    total_optimizer_steps = updates_per_epoch * args.epochs
    progress_started = perf_counter()
    progress_initial_step = global_step
    recorder.write_progress(
        global_step,
        status="starting",
        epoch=f"{start_epoch + 1}/{args.epochs}",
        batch=f"0/{len(loader)}",
        total_steps=total_optimizer_steps,
        learning_rate=f"{float(optimizer.param_groups[0]['lr']):.6g}",
    )
    model.train()
    observe_interval = int(getattr(args, "observe_interval", 0))
    observation_loss_sum = 0.0
    observation_metrics_sum: dict[str, float] = {}
    observation_steps = 0
    pending_step_loss_sum = 0.0
    pending_step_metrics_sum: dict[str, float] = {}
    pending_step_microbatches = 0
    consumed_microbatches = 0
    stop_controller = GracefulStop(
        "Ctrl-C received; finishing the current optimizer step and saving a checkpoint"
    )
    stop_controller.install()

    def checkpoint_sampler(loader):
        batch_sampler = loader.batch_sampler
        if callable(getattr(batch_sampler, "state_dict", None)):
            return batch_sampler, True
        sampler = loader.sampler
        if callable(getattr(sampler, "state_dict", None)):
            return sampler, False
        return None, False

    train_sampler, sampler_positions_are_batches = checkpoint_sampler(loader)
    resume_sampler_state = getattr(args, "_resume_sampler_state", None)
    resume_mid_epoch = bool(getattr(args, "_resume_epoch_in_progress", False))
    for epoch in range(start_epoch, args.epochs):
        if loader.generator is not None:
            loader.generator.manual_seed(args.seed + epoch)
        sampler_generator = getattr(loader.sampler, "generator", None)
        if args.samples_per_epoch is not None and sampler_generator is not None:
            sampler_generator.manual_seed(args.seed + epoch)
        if resume_mid_epoch and epoch == start_epoch and resume_sampler_state is not None:
            if train_sampler is None:
                raise ValueError("Checkpoint has sampler state but current loader is not resumable")
            train_sampler.load_state_dict(resume_sampler_state)
        elif train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_position_start = int(getattr(train_sampler, "position", 0))
        epoch_batch_offset = (
            epoch_position_start if sampler_positions_are_batches
            else math.ceil(epoch_position_start / args.batch_size)
        )
        # Resumable samplers report only their remaining items from __len__.
        # Cache that count before iteration: set_position() below advances the
        # sampler each batch, so len(loader) shrinks while this loop is running.
        epoch_remaining_batches = len(loader)
        epoch_total_batches = epoch_remaining_batches + epoch_batch_offset
        running: dict[str, float] = {}
        seen = 0
        epoch_samples = 0
        optimizer.zero_grad(set_to_none=True)
        performance_started, start_allocated, start_reserved = _start_performance_window(device)
        pending_step_started: float | None = None
        with RichProgress(
            total=epoch_remaining_batches,
            description=f"{script_name} epoch {epoch + 1}/{args.epochs}",
        ) as progress:
            batch_iterator = _iter_prepared_batches(
                loader,
                prepare_batch_fn=prepare_batch_fn,
                prefetch_batches=prefetch_batches,
            )
            for index, (raw_batch, batch) in enumerate(batch_iterator):
                batch_started = perf_counter()
                if pending_step_microbatches == 0:
                    pending_step_started = batch_started
                if prepare_batch_fn is None:
                    batch = (raw_batch if args.data_mode == "hf"
                             else move_batch_to_device(raw_batch, device))
                profile_enabled = bool(getattr(args, "profile_components", False))
                anomaly_enabled = (
                    getattr(args, "anomaly_detection_batch", 0) == index + 1
                )
                anomaly_context = (
                    torch.autograd.detect_anomaly(check_nan=True)
                    if anomaly_enabled
                    else nullcontext()
                )
                with anomaly_context:
                    with autocast_context(device, args.amp):
                        loss, metrics = step_fn(model, batch, args, device)
                        window_start = (index // args.grad_accumulation) * args.grad_accumulation
                        window_size = min(
                            args.grad_accumulation,
                            epoch_remaining_batches - window_start,
                        )
                        if window_size <= 0:
                            raise RuntimeError(
                                "Invalid gradient-accumulation window: "
                                f"index={index}, epoch_batches={epoch_remaining_batches}, "
                                f"window_start={window_start}, window_size={window_size}"
                        )
                        scaled_loss = loss / window_size
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite loss at epoch={epoch + 1}, batch={index + 1}; "
                            f"{_debug_batch_identity(batch)}"
                        )
                    with component_timer(profile_enabled, device) as backward_timer:
                        try:
                            scaled_loss.backward()
                        except RuntimeError as error:
                            if not anomaly_enabled:
                                raise
                            loss_value = float(loss.detach().float().item())
                            raise RuntimeError(
                                f"Backward failed at epoch={epoch + 1}, batch={index + 1}, "
                                f"loss={loss_value:.7g}; {_debug_batch_identity(batch)}"
                            ) from error
                if profile_enabled:
                    metrics["profile_backward_seconds"] = torch.tensor(
                        backward_timer["seconds"], device=device, dtype=torch.float32,
                    )
                should_step = (
                    (index + 1) % args.grad_accumulation == 0
                    or index + 1 == epoch_remaining_batches
                )
                if should_step and getattr(args, "check_finite_updates", False):
                    if not _all_finite_gradients(model):
                        bad_gradients = _nonfinite_gradient_names(model)
                        raise FloatingPointError(
                            f"Non-finite gradient before optimizer update at "
                            f"epoch={epoch + 1}, batch={index + 1}, step={global_step + 1}; "
                            f"parameters={bad_gradients}; {_debug_batch_identity(batch)}"
                        )
                if should_step and getattr(args, "log_adapter_gradients", False):
                    adapter = getattr(model, "adapter", None)
                    gradient_norms = getattr(adapter, "gradient_norms", None)
                    if not callable(gradient_norms):
                        raise ValueError(
                            "--log-adapter-gradients requires a model adapter with gradient_norms()"
                        )
                    gradient_values = gradient_norms()
                    gradient_names = list(gradient_values)
                    gradient_tensor = torch.stack([
                        gradient_values[name].detach().float()
                        for name in gradient_names
                    ])
                    gradient_values_cpu = gradient_tensor.cpu().tolist()
                    condition_types = raw_batch.get("conditioning_types", ())
                    condition_type_counts = {
                        "t2i_samples": sum(str(value).lower() == "t2i" for value in condition_types),
                        "ti2i_samples": sum(str(value).lower() == "ti2i" for value in condition_types),
                    }
                    recorder.record(
                        "adapter_gradient",
                        step=global_step + 1,
                        global_step=global_step + 1,
                        adapter_type=getattr(adapter, "adapter_type", "unknown"),
                        condition_drop_fraction=float(
                            metrics["condition_drop_fraction"].detach().float().item()
                        ) if "condition_drop_fraction" in metrics else None,
                        **condition_type_counts,
                        trainable_adapter_parameters=sum(
                            parameter.numel() for parameter in adapter.parameters()
                            if parameter.requires_grad
                        ),
                        **dict(zip(gradient_names, gradient_values_cpu, strict=True)),
                    )
                    del gradient_tensor, gradient_values
                if should_step:
                    with component_timer(profile_enabled, device) as optimizer_timer:
                        if args.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        scheduler.step(global_step + 1)
                        optimizer.step()
                        if (
                            getattr(args, "check_finite_updates", False)
                            and not _all_finite_parameters(model)
                        ):
                            bad_parameters = _nonfinite_parameter_names(model)
                            raise FloatingPointError(
                                f"Non-finite parameter after optimizer update at "
                                f"epoch={epoch + 1}, batch={index + 1}, step={global_step + 1}; "
                                f"parameters={bad_parameters}; {_debug_batch_identity(batch)}"
                            )
                        optimizer.zero_grad(set_to_none=True)
                    if profile_enabled:
                        metrics["profile_optimizer_step_seconds"] = torch.tensor(
                            optimizer_timer["seconds"], device=device, dtype=torch.float32,
                        )
                    global_step += 1
                if train_sampler is not None:
                    if sampler_positions_are_batches:
                        consumed_position = epoch_position_start + index + 1
                    else:
                        consumed_in_batch = (
                            len(batch["prompts"]) if args.data_mode == "hf"
                            else int(batch["clean_latent"].shape[0])
                        )
                        consumed_position = epoch_position_start + index * args.batch_size + consumed_in_batch
                    train_sampler.set_position(consumed_position)
                seen += 1
                epoch_samples += (
                    len(batch["prompts"]) if args.data_mode == "hf"
                    else int(batch["clean_latent"].shape[0])
                )
                metric_values = {
                    key: float(value.detach().float().mean().item())
                    for key, value in metrics.items()
                    if key != "per_example_loss"
                }
                for key, value in metric_values.items():
                    running[key] = running.get(key, 0.0) + value
                loss_value = float(loss.detach().float().item())
                pending_step_loss_sum += loss_value
                pending_step_microbatches += 1
                for key, value in metric_values.items():
                    pending_step_metrics_sum[key] = (
                        pending_step_metrics_sum.get(key, 0.0) + value
                    )
                if should_step:
                    microbatch_count = max(pending_step_microbatches, 1)
                    observation_loss_sum += pending_step_loss_sum / microbatch_count
                    for key, value in pending_step_metrics_sum.items():
                        observation_metrics_sum[key] = (
                            observation_metrics_sum.get(key, 0.0)
                            + value / microbatch_count
                        )
                    observation_steps += 1
                    update_loss = pending_step_loss_sum / microbatch_count
                    update_metrics = {
                        key: value / microbatch_count
                        for key, value in pending_step_metrics_sum.items()
                    }
                    elapsed_seconds = perf_counter() - progress_started
                    completed_this_run = max(global_step - progress_initial_step, 1)
                    steps_per_second = completed_this_run / max(elapsed_seconds, 1e-12)
                    remaining_steps = max(total_optimizer_steps - global_step, 0)
                    eta_seconds = remaining_steps / max(steps_per_second, 1e-12)
                    optimizer_step_seconds = (
                        perf_counter() - pending_step_started
                        if pending_step_started is not None else perf_counter() - batch_started
                    )
                    progress_metrics = " ".join(
                        f"{key}={value:.5f}"
                        for key, value in update_metrics.items()
                        if key not in {
                            "loss", "mse_weight", "direction_weight", "relational_weight",
                        }
                    )
                    recorder.write_progress(
                        global_step,
                        status="training",
                        epoch=f"{epoch + 1}/{args.epochs}",
                        batch=f"{epoch_batch_offset + index + 1}/{epoch_total_batches}",
                        total_steps=total_optimizer_steps,
                        loss=f"{update_loss:.6f}",
                        loss_components=format_loss_components(update_loss, update_metrics),
                        metrics=progress_metrics or None,
                        learning_rate=f"{float(optimizer.param_groups[0]['lr']):.6g}",
                        step_time=f"{optimizer_step_seconds:.3f}s",
                        elapsed=format_duration(elapsed_seconds),
                        speed=f"{steps_per_second:.3f} optimizer_steps/s",
                        eta=format_duration(eta_seconds),
                    )
                    pending_step_loss_sum = 0.0
                    pending_step_metrics_sum.clear()
                    pending_step_microbatches = 0
                    pending_step_started = None
                    if observe_interval > 0 and global_step % observe_interval == 0:
                        sample_info = None
                        if observe_fn is not None and not getattr(
                            args, "no_observe_samples", False,
                        ):
                            sample_info = observe_fn(
                                model, args, device,
                                recorder.artifacts_dir / "observations",
                                global_step,
                            )
                        interval_metrics = {
                            key: value / max(observation_steps, 1)
                            for key, value in observation_metrics_sum.items()
                        }
                        mean_observation_loss = (
                            observation_loss_sum / max(observation_steps, 1)
                        )
                        recorder.record(
                            "training_observation",
                            epoch=epoch + 1,
                            step=global_step,
                            global_step=global_step,
                            interval_steps=observation_steps,
                            loss=mean_observation_loss,
                            metrics=interval_metrics,
                            sample=sample_info,
                        )
                        print(
                            f"observation step={global_step} "
                            f"loss={mean_observation_loss:.6f} "
                            f"sample={sample_info.get('path') if sample_info else 'disabled'}"
                        )
                        observation_loss_sum = 0.0
                        observation_metrics_sum.clear()
                        observation_steps = 0
                step_seconds = perf_counter() - batch_started
                loss_components = format_loss_components(loss_value, metric_values)
                progress.update(
                    advance=1,
                    postfix=(
                        f"step={index + 1}/{epoch_remaining_batches} "
                        f"time={step_seconds:.3f}s loss={loss_value:.6f}"
                    ),
                )
                status_rows = {
                    "step": (
                        f"epoch={epoch + 1}/{args.epochs} "
                        f"batch={epoch_batch_offset + index + 1}/{epoch_total_batches} "
                        f"global_step={global_step}/{updates_per_epoch * args.epochs} "
                        f"time={step_seconds:.3f}s"
                    ),
                    "loss": (
                        f"total={loss_value:.6f} components={loss_components}"
                    ),
                }
                raw_metrics = {
                    key: value for key, value in metric_values.items()
                    if key not in {
                        "loss", "mse_weight", "direction_weight", "relational_weight",
                    }
                }
                if raw_metrics:
                    status_rows["metrics"] = " ".join(
                        f"{key}={value:.5f}" for key, value in raw_metrics.items()
                    )
                progress.set_status(rows=status_rows)
                # The graph has already been backpropagated. Drop per-batch
                # references before asking Python/CUDA to reclaim unused memory.
                del batch, raw_batch, loss, metrics, scaled_loss
                consumed_microbatches += 1
                maybe_collect_memory(
                    consumed_microbatches,
                    gc_interval=args.gc_interval,
                    empty_cache_interval=args.empty_cache_interval,
                )
                if should_step and stop_controller.requested:
                    checkpoint = recorder.checkpoints_dir / "checkpoint_latest.safetensors"
                    resume_config = {
                        key: value for key, value in vars(args).items()
                        if not key.startswith("_resume_")
                    }
                    save_model_checkpoint(
                        model,
                        checkpoint,
                        metadata={"stage": script_name},
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        config=resume_config,
                        scheduler=scheduler,
                        extra={
                            "epoch_in_progress": True,
                            "sampler_state": (
                                train_sampler.state_dict() if train_sampler is not None else None
                            ),
                        },
                    )
                    recorder.record(
                        "training_interrupted",
                        epoch=epoch + 1,
                        global_step=global_step,
                        sampler_position=(
                            getattr(train_sampler, "position", None)
                            if train_sampler is not None else None
                        ),
                        checkpoint=str(checkpoint),
                    )
                    recorder.write_progress(
                        global_step,
                        status="interrupted_checkpoint_saved",
                        epoch=f"{epoch + 1}/{args.epochs}",
                        batch=f"{epoch_batch_offset + index + 1}/{epoch_total_batches}",
                        checkpoint=str(checkpoint),
                    )
                    print(f"Saved interrupt checkpoint: {checkpoint}")
                    stop_controller.restore()
                    recorder.finish(
                        status="interrupted",
                        global_step=global_step,
                        checkpoint=str(checkpoint),
                    )
                    return
        train_seconds, train_memory = _finish_performance_window(
            device, performance_started, start_allocated, start_reserved,
        )
        train_memory.update(_training_memory_breakdown(
            model,
            optimizer,
            device,
            train_memory.get("peak_allocated_gib", 0.0),
        ))
        means = {key: value / max(seen, 1) for key, value in running.items()}
        means["train_samples"] = float(epoch_samples)
        means["train_seconds"] = train_seconds
        means["train_samples_per_second"] = epoch_samples / max(train_seconds, 1e-12)
        means.update({f"train_{key}": value for key, value in train_memory.items()})
        mean_loss = means.pop("loss", None)
        train_memory_summary = ""
        if "peak_allocated_gib" in train_memory:
            train_memory_summary = (
                f" peak_allocated={train_memory['peak_allocated_gib']:.2f}GiB"
                f" peak_reserved={train_memory['peak_reserved_gib']:.2f}GiB"
            )
        print(
            f"training epoch={epoch + 1}/{args.epochs} samples={epoch_samples}"
            f" seconds={train_seconds:.2f}"
            f" samples_per_second={epoch_samples / max(train_seconds, 1e-12):.2f}"
            f"{train_memory_summary}"
        )
        elapsed_seconds = perf_counter() - progress_started
        completed_this_run = max(global_step - progress_initial_step, 1)
        steps_per_second = completed_this_run / max(elapsed_seconds, 1e-12)
        remaining_steps = max(total_optimizer_steps - global_step, 0)
        recorder.write_progress(
            global_step,
            status="validating" if validation_loader is not None else "checkpointing",
            epoch=f"{epoch + 1}/{args.epochs}",
            batch=f"{epoch_total_batches}/{epoch_total_batches}",
            total_steps=total_optimizer_steps,
            train_loss=f"{float(mean_loss):.6f}" if mean_loss is not None else None,
            learning_rate=f"{float(optimizer.param_groups[0]['lr']):.6g}",
            elapsed=format_duration(elapsed_seconds),
            speed=f"{steps_per_second:.3f} optimizer_steps/s",
            eta=format_duration(remaining_steps / max(steps_per_second, 1e-12)),
        )
        validation_means = (
            evaluate_epoch(
                model, validation_loader, step_fn, args, device, epoch=epoch,
            )
            if validation_loader is not None else {}
        )
        validation_loss = validation_means.pop("loss", None)
        if validation_loss is not None:
            validation_metrics = " ".join(
                f"{key}={value:.6f}"
                for key, value in sorted(validation_means.items())
                if not key.startswith((
                    "tbin_", "snrbin_", "prediction_spectrum_", "teacher_spectrum_",
                ))
            )
            print(
                f"validation epoch={epoch + 1}/{args.epochs} "
                f"loss={validation_loss:.6f} {validation_metrics}".rstrip()
            )
        means.update({f"val_{key}": value for key, value in validation_means.items()})
        elapsed_seconds = perf_counter() - progress_started
        completed_this_run = max(global_step - progress_initial_step, 1)
        steps_per_second = completed_this_run / max(elapsed_seconds, 1e-12)
        remaining_steps = max(total_optimizer_steps - global_step, 0)
        recorder.write_progress(
            global_step,
            status="checkpointing",
            epoch=f"{epoch + 1}/{args.epochs}",
            batch=f"{epoch_total_batches}/{epoch_total_batches}",
            total_steps=total_optimizer_steps,
            train_loss=f"{float(mean_loss):.6f}" if mean_loss is not None else None,
            validation_loss=(
                f"{float(validation_loss):.6f}" if validation_loss is not None else None
            ),
            learning_rate=f"{float(optimizer.param_groups[0]['lr']):.6g}",
            elapsed=format_duration(elapsed_seconds),
            speed=f"{steps_per_second:.3f} optimizer_steps/s",
            eta=format_duration(remaining_steps / max(steps_per_second, 1e-12)),
        )
        recorder.record_training_step(
            epoch=epoch + 1,
            global_step=global_step,
            train_loss=mean_loss,
            eval_loss=validation_loss,
            effective_lr=float(optimizer.param_groups[0]["lr"]),
            metrics=means,
        )
        checkpoint = recorder.checkpoints_dir / "checkpoint_latest.safetensors"
        save_model_checkpoint(
            model,
            checkpoint,
            metadata={"stage": script_name},
            optimizer=optimizer,
            epoch=epoch + 1,
            global_step=global_step,
            config={
                key: value for key, value in vars(args).items()
                if not key.startswith("_resume_")
            },
            scheduler=scheduler,
        )
        print(f"Saved checkpoint: {checkpoint}")
        if stop_controller.requested:
            stop_controller.restore()
            recorder.finish(
                status="interrupted",
                global_step=global_step,
                checkpoint=str(checkpoint),
            )
            return
    stop_controller.restore()
    recorder.write_progress(
        global_step,
        status="completed",
        epoch=f"{args.epochs}/{args.epochs}",
        batch=f"{epoch_total_batches}/{epoch_total_batches}",
        total_steps=total_optimizer_steps,
        train_loss=f"{float(mean_loss):.6f}" if mean_loss is not None else None,
        validation_loss=(
            f"{float(validation_loss):.6f}" if validation_loss is not None else None
        ),
        learning_rate=f"{float(optimizer.param_groups[0]['lr']):.6g}",
        elapsed=format_duration(perf_counter() - progress_started),
    )
    recorder.finish(status="completed", global_step=global_step)
