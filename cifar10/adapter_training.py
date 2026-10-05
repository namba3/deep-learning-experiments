"""CIFAR-10 adapter configuration and optimizer parameter selection."""

from __future__ import annotations

import argparse

from core.low_rank import (
    canonicalize_adapter_type,
    inject_adapter,
    mark_only_adapter_trainable,
    resolve_adapter_type,
)
from core.utils import build_parameter_groups


ADAPTER_CHOICES = ("none", "lora", "dora", "loha", "glu_lora", "rglu_lora")
DEFAULT_ADAPTER_TARGETS = (
    r"\.attention\.(qkv|output)$",
    r"^pooling\.attn\.(q_proj|kv_proj|out_proj)$",
)


def add_adapter_arguments(parser: argparse.ArgumentParser) -> None:
    """Add adapter-only options to an experiment parser."""
    parser.add_argument(
        "--lora-rank", type=int, default=0,
        help="Enable a low-rank adapter with this rank.",
    )
    parser.add_argument(
        "--adapter", type=canonicalize_adapter_type,
        choices=ADAPTER_CHOICES, default="none",
        help="Low-rank adapter type. A positive --lora-rank is required.",
    )
    parser.add_argument(
        "--lora-alpha", type=float, default=None,
        help="Adapter scaling alpha. Defaults to the rank.",
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.0,
        help="Dropout applied before the adapter update. Default: 0.",
    )
    parser.add_argument(
        "--lora-target", action="append", default=None,
        help="Regex for target Linear modules; repeat to add patterns.",
    )
    parser.add_argument(
        "--base-init", choices=("checkpoint", "random"),
        default="checkpoint",
        help=(
            "Base initialization for adapter training. checkpoint uses "
            "--init-checkpoint/--resume; random uses init_weights."
        ),
    )
    parser.add_argument(
        "--adapter-init", choices=("identity", "lora_warm"),
        default="identity",
        help=(
            "GLU-LoRA initialization. identity preserves the "
            "base output; lora_warm initializes its Value branch as a "
            "nonzero LoRA-like update."
        ),
    )


def resolve_adapter_config(args, *, resume_path, init_path) -> str:
    """Validate adapter CLI/checkpoint choices and return the adapter type."""
    if args.lora_rank < 0:
        raise ValueError("--lora-rank must be >= 0")
    if args.lora_alpha is not None and args.lora_alpha <= 0:
        raise ValueError("--lora-alpha must be positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora-dropout must be in [0, 1)")
    adapter_type = resolve_adapter_type(args.adapter, args.lora_rank)
    if adapter_type == "none":
        raise ValueError("train_adapter requires --adapter or a positive --lora-rank")
    if args.adapter_init != "identity" and adapter_type not in {"glu_lora", "rglu_lora"}:
        raise ValueError(
            "--adapter-init lora_warm is only supported with "
            "--adapter glu_lora or --adapter rglu_lora"
        )
    if args.base_init == "random":
        if init_path:
            raise ValueError(
                "--base-init random cannot be combined with --init-checkpoint"
            )
    elif not resume_path and not init_path:
        raise ValueError(
            "train_adapter with --base-init checkpoint requires "
            "--init-checkpoint or --resume"
        )
    return adapter_type


def enable_adapter(model, args) -> tuple[list[str], int]:
    """Inject the configured adapter and return targets and trainable count."""
    targets = args.lora_target or DEFAULT_ADAPTER_TARGETS
    matched = inject_adapter(
        model,
        args.adapter,
        targets,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        init_mode=args.adapter_init,
    )
    trainable = mark_only_adapter_trainable(model)
    return matched, trainable


def build_adapter_parameter_groups(model, weight_decay):
    """Build optimizer groups after adapter injection/freeze has completed."""
    groups = build_parameter_groups(
        model,
        target_param_regexes=[r"linear", r"conv2d", r"super_weight"],
        weight_decay=weight_decay,
    )
    if not groups:
        raise ValueError("adapter training found no trainable parameters")
    return groups
