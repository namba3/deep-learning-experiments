"""Text LM adapter configuration and trainable-parameter selection."""

from __future__ import annotations

import argparse

from core.low_rank import (
    canonicalize_adapter_type,
    inject_adapter,
    mark_only_adapter_trainable,
    resolve_adapter_type,
)


ADAPTER_CHOICES = ("none", "lora", "dora", "loha", "glu_lora", "rglu_lora")
DEFAULT_ADAPTER_TARGETS = (
    r"\.attn\.(q_proj|k_proj|v_proj|kv_proj|out_proj)$",
    r"\.ffn\.2$",
)


def add_adapter_arguments(parser: argparse.ArgumentParser) -> None:
    """Add options that are meaningful only for adapter training."""
    parser.add_argument(
        "--lora-base-checkpoint", default=None,
        help="Load a complete text LM and train only adapter parameters.",
    )
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
        "--adapter-init", choices=("identity", "lora_warm"),
        default="identity",
        help=(
            "GLU-LoRA initialization. identity preserves the "
            "base output; lora_warm initializes its Value branch nonzero."
        ),
    )


def resolve_adapter_config(args, *, resume_path, base_path) -> str:
    """Validate adapter-only CLI/checkpoint choices and return its type."""
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint cannot be used together")
    if base_path and (args.resume or args.init_checkpoint):
        raise ValueError(
            "--lora-base-checkpoint cannot be combined with --resume or "
            "--init-checkpoint"
        )
    if args.lora_rank < 0:
        raise ValueError("--lora-rank must be >= 0")
    if args.lora_rank > 0 and args.init_checkpoint:
        raise ValueError("--init-checkpoint cannot be used with --lora-rank")
    if args.lora_rank == 0 and base_path:
        raise ValueError("--lora-base-checkpoint requires --lora-rank > 0")
    if args.lora_alpha is not None and args.lora_alpha <= 0:
        raise ValueError("--lora-alpha must be positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora-dropout must be in [0, 1)")
    adapter_type = resolve_adapter_type(args.adapter, args.lora_rank)
    if adapter_type == "none":
        raise ValueError(
            "text_lm.train_adapter requires --adapter or a positive --lora-rank"
        )
    if args.adapter_init != "identity" and adapter_type not in {"glu_lora", "rglu_lora"}:
        raise ValueError(
            "--adapter-init lora_warm is only supported with "
            "--adapter glu_lora or --adapter rglu_lora"
        )
    if not (base_path or resume_path):
        raise ValueError(
            "adapter training requires --lora-base-checkpoint or "
            "an adapter --resume checkpoint"
        )
    if args.architecture != "naive":
        raise ValueError(
            "Text LM adapter training currently requires --architecture naive; "
            "shared and looped architectures need a separate parameterization"
        )
    return adapter_type


def enable_adapter(model, args) -> tuple[list[str], int]:
    """Inject the configured adapter and freeze the base model."""
    targets = args.lora_target or DEFAULT_ADAPTER_TARGETS
    args.lora_target = list(targets)
    matched = inject_adapter(
        model,
        args.adapter,
        targets,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        init_mode=args.adapter_init,
    )
    trainable_count = mark_only_adapter_trainable(model)
    if trainable_count <= 0:
        raise ValueError("adapter training found no trainable parameters")
    return matched, trainable_count


def optimizer_parameters(model):
    """Return only adapter parameters for the optimizer."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("adapter training found no trainable parameters")
    return parameters
