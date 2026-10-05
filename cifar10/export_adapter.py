"""Export a CIFAR-10 adapter checkpoint as a merged plain-model checkpoint."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

from cifar10.adapter_training import DEFAULT_ADAPTER_TARGETS
from cifar10.train import (
    CIFAR10_CONFIG_METADATA_KEY,
    CIFAR10ViT,
    EMBED_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    PATCH_SIZE,
    init_weights,
)
from core.low_rank import inject_adapter, materialize_adapter
from core.utils import compact_state_dict, convert_linear_to_bf16
from runtime.config import checkpoint_config_metadata, read_checkpoint_config
from runtime.device import add_device_argument, resolve_device


def _build_model(config: dict[str, object], device: torch.device) -> torch.nn.Module:
    bf16 = bool(config.get("bf16", False))
    model = CIFAR10ViT(
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        compute_dtype=torch.bfloat16 if bf16 else None,
        attention_type=str(config.get("attention_type", "full")),
        window_size=int(config.get("window_size", 4)),
        mhla_block_size=(
            None
            if config.get("mhla_block_size") is None
            else int(config["mhla_block_size"])
        ),
        mhla_backend=str(config.get("mhla_backend", "auto")),
    ).to(device)
    model.apply(init_weights)
    if bf16:
        convert_linear_to_bf16(model.encoder)
    return model


def export_checkpoint(
    checkpoint: str,
    output: str,
    *,
    device: str = "cpu",
) -> None:
    """Load, merge, and save one adapter checkpoint without resume state."""
    config = read_checkpoint_config(checkpoint, CIFAR10_CONFIG_METADATA_KEY)
    if not config:
        raise ValueError(
            f"checkpoint has no {CIFAR10_CONFIG_METADATA_KEY} metadata: {checkpoint}"
        )
    adapter = str(config.get("adapter", "none"))
    rank = int(config.get("lora_rank", 0))
    if adapter == "none" or rank <= 0:
        raise ValueError("checkpoint does not contain a configured adapter")

    model = _build_model(config, resolve_device(device, default=torch.device("cpu")))
    targets = config.get("lora_target") or DEFAULT_ADAPTER_TARGETS
    if not isinstance(targets, (list, tuple)):
        raise ValueError("checkpoint lora_target must be a list of regex patterns")
    inject_adapter(
        model,
        adapter,
        targets,
        rank=rank,
        alpha=(
            None if config.get("lora_alpha") is None
            else float(config["lora_alpha"])
        ),
        dropout=float(config.get("lora_dropout", 0.0)),
        init_mode=str(config.get("adapter_init", "identity")),
    )
    state_dict = load_file(checkpoint, device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(
            f"checkpoint contains unexpected model keys: {unexpected[:5]}"
        )
    if missing:
        # Compact checkpoints may omit shared aliases, but adapter parameters
        # must always be present before materialization.
        adapter_missing = [
            key for key in missing
            if "lora_" in key or key.endswith("magnitude")
        ]
        if adapter_missing:
            raise ValueError(
                f"checkpoint is missing adapter parameters: {adapter_missing[:5]}"
            )
    materialize_adapter(model)

    output_config = dict(config)
    output_config.update(
        adapter="none",
        lora_rank=0,
        lora_alpha=None,
        lora_dropout=0.0,
        lora_target=None,
        adapter_init="identity",
        merged_adapter=True,
        merged_from=checkpoint,
    )
    save_file(
        compact_state_dict(model, cast_bf16=bool(config.get("bf16", False))),
        output,
        metadata=checkpoint_config_metadata(
            SimpleNamespace(**output_config), CIFAR10_CONFIG_METADATA_KEY,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    add_device_argument(parser)
    args = parser.parse_args(argv)
    export_checkpoint(args.checkpoint, args.output, device=args.device)
    print(f"Saved merged checkpoint to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
