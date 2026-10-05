"""Export an ImageAE adapter checkpoint as a merged plain-model checkpoint."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from core.low_rank import inject_adapter, materialize_adapter
from core.utils import compact_state_dict
from image_ae.adapter_training import DEFAULT_LORA_TARGETS
from image_ae.train import (
    NETWORK_CONFIG_METADATA_KEY,
    TRAINING_CONFIG_METADATA_KEY,
    checkpoint_epoch,
    save_model_checkpoint,
)
from runtime.config import read_checkpoint_config
from runtime.device import add_device_argument, resolve_device


def _config_value(config: dict[str, object], name: str, default):
    value = config.get(name)
    return default if value is None else value


def _build_model(config: dict[str, object], device: torch.device) -> torch.nn.Module:
    from image_ae.train import ImageAE

    model = ImageAE(
        latent_channels=int(_config_value(config, "latent_channels", 16)),
        encoder_type=str(_config_value(config, "encoder", "window_transformer")),
        decoder_type=str(_config_value(config, "decoder", "window_transformer")),
        bottleneck_channels=int(_config_value(config, "bottleneck_channels", 256)),
        encoder_blocks=int(_config_value(config, "encoder_blocks", 1)),
        decoder_blocks=int(_config_value(config, "decoder_blocks", 1)),
        hidden_channels=_config_value(config, "hidden_channels", None),
        downsample_stages=int(_config_value(config, "downsample_stages", 3)),
        encoder_layers=int(_config_value(config, "encoder_layers", 2)),
        encoder_window_size=int(_config_value(config, "encoder_window_size", 8)),
        decoder_layers=int(_config_value(config, "decoder_layers", 2)),
        vae=bool(_config_value(config, "vae", False)),
    )
    return model.to(device)


def export_checkpoint(
    checkpoint: str,
    output: str,
    *,
    device: str = "cpu",
) -> None:
    """Load, merge, and save one ImageAE adapter checkpoint."""
    network_config = read_checkpoint_config(checkpoint, NETWORK_CONFIG_METADATA_KEY)
    training_config = read_checkpoint_config(checkpoint, TRAINING_CONFIG_METADATA_KEY)
    if not network_config:
        raise ValueError(
            f"checkpoint has no {NETWORK_CONFIG_METADATA_KEY} metadata: {checkpoint}"
        )
    if not training_config:
        raise ValueError(
            f"checkpoint has no {TRAINING_CONFIG_METADATA_KEY} metadata: {checkpoint}"
        )
    adapter = str(training_config.get("adapter", "none"))
    rank = int(training_config.get("lora_rank", 0))
    if adapter == "none" or rank <= 0:
        raise ValueError("checkpoint does not contain a configured adapter")

    model = _build_model(
        network_config,
        resolve_device(device, default=torch.device("cpu")),
    )
    targets = training_config.get("lora_target") or DEFAULT_LORA_TARGETS
    if not isinstance(targets, (list, tuple)):
        raise ValueError("checkpoint lora_target must be a list of regex patterns")
    inject_adapter(
        model,
        adapter,
        targets,
        rank=rank,
        alpha=(
            None
            if training_config.get("lora_alpha") is None
            else float(training_config["lora_alpha"])
        ),
        dropout=float(training_config.get("lora_dropout", 0.0)),
        init_mode=str(training_config.get("adapter_init", "identity")),
    )
    state_dict = load_file(checkpoint, device="cpu")
    model.load_state_dict(state_dict, strict=True)
    materialize_adapter(model)

    output_config = dict(network_config)
    output_config.update(adapter="none", lora_rank=0)
    output_training = dict(training_config)
    output_training.update(
        adapter="none",
        lora_rank=0,
        lora_alpha=None,
        lora_dropout=0.0,
        lora_target=None,
        adapter_init="identity",
        merged_adapter=True,
        merged_from=checkpoint,
    )
    output_args = SimpleNamespace(**{**output_config, **output_training})
    save_model_checkpoint(
        compact_state_dict(model),
        output,
        output_args,
        epoch=checkpoint_epoch(checkpoint),
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
