"""Export a text LM adapter checkpoint as a merged plain-model checkpoint."""

from __future__ import annotations

import argparse
import json

import torch
from safetensors.torch import load_file, save_file

from text_lm.adapter_training import DEFAULT_ADAPTER_TARGETS
from text_lm.train import (
    TEXT_LM_CONFIG_METADATA_KEY,
    TinyTextLM,
    read_text_lm_checkpoint_config,
)
from core.low_rank import inject_adapter, materialize_adapter
from core.utils import compact_state_dict, convert_linear_to_bf16
from runtime.device import add_device_argument, resolve_device


def _value(config, name, default):
    value = config.get(name)
    return default if value is None else value


def _build_model(config, vocab_size, device):
    model = TinyTextLM(
        vocab_size=vocab_size,
        max_seq_len=int(_value(config, "max_seq_len", 512)),
        embed_dim=int(_value(config, "embed_dim", 2048)),
        num_layers=int(_value(config, "num_layers", 16)),
        num_heads=int(_value(config, "num_heads", 32)),
        kv_heads=int(_value(config, "kv_heads", 8)),
        condition_dim=int(_value(config, "condition_dim", 64)),
        transform_rank=int(_value(config, "transform_rank", 10)),
        architecture=str(_value(config, "architecture", "naive")),
        looped_blocks=_value(config, "looped_blocks", None),
        looped_prefix_layers=int(_value(config, "looped_prefix_layers", 4)),
        looped_repeats=int(_value(config, "looped_repeats", 4)),
        looped_suffix_layers=int(_value(config, "looped_suffix_layers", 4)),
        mhla_looped_prefix_cycles=int(
            _value(config, "mhla_looped_prefix_cycles", 1)
        ),
        mhla_looped_repeats=int(_value(config, "mhla_looped_repeats", 2)),
        mhla_looped_suffix_cycles=int(
            _value(config, "mhla_looped_suffix_cycles", 1)
        ),
        compute_dtype=torch.bfloat16
        if bool(_value(config, "bf16", False)) else None,
    ).to(device)
    if bool(_value(config, "bf16", False)):
        depth_embedding = getattr(model.decoder, "depth_embedding", None)
        convert_linear_to_bf16(
            model.decoder,
            skip_modules=()
            if depth_embedding is None else (depth_embedding,),
        )
    return model


def export_checkpoint(checkpoint: str, output: str, *, device: str = "cpu") -> None:
    """Load, merge, and save one text LM adapter checkpoint."""
    config = read_text_lm_checkpoint_config(checkpoint)
    if not config:
        raise ValueError(
            f"checkpoint has no {TEXT_LM_CONFIG_METADATA_KEY} metadata: {checkpoint}"
        )
    adapter = str(config.get("adapter", "none"))
    rank = int(config.get("lora_rank", 0))
    if adapter == "none" or rank <= 0:
        raise ValueError("checkpoint does not contain a configured adapter")

    state_dict = load_file(checkpoint, device="cpu")
    embedding = state_dict.get("token_embedding.weight")
    if embedding is None:
        raise ValueError("checkpoint is missing token_embedding.weight")
    model = _build_model(
        config,
        embedding.shape[0],
        resolve_device(device, default=torch.device("cpu")),
    )
    targets = config.get("lora_target") or DEFAULT_ADAPTER_TARGETS
    if not isinstance(targets, (list, tuple)):
        raise ValueError("checkpoint lora_target must be a list of regex patterns")
    inject_adapter(
        model,
        adapter,
        targets,
        rank=rank,
        alpha=None if config.get("lora_alpha") is None
        else float(config["lora_alpha"]),
        dropout=float(config.get("lora_dropout", 0.0)),
        init_mode=str(config.get("adapter_init", "identity")),
    )
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ValueError(
            "adapter checkpoint does not match the reconstructed model: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
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
        metadata={
            TEXT_LM_CONFIG_METADATA_KEY: json.dumps(
                {"schema_version": 1, "args": output_config},
                sort_keys=True,
            )
        },
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
