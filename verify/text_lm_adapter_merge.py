"""Validate merge export and strict loading for text-lm adapter runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import torch
from safetensors.torch import load_file

from core.low_rank import canonicalize_adapter_type, inject_adapter
from runtime.device import resolve_device
from text_lm.adapter_training import DEFAULT_ADAPTER_TARGETS
from text_lm.export_adapter import _build_model, export_checkpoint
from text_lm.train import read_text_lm_checkpoint_config


def _load_wrapped_model(checkpoint: Path, config: dict, device: torch.device):
    state_dict = load_file(str(checkpoint), device="cpu")
    embedding = state_dict.get("token_embedding.weight")
    if embedding is None:
        raise ValueError(
            f"checkpoint is missing token_embedding.weight: {checkpoint}"
        )
    model = _build_model(config, embedding.shape[0], device)
    targets = config.get("lora_target") or DEFAULT_ADAPTER_TARGETS
    inject_adapter(
        model,
        str(config.get("adapter", "none")),
        targets,
        rank=int(config.get("lora_rank", 0)),
        alpha=(
            None if config.get("lora_alpha") is None
            else float(config["lora_alpha"])
        ),
        dropout=float(config.get("lora_dropout", 0.0)),
        init_mode=str(config.get("adapter_init", "identity")),
    )
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ValueError(
            "adapter checkpoint does not match reconstructed model: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    return model.eval(), embedding.shape[0]


def validate_checkpoint(
    checkpoint: Path, *, device: torch.device, probe_seq_len: int,
) -> dict:
    config = read_text_lm_checkpoint_config(str(checkpoint))
    if not config:
        raise ValueError(f"checkpoint has no text_lm metadata: {checkpoint}")
    adapter = canonicalize_adapter_type(str(config.get("adapter", "none")))
    if adapter == "none" or int(config.get("lora_rank", 0)) <= 0:
        raise ValueError(f"checkpoint does not contain an adapter: {checkpoint}")

    wrapped, vocab_size = _load_wrapped_model(checkpoint, config, device)
    sequence_length = min(
        probe_seq_len, int(config.get("max_seq_len", probe_seq_len)),
    )
    if sequence_length <= 1:
        raise ValueError("probe sequence length must be greater than one")
    input_ids = (
        torch.arange(sequence_length, device=device, dtype=torch.long)
        % vocab_size
    ).unsqueeze(0)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        wrapped_output = wrapped(input_ids)
    with tempfile.TemporaryDirectory(prefix="text-lm-merge-") as temp_dir:
        merged_path = Path(temp_dir) / "merged.safetensors"
        # Export on CPU so validation does not keep the wrapped model and a
        # second full model resident on the accelerator at the same time.
        export_checkpoint(str(checkpoint), str(merged_path), device="cpu")
        merged_state = load_file(str(merged_path), device="cpu")
        if any("lora_" in key for key in merged_state):
            raise ValueError(
                f"merged checkpoint still contains adapter keys: {checkpoint}"
            )
        merged_model = _build_model(config, vocab_size, device).eval()
        merged_model.load_state_dict(merged_state, strict=True)
        with torch.inference_mode():
            merged_output = merged_model(input_ids)

    max_abs_error = (wrapped_output - merged_output).abs().max().item()
    # Materializing a low-rank delta changes matmul evaluation order.  Use
    # the same atol/rtol-style comparison as the text-lm integration test.
    tolerance = 5e-3 if bool(config.get("bf16", False)) else 3e-4
    relative_tolerance = 1e-3 if bool(config.get("bf16", False)) else 1e-5
    merge_equivalent = torch.allclose(
        wrapped_output,
        merged_output,
        atol=tolerance,
        rtol=relative_tolerance,
    )
    return {
        "checkpoint": str(checkpoint),
        "adapter": adapter,
        "rank": int(config.get("lora_rank", 0)),
        "alpha": config.get("lora_alpha"),
        "max_seq_len": int(config.get("max_seq_len", 0)),
        "probe_seq_len": sequence_length,
        "device": str(device),
        "max_abs_error": max_abs_error,
        "tolerance": tolerance,
        "relative_tolerance": relative_tolerance,
        "merged_strict_load": True,
        "merged_adapter_keys": False,
        "merge_equivalent": merge_equivalent,
        "cuda_peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
            if device.type == "cuda" else None
        ),
    }


def collect_checkpoints(input_dir: Path) -> list[Path]:
    checkpoints = []
    for checkpoint in sorted(input_dir.glob("runs/*/artifacts/model.safetensors")):
        config = read_text_lm_checkpoint_config(str(checkpoint))
        if config and config.get("adapter", "none") != "none":
            checkpoints.append(checkpoint)
    return checkpoints


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--probe-seq-len", type=int, default=16)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda")
    )
    args = parser.parse_args(argv)
    if args.probe_seq_len <= 1:
        raise ValueError("--probe-seq-len must be greater than one")

    device = resolve_device(args.device)
    checkpoints = collect_checkpoints(args.input_dir)
    if not checkpoints:
        raise FileNotFoundError(
            f"no adapter checkpoints found below {args.input_dir}/runs"
        )
    results = []
    for checkpoint in checkpoints:
        results.append(
            validate_checkpoint(
                checkpoint, device=device, probe_seq_len=args.probe_seq_len,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "input_dir": str(args.input_dir),
        "device": str(device),
        "probe_seq_len": args.probe_seq_len,
        "checkpoint_count": len(results),
        "passed": all(result["merge_equivalent"] for result in results),
        "results": results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
