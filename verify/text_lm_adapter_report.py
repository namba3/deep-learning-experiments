"""Summarize text-lm adapter runs by context length and adapter type."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from core.low_rank import canonicalize_adapter_type


VALUE_KEYS = (
    "train_loss",
    "eval_loss",
    "train_ppl",
    "eval_ppl",
    "step_time_sec",
    "cuda_peak_allocated_mib",
    "cuda_peak_reserved_mib",
    "optimizer_state_bytes",
)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _optimizer_state_bytes(path: Path) -> int | None:
    if not path.is_file():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    optimizer = state.get("optimizer") if isinstance(state, dict) else None
    if not isinstance(optimizer, dict):
        return None
    total = 0
    for values in optimizer.get("state", {}).values():
        if not isinstance(values, dict):
            continue
        for value in values.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


def _run_summary(metrics_path: Path) -> dict | None:
    events = _read_jsonl(metrics_path)
    started = next(
        (event for event in events if event.get("event") == "run_started"),
        {},
    )
    preflight = next(
        (event for event in events if event.get("event") == "preflight"),
        {},
    )
    epochs = [event for event in events if event.get("event") == "epoch"]
    if not epochs:
        return None
    config_path = started.get("config_path")
    config = {}
    if config_path:
        config_file = metrics_path.parent / Path(config_path).name
        if config_file.is_file():
            config = json.loads(config_file.read_text(encoding="utf-8"))
    config = config.get("args", config)
    last = epochs[-1]
    max_seq_len = config.get("max_seq_len")
    batch_size = config.get("batch_size")
    return {
        "run_id": metrics_path.parent.name,
        "dataset": config.get("dataset_name"),
        "adapter": canonicalize_adapter_type(str(config.get("adapter"))),
        "rank": config.get("lora_rank"),
        "alpha": config.get("lora_alpha"),
        "learning_rate": config.get("lr"),
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "tokens_per_step": (
            int(max_seq_len) * int(batch_size)
            if max_seq_len is not None and batch_size is not None else None
        ),
        "seed": config.get("seed"),
        "dtype": preflight.get("dtype"),
        "device": preflight.get("device"),
        "status": next(
            (
                event.get("status")
                for event in reversed(events)
                if event.get("event") == "run_finished"
            ),
            "unknown",
        ),
        "epoch": last.get("epoch"),
        "parameters": last.get("model_parameters"),
        "trainable_parameters": last.get("trainable_parameters"),
        "train_tokens": last.get("train_tokens"),
        "eval_tokens": last.get("eval_tokens"),
        "train_loss": last.get("train_loss"),
        "eval_loss": last.get("eval_loss"),
        "train_ppl": last.get("train_ppl"),
        "eval_ppl": last.get("eval_ppl"),
        "step_time_sec": (
            1.0 / last["steps_per_second"]
            if last.get("steps_per_second", 0) else None
        ),
        "cuda_peak_allocated_mib": last.get("cuda_peak_allocated_mib"),
        "cuda_peak_reserved_mib": last.get("cuda_peak_reserved_mib"),
        "optimizer_state_bytes": _optimizer_state_bytes(
            metrics_path.parent / "artifacts" / "model.resume.pt"
        ),
    }


def collect_runs(input_dir: Path) -> list[dict]:
    runs = []
    for metrics_path in sorted(input_dir.glob("runs/*/metrics.jsonl")):
        summary = _run_summary(metrics_path)
        if summary is not None and summary.get("adapter") not in (None, "none"):
            runs.append(summary)
    return runs


def _group_key(run: dict) -> tuple:
    return (
        run["adapter"], run["rank"], run["alpha"], run["learning_rate"],
        run["max_seq_len"], run["batch_size"], run["tokens_per_step"],
    )


def summarize_runs(runs: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for run in runs:
        grouped.setdefault(_group_key(run), []).append(run)
    summaries = []
    for (
        adapter, rank, alpha, learning_rate, max_seq_len, batch_size,
        tokens_per_step,
    ), entries in sorted(
        grouped.items(),
        key=lambda item: tuple(
            "" if value is None else str(value) for value in item[0]
        ),
    ):
        summary = {
            "adapter": adapter,
            "rank": rank,
            "alpha": alpha,
            "learning_rate": learning_rate,
            "max_seq_len": max_seq_len,
            "batch_size": batch_size,
            "tokens_per_step": tokens_per_step,
            "run_count": len(entries),
            "seeds": sorted({entry["seed"] for entry in entries}),
        }
        for key in VALUE_KEYS:
            values = [entry[key] for entry in entries if entry[key] is not None]
            summary[f"{key}_mean"] = statistics.mean(values) if values else None
            summary[f"{key}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
                if values else None
            )
        summaries.append(summary)
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    runs = collect_runs(args.input_dir)
    report = {
        "input_dir": str(args.input_dir),
        "runs": runs,
        "summary": summarize_runs(runs),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
