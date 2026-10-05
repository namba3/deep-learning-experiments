"""Summarize a matched activation-checkpointing comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_run(run_dir: Path, expected_name: str) -> dict[str, Any] | None:
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.jsonl"
    if not config_path.is_file() or not metrics_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args = config.get("args", {})
    if config.get("run_name") != expected_name and args.get("run_name") != expected_name:
        return None

    events = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    steps = [event for event in events if event.get("event") == "step"]
    terminal = next(
        (event for event in reversed(events) if event.get("event") in {"run_finished", "run_failed"}),
        {},
    )
    final_step = max(steps, key=lambda event: int(event.get("epoch", 0)), default={})
    return {
        "run_id": config.get("run_id", run_dir.name),
        "status": terminal.get("status", "incomplete"),
        "error": terminal.get("error_type"),
        "args": args,
        "step": final_step,
    }


def _latest_run(output_dir: Path, run_name: str) -> dict[str, Any] | None:
    runs_dir = output_dir / "runs"
    if not runs_dir.is_dir():
        return None
    candidates = []
    for config_path in runs_dir.glob("*/config.json"):
        try:
            run = _read_run(config_path.parent, run_name)
        except (OSError, json.JSONDecodeError) as error:
            print(f"Skipping unreadable run metadata {config_path}: {error}")
            continue
        if run is not None:
            candidates.append((config_path.stat().st_mtime_ns, run))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _metric(run: dict[str, Any] | None, key: str) -> str:
    if run is None:
        return "n/a"
    value = run["step"].get(key)
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def summarize(output_dir: Path, base_run_name: str) -> str:
    arms = {}
    for mode, enabled in (("disabled", False), ("enabled", True)):
        expected_name = f"{base_run_name}-{mode}"
        run = _latest_run(output_dir, expected_name)
        if run is not None and bool(run["args"].get("gradient_checkpointing", False)) != enabled:
            raise ValueError(
                f"Run {run['run_id']} has an unexpected gradient_checkpointing setting"
            )
        arms[mode] = run

    lines = [
        "# VFP-DiT simple activation-checkpointing comparison",
        "",
        f"Base run: `{base_run_name}`",
        "",
        "| Arm | Status | Epoch | Train loss | Eval loss | T2I eval (n) | TI2I eval (n) | Peak allocated (GiB) | Peak reserved (GiB) | Samples/s | Run ID |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for mode in ("disabled", "enabled"):
        run = arms[mode]
        if run is None:
            lines.append(
                f"| {mode} | missing | n/a | n/a | n/a | n/a | n/a "
                "| n/a | n/a | n/a | n/a |"
            )
            continue
        step = run["step"]
        status = run["status"]
        if run["error"]:
            status = f"{status}: {run['error']}"
        lines.append(
            f"| {mode} | {status} | {step.get('epoch', 'n/a')} "
            f"| {_metric(run, 'loss')} | {_metric(run, 'eval_loss')} "
            f"| {_metric(run, 'val_loss_t2i')} ({_metric(run, 'val_loss_t2i_samples')}) "
            f"| {_metric(run, 'val_loss_ti2i')} ({_metric(run, 'val_loss_ti2i_samples')}) "
            f"| {_metric(run, 'train_peak_allocated_gib')} "
            f"| {_metric(run, 'train_peak_reserved_gib')} "
            f"| {_metric(run, 'train_samples_per_second')} | `{run['run_id']}` |"
        )
    lines.extend([
        "",
        "Peak memory and throughput are epoch-level training metrics. Missing metrics mean the run ended before an epoch summary was recorded, commonly because it failed during training.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("vfp_dit/output"))
    parser.add_argument("--base-run-name", required=True)
    args = parser.parse_args(argv)
    report = summarize(args.output_dir, args.base_run_name)
    safe_name = "".join(char if char.isalnum() or char in "-_." else "_" for char in args.base_run_name)
    output_path = args.output_dir / f"checkpointing_comparison_{safe_name}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
