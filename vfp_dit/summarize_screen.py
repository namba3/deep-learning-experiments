"""Create a compact Markdown report for one VFP-DiT simple quality screen."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read complete JSONL records, skipping a concurrently written last line."""
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def read_progress(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            result[key.strip()] = value.strip()
    return result


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    return str(value)


def metric_cell(event: dict[str, Any], key: str, digits: int = 4) -> str:
    return fmt(event.get(key), digits)


def image_link(report_path: Path, raw_path: Any, run_dir: Path) -> str:
    if not raw_path:
        return "—"
    path = Path(str(raw_path))
    # Archived metrics keep their original output path. Resolve an artifacts
    # suffix against the supplied run directory so reports remain portable.
    if "artifacts" in path.parts:
        artifact_index = path.parts.index("artifacts")
        archived_path = run_dir.joinpath(*path.parts[artifact_index:])
        if archived_path.is_file():
            path = archived_path
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        relative = os.path.relpath(path.resolve(), report_path.parent.resolve())
    except (OSError, ValueError):
        return f"`{raw_path}`"
    return f"[image]({relative})"


def summarize(run_dir: Path, output_path: Path | None = None) -> str:
    run_dir = run_dir.resolve()
    output_path = (output_path or run_dir / "artifacts" / "screen_report.md").resolve()
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    args = config.get("args", {})
    events = read_jsonl(run_dir / "metrics.jsonl")
    progress = read_progress(run_dir / "progress.txt")
    epochs = sorted(
        (event for event in events if event.get("event") == "step"),
        key=lambda event: (int(event.get("epoch", 0)), int(event.get("global_step", 0))),
    )
    observations = sorted(
        (event for event in events if event.get("event") == "training_observation"),
        key=lambda event: int(event.get("global_step", event.get("step", 0))),
    )
    terminal = next(
        (event for event in reversed(events)
         if event.get("event") in {"run_finished", "run_failed"}),
        None,
    )
    status = terminal.get("status", "unknown") if terminal else progress.get("status", "incomplete")
    if terminal and terminal.get("event") == "run_failed":
        status = f"failed ({terminal.get('error_type', 'unknown error')})"
    total_epochs = args.get("epochs", "—")
    total_steps = int(total_epochs) * int(args.get("samples_per_epoch", 0)) if str(total_epochs).isdigit() else None

    lines = [
        "# VFP-DiT Simple quality screen report", "",
        f"- **Run:** `{config.get('run_id', run_dir.name)}`",
        f"- **Status:** {status}",
        f"- **Started:** {config.get('started_at', '—')}",
        f"- **Learning rate:** {fmt(args.get('lr'))} ({args.get('lr_scheduler', '—')})",
        f"- **Progress:** {progress.get('global_step', '—')} / {total_steps or '—'} optimizer steps; "
        f"epoch {progress.get('epoch', '—')}; batch {progress.get('batch', '—')}",
        "",
        "## Configuration", "",
        "| Setting | Value |", "|---|---|",
    ]
    config_items = [
        ("resolution", "resolution"), ("model_width", "width"), ("depth", "depth"),
        ("heads", "query heads"), ("kv_heads", "KV heads"),
        ("target_latent_downsample_factor", "target latent downsample"),
        ("latent_downsample_factor", "reference latent downsample"),
        ("timesteps_per_image", "timesteps / image"), ("adapter_depth", "adapter depth"),
        ("condition_layer", "condition tap"), ("condition_dropout", "condition dropout"),
        ("batch_size", "batch size"), ("num_workers", "workers"),
        ("optimizer", "optimizer"), ("apollo_rank", "APOLLO rank"),
        ("amp", "mixed precision"), ("seed", "seed"),
        ("samples_per_epoch", "samples / epoch"), ("validation_samples", "validation samples / epoch"),
    ]
    for key, label in config_items:
        if key in args:
            lines.append(f"| {label} | {fmt(args[key])} |")
    if progress:
        lines.extend(["", "## Current progress snapshot", "", "```text"])
        lines.extend(f"{key}={value}" for key, value in progress.items())
        lines.extend(["```"])

    lines.extend([
        "", "## Epoch metrics", "",
        "| Epoch | Step | Train loss | Validation loss | T2I (n) | TI2I (n) | Train images/s | Train peak allocated/reserved GiB | Val seconds |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for event in epochs:
        lines.append(
            f"| {fmt(event.get('epoch'))} | {fmt(event.get('global_step'))} "
            f"| {metric_cell(event, 'loss')} | {metric_cell(event, 'eval_loss')} "
            f"| {metric_cell(event, 'val_loss_t2i')} ({fmt(event.get('val_loss_t2i_samples'), 0)}) "
            f"| {metric_cell(event, 'val_loss_ti2i')} ({fmt(event.get('val_loss_ti2i_samples'), 0)}) "
            f"| {metric_cell(event, 'train_samples_per_second', 3)} "
            f"| {metric_cell(event, 'train_peak_allocated_gib', 2)} / {metric_cell(event, 'train_peak_reserved_gib', 2)} "
            f"| {metric_cell(event, 'val_seconds', 1)} |"
        )
    if not epochs:
        lines.append("| — | — | — | — | — | — | — | — | — |")

    lines.extend([
        "", "## Interval observations", "",
        "| Step | Epoch | Interval steps | Mean loss | Condition drop | Solver | Scheduler (shift) | Fixed-prompt sample |",
        "|---:|---:|---:|---:|---:|---|---|---|",
    ])
    for event in observations:
        metrics = event.get("metrics") or {}
        lines.append(
            f"| {fmt(event.get('global_step', event.get('step')))} | {fmt(event.get('epoch'))} "
            f"| {fmt(event.get('interval_steps'))} | {metric_cell(event, 'loss')} "
            f"| {fmt(metrics.get('condition_drop_fraction'))} "
            f"| {fmt((event.get('sample') or {}).get('solver'), 0)} "
            f"| {fmt((event.get('sample') or {}).get('scheduler'), 0)} "
            f"({fmt((event.get('sample') or {}).get('flow_shift'), 2)}) "
            f"| {image_link(output_path, (event.get('sample') or {}).get('path'), run_dir)} |"
        )
    if not observations:
        lines.append("| — | — | — | — | — | — | — | — |")

    final_sample = run_dir / "artifacts" / "screen_samples.png"
    if final_sample.is_file():
        lines.extend(["", "## Final samples", "", f"[Open final sample grid]({os.path.relpath(final_sample, output_path.parent)})"])
    lines.extend([
        "", "## Reading this screen", "",
        "A single seed and a bounded validation subset support a learning and pipeline screen. "
        "They do not establish architecture ranking or convergence. Compare fixed-prompt images at the same steps and inspect T2I/TI2I validation losses separately.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing config.json and metrics.jsonl")
    parser.add_argument("--output", type=Path, help="Markdown destination (default: <run-dir>/artifacts/screen_report.md)")
    args = parser.parse_args(argv)
    output_path = args.output or args.run_dir / "artifacts" / "screen_report.md"
    report = summarize(args.run_dir, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
