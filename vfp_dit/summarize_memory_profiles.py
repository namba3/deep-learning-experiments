"""Summarize VFP-DiT Simple resource-profile runs into one Markdown table."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORY_KEYS = ("train_peak_allocated_gib", "train_peak_reserved_gib")
COMPONENT_KEYS = (
    ("profile_vae_target_encode_seconds", "Target VAE"),
    ("profile_vae_reference_encode_seconds", "Reference VAE"),
    ("profile_qwen_encode_seconds", "Qwen"),
    ("profile_condition_prepare_seconds", "Condition prep"),
    ("profile_dit_forward_seconds", "DiT forward"),
    ("profile_backward_seconds", "Backward"),
    ("profile_optimizer_step_seconds", "Optimizer"),
)


def read_events(path: Path) -> list[dict[str, Any]]:
    """Read complete JSONL objects, ignoring a partially written final line."""
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def read_progress(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    return values


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def relative_link(target: Path, base: Path, label: str) -> str:
    try:
        href = os.path.relpath(target.resolve(), base.resolve())
    except (OSError, ValueError):
        return f"`{label}`"
    return f"[{label}]({href})"


def collect_runs(runs_root: Path) -> list[dict[str, Any]]:
    rows = []
    for run_dir in sorted(runs_root.glob("*/")):
        config_path = run_dir / "config.json"
        if not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        events = read_events(run_dir / "metrics.jsonl")
        steps = [
            event for event in events
            if event.get("event") == "step"
            and any(key in event for key in MEMORY_KEYS)
        ]
        if not steps:
            continue
        step = max(
            steps,
            key=lambda event: (
                int(event.get("epoch", 0)), int(event.get("global_step", 0)),
            ),
        )
        terminal = next(
            (event for event in reversed(events)
             if event.get("event") in {"run_finished", "run_failed"}),
            None,
        )
        progress = read_progress(run_dir / "progress.txt")
        status = (
            terminal.get("status", "unknown") if terminal
            else progress.get("status", "incomplete")
        )
        if terminal and terminal.get("event") == "run_failed":
            status = f"failed: {terminal.get('error_type', 'unknown')}"
        args = config.get("args", {})
        rows.append({
            "run_dir": run_dir,
            "run_id": config.get("run_id", run_dir.name),
            "status": status,
            "args": args,
            "step": step,
            "reference_timing": "yes" if "profile_vae_reference_encode_seconds" in step else "no",
        })
    return sorted(rows, key=lambda row: str(row["run_id"]))


def render(rows: list[dict[str, Any]], report_path: Path) -> str:
    lines = [
        "# VFP-DiT Simple memory and profile summary", "",
        f"Runs with CUDA memory metrics: **{len(rows)}**", "",
        "`Peak overhead` includes live activations, gradients, and temporary workspaces; it is not an activation-only measurement. `Reserved` is allocator memory, not live tensor memory. Component times are synchronized measurements and can add profiling overhead.", "",
        "`Ref timing=yes` means at least one batch recorded reference-VAE encoding in that run; `no` means no such timing field was emitted.", "",
        "## Configuration, memory, and throughput", "",
        "| Run | Status | Resolution levels | Steps/image | Width × depth | Ref timing | Peak alloc / reserved (GiB) | Peak overhead (GiB) | Optimizer state (GiB) | Train time (s) | Samples/s |",
        "|---|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        args, step = row["args"], row["step"]
        levels = args.get("resolution_levels") or [args.get("resolution", "—")]
        levels_text = ", ".join(str(value) for value in levels)
        label = str(row["run_id"])
        lines.append(
            f"| {relative_link(row['run_dir'], report_path.parent, label)}"
            f" | {row['status']} | {levels_text}"
            f" | {fmt(args.get('timesteps_per_image'), 0)}"
            f" | {fmt(args.get('model_width'), 0)} × {fmt(args.get('depth'), 0)}"
            f" | {row['reference_timing']}"
            f" | {fmt(step.get('train_peak_allocated_gib'), 3)} / {fmt(step.get('train_peak_reserved_gib'), 3)}"
            f" | {fmt(step.get('train_peak_overhead_over_persistent_gib'), 3)}"
            f" | {fmt(step.get('train_optimizer_state_gib'), 3)}"
            f" | {fmt(step.get('train_seconds'))}"
            f" | {fmt(step.get('train_samples_per_second'), 3)} |"
        )
    lines += [
        "", "## Component timings (seconds)", "",
        "Values are per-batch component timings averaged over the recorded training batches; `—` means that component was not recorded for the run.", "",
        "| Run | Target VAE | Reference VAE | Qwen | Condition prep | DiT forward | Backward | Optimizer |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        step = row["step"]
        label = str(row["run_id"])
        components = " | ".join(
            fmt(step.get(key)) for key, _ in COMPONENT_KEYS
        )
        lines.append(
            f"| {relative_link(row['run_dir'], report_path.parent, label)} | {components} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root", type=Path,
        default=PROJECT_ROOT / "vfp_dit" / "output" / "runs",
        help="Directory containing one subdirectory per training run.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "vfp_dit" / "output" / "memory_profile_summary.md",
        help="Markdown report output path.",
    )
    args = parser.parse_args()
    rows = collect_runs(args.runs_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(rows, args.output), encoding="utf-8")
    print(f"runs={len(rows)} wrote {args.output}")


if __name__ == "__main__":
    main()
