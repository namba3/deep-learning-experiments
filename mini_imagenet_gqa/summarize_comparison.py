"""Aggregate completed Mini-ImageNet GQA runs into JSON and Markdown reports."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


_PROTOCOL_KEYS = (
    "dataset", "image_size", "widths", "heads", "kv_heads", "blocks_per_stage",
    "dropout", "ff_mult", "epochs", "batch_size", "lr", "weight_decay", "amp",
    "bucket_sizes", "transform_degrees", "transform_shear", "steps_per_epoch", "eval_batches",
    "device", "num_workers", "common_init", "deterministic", "optimizer",
    "adamw_sf_backend", "apollo_rank", "apollo_fallback",
    "apollo_matrix_fallback", "lr_scheduler", "warmup_steps",
    "warmup_ratio", "min_lr_ratio", "lr_schedule_mode",
    "apollo_sf_quant_block_size", "apollo_sf_delta_refresh",
    "apollo_sf_delta_refresh_window", "apollo_update_proj_gap",
)


def _read_events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def arm_state(output_dir: Path, run_name: str) -> tuple[str, Path | None]:
    """Return completed, resume:<checkpoint>, or fresh for an exact run name."""
    runs_root = output_dir / "runs"
    if not runs_root.is_dir():
        return "fresh", None
    resume_candidates = []
    for run_dir in runs_root.iterdir():
        config_path = run_dir / "config.json"
        if not run_dir.is_dir() or not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if config.get("args", {}).get("run_name") != run_name:
            continue
        events = _read_events(run_dir / "metrics.jsonl")
        terminal = next(
            (event for event in reversed(events)
             if event.get("event") in {"run_finished", "run_failed"}),
            None,
        )
        if terminal and terminal.get("event") == "run_finished" and terminal.get("status") == "completed":
            return "completed", None
        if any(event.get("event") == "test" for event in events) and any(
            event.get("event") == "optimizer_state" for event in events
        ):
            # Evaluation and optimizer accounting finished; avoid duplicating a
            # completed measurement if shutdown happened just before finish().
            return "completed", None
        checkpoint = run_dir / "checkpoints" / "checkpoint_latest.safetensors"
        state_path = checkpoint.with_suffix(".resume.pt")
        if checkpoint.is_file() and state_path.is_file() and state_path.stat().st_size > 0:
            resume_candidates.append((checkpoint.stat().st_mtime_ns, checkpoint))
    if resume_candidates:
        return "resume", max(resume_candidates, key=lambda item: item[0])[1]
    return "fresh", None


def _read_run(run_dir: Path) -> dict | None:
    config_path = run_dir / "config.json"
    events_path = run_dir / "metrics.jsonl"
    if not config_path.is_file() or not events_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args = config.get("args", {})
    # A reboot or forced termination can leave a partially written final JSONL
    # record. Keep valid events so completed runs remain aggregatable.
    events = _read_events(events_path)
    # Resume attempts get a new run directory. Fold in the prior attempt's
    # epoch records so best-validation and throughput still cover the full run.
    resume_path = args.get("resume")
    visited = {run_dir.resolve()}
    previous_events: list[dict] = []
    while resume_path:
        checkpoint = Path(resume_path).expanduser()
        previous_run = checkpoint.parent.parent
        try:
            resolved_previous = previous_run.resolve(strict=True)
        except OSError:
            break
        if resolved_previous in visited:
            break
        visited.add(resolved_previous)
        previous_events = _read_events(previous_run / "metrics.jsonl") + previous_events
        previous_config_path = previous_run / "config.json"
        try:
            previous_config = json.loads(previous_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            break
        resume_path = previous_config.get("args", {}).get("resume")
    events = previous_events + events
    epochs = [event for event in events if event.get("event") == "epoch"]
    test_events = [event for event in events if event.get("event") == "test"]
    if not epochs or not test_events:
        return None
    best_epoch = max(epochs, key=lambda event: event.get("validation", {}).get("top1", float("-inf")))
    train_samples = sum(event.get("train", {}).get("samples", 0.0) for event in epochs)
    train_seconds = sum(event.get("train", {}).get("seconds", 0.0) for event in epochs)
    memory_allocated = [
        event["train"]["peak_allocated_mb"]
        for event in epochs if "peak_allocated_mb" in event.get("train", {})
    ]
    memory_reserved = [
        event["train"]["peak_reserved_mb"]
        for event in epochs if "peak_reserved_mb" in event.get("train", {})
    ]
    test_metrics = test_events[-1].get("metrics", {})
    model_events = [event for event in events if event.get("event") == "model"]
    model_event = model_events[-1] if model_events else {}
    optimizer_state_events = [
        event for event in events if event.get("event") == "optimizer_state"
    ]
    optimizer_state_event = optimizer_state_events[-1] if optimizer_state_events else {}
    return {
        "run_id": config.get("run_id", run_dir.name),
        "variant": args.get("variant", "unknown"),
        "seed": int(args.get("seed", 0)),
        "parameters": int(model_event["parameters"]) if model_event else None,
        "common_init_copied_parameters": model_event.get("common_init_copied_parameters"),
        "initial_state_sha256": model_event.get("initial_state_sha256"),
        "protocol": {key: args[key] for key in _PROTOCOL_KEYS if key in args},
        "best_epoch": int(best_epoch.get("epoch", 0)),
        "best_validation_top1": float(best_epoch["validation"]["top1"]),
        "test_top1": float(test_metrics["top1"]),
        "test_loss": float(test_metrics["loss"]),
        "train_samples_per_second": train_samples / max(train_seconds, 1e-12),
        "peak_allocated_mb": max(memory_allocated) if memory_allocated else None,
        "peak_reserved_mb": max(memory_reserved) if memory_reserved else None,
        "persistent_state_mib": (
            optimizer_state_event.get("persistent_state_bytes", 0) / (1024 ** 2)
            if optimizer_state_event else None
        ),
        "sf_delta_commit_count": optimizer_state_event.get(
            "sf_delta_commit_count", 0,
        ),
        "sf_delta_commit_norm_sum": optimizer_state_event.get(
            "sf_delta_commit_norm_sum", 0.0,
        ),
        "sf_delta_commit_norm_max": optimizer_state_event.get(
            "sf_delta_commit_norm_max", 0.0,
        ),
        "sf_delta_blend_count": optimizer_state_event.get(
            "sf_delta_blend_count", 0,
        ),
        "sf_delta_blend_step_count": optimizer_state_event.get(
            "sf_delta_blend_step_count", 0,
        ),
        "sf_delta_blend_norm_sum": optimizer_state_event.get(
            "sf_delta_blend_norm_sum", 0.0,
        ),
        "parameter_counts_by_backend": optimizer_state_event.get(
            "parameter_counts_by_backend", {},
        ),
        "parameter_numel_by_backend": optimizer_state_event.get(
            "parameter_numel_by_backend", {},
        ),
    }


def summarize(output_dir: Path) -> dict:
    runs_root = output_dir / "runs"
    runs = [
        run for run_dir in sorted(runs_root.iterdir())
        if run_dir.is_dir() and (run := _read_run(run_dir)) is not None
    ] if runs_root.is_dir() else []
    protocol_signatures = {
        json.dumps(run["protocol"], sort_keys=True, separators=(",", ":"))
        for run in runs
    }
    if len(protocol_signatures) > 1:
        raise ValueError(
            "output directory contains incompatible comparison settings; "
            "use a separate OUTPUT_DIR for each protocol"
        )
    protocol = runs[0]["protocol"] if runs else {}
    groups: dict[str, list[dict]] = {}
    for run in runs:
        groups.setdefault(run["variant"], []).append(run)
    metrics = (
        "best_validation_top1", "test_top1", "test_loss",
        "train_samples_per_second", "peak_allocated_mb", "peak_reserved_mb",
        "persistent_state_mib",
        "sf_delta_commit_count", "sf_delta_commit_norm_sum",
        "sf_delta_commit_norm_max",
        "sf_delta_blend_count", "sf_delta_blend_step_count",
        "sf_delta_blend_norm_sum",
    )
    variants = {}
    for variant, group in sorted(groups.items()):
        summary = {"seeds": sorted(run["seed"] for run in group), "runs": len(group)}
        for metric in metrics:
            values = [run[metric] for run in group if run[metric] is not None]
            summary[metric] = {
                "n": len(values),
                "mean": statistics.fmean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
            }
        backend_keys = sorted({
            key
            for run in group
            for key in run.get("parameter_counts_by_backend", {})
        })
        summary["parameter_counts_by_backend"] = {
            backend: {
                "n": len(values := [
                    run["parameter_counts_by_backend"][backend]
                    for run in group
                    if backend in run.get("parameter_counts_by_backend", {})
                ]),
                "mean": statistics.fmean(values),
            }
            for backend in backend_keys
        }
        summary["parameter_numel_by_backend"] = {
            backend: {
                "n": len(values := [
                    run["parameter_numel_by_backend"][backend]
                    for run in group
                    if backend in run.get("parameter_numel_by_backend", {})
                ]),
                "mean": statistics.fmean(values),
            }
            for backend in backend_keys
        }
        summary["parameters"] = group[0]["parameters"]
        variants[variant] = summary
    return {"runs": runs, "variants": variants, "protocol": protocol}


def _format_stat(value: dict | None, *, percent: bool = False) -> str:
    if value is None or value.get("mean") is None:
        return "—"
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    mean = f"{value['mean'] * scale:.2f}{suffix}"
    if value.get("n") == 1:
        return f"{mean} (n=1)"
    return f"{mean} ± {value['std'] * scale:.2f}{suffix}"


def write_reports(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison_summary.json"
    markdown_path = output_dir / "comparison_summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = [
        "# Mini-ImageNet GQA comparison summary", "",
        "For one run, cells show the observed value with n=1; otherwise they show mean ± sample standard deviation.",
        "Memory and throughput summarize the measured run-level values.",
        f"Protocol: `{json.dumps(report.get('protocol', {}), sort_keys=True, separators=(',', ':')) or 'unspecified'}`",
    ]
    protocol = report.get("protocol", {})
    if protocol.get("steps_per_epoch", 0) or protocol.get("eval_batches", 0):
        rows.append("Capped screening runs: accuracy is not a full-split comparison result.")
    rows.extend([
        "",
        "| Variant | Seeds | Params | Best val top-1 | Test top-1 | Test loss | Images/s | Peak allocated MB | Peak reserved MB | Optimizer state MiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for variant, summary in report["variants"].items():
        rows.append(
            f"| {variant} | {', '.join(map(str, summary['seeds']))} | {summary['parameters'] or '—'} "
            f"| {_format_stat(summary['best_validation_top1'], percent=True)} "
            f"| {_format_stat(summary['test_top1'], percent=True)} "
            f"| {_format_stat(summary['test_loss'])} "
            f"| {_format_stat(summary['train_samples_per_second'])} "
            f"| {_format_stat(summary['peak_allocated_mb'])} "
            f"| {_format_stat(summary['peak_reserved_mb'])} "
            f"| {_format_stat(summary['persistent_state_mib'])} |"
        )
    rows.extend(["", "## Parameter allocation by optimizer backend", ""])
    for variant, summary in report["variants"].items():
        backends = sorted(set(summary["parameter_counts_by_backend"]) | set(
            summary["parameter_numel_by_backend"]
        ))
        allocations = []
        for backend in backends:
            tensors = summary["parameter_counts_by_backend"].get(backend)
            elements = summary["parameter_numel_by_backend"].get(backend)
            tensor_text = (
                f"{tensors['mean']:.1f} tensors (n={tensors['n']})"
                if tensors else "tensor count unavailable"
            )
            element_text = (
                f"{elements['mean']:.0f} elements (n={elements['n']})"
                if elements else "element count unavailable"
            )
            allocations.append(f"{backend}: {tensor_text}, {element_text}")
        rows.append(f"- **{variant}**: {', '.join(allocations) or 'not recorded'}.")
    rows.extend(["", f"Completed runs: {len(report['runs'])}", ""])
    markdown_path.write_text("\n".join(rows), encoding="utf-8")
    return json_path, markdown_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("mini_imagenet_gqa/output/bucketed"))
    parser.add_argument("--arm-state", action="store_true",
                        help="Print fresh, completed, or resume:<checkpoint> for --run-name.")
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)
    if args.arm_state:
        if not args.run_name:
            parser.error("--arm-state requires --run-name")
        state, checkpoint = arm_state(args.output_dir, args.run_name)
        print(f"resume:{checkpoint}" if state == "resume" else state)
        return
    report = summarize(args.output_dir)
    json_path, markdown_path = write_reports(report, args.output_dir)
    print(f"completed_runs={len(report['runs'])} variants={len(report['variants'])}")
    print(f"wrote {markdown_path} and {json_path}")


if __name__ == "__main__":
    main()
