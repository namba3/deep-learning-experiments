"""Summarize paired VFP-DiT Simple metadata Ada scale/shift arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROTOCOL_FIELDS = (
    "seed", "data_mode", "train_manifest", "multi_edit_data_root",
    "coco_split", "edit_split", "coco_weight", "ti2i_weight",
    "resolution", "resolution_levels", "aspect_ratios",
    "validation_fraction", "validation_samples", "samples_per_epoch",
    "epochs", "batch_size", "grad_accumulation", "num_workers",
    "lr", "weight_decay", "grad_clip", "condition_dropout",
    "optimizer", "apollo_rank", "lr_scheduler", "warmup_steps",
    "warmup_ratio", "min_lr_ratio", "amp", "vlm_model", "vae_model",
    "condition_layer", "condition_dim", "latent_channels",
    "latent_downsample_factor", "fuse_reference_latent_to_vision",
    "vae_latent_mode", "model_width", "depth", "heads", "kv_heads",
    "adapter_depth", "adapter_type", "ff_mult",
    "gradient_checkpointing", "timesteps_per_image",
    "metadata_conditioning", "metadata_scale_mapping", "metadata_shift",
    "metadata_ffn_residual_gate", "attention_head_gate",
    "fuse_same_input_projections",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _fmt_fraction(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.1f}%"


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arm_state(output_dir: Path, run_name: str) -> tuple[str, Path | None]:
    """Return completion or the latest usable checkpoint for an exact arm name."""
    runs_root = output_dir / "runs"
    if not runs_root.is_dir():
        return "fresh", None

    incomplete_checkpoints = []
    for run_dir in sorted(runs_root.iterdir(), reverse=True):
        config_path = run_dir / "config.json"
        if not run_dir.is_dir() or not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if config.get("args", {}).get("run_name") != run_name:
            continue

        events = _read_jsonl(run_dir / "metrics.jsonl")
        terminal = next(
            (event for event in reversed(events)
             if event.get("event") in {"run_finished", "run_failed"}),
            None,
        )
        if terminal and terminal.get("event") == "run_finished" and terminal.get("status") == "completed":
            return "completed", None
        latest_checkpoint = run_dir / "checkpoints" / "checkpoint_latest.safetensors"
        if latest_checkpoint.is_file():
            incomplete_checkpoints.append(latest_checkpoint)

    if incomplete_checkpoints:
        return "resume", incomplete_checkpoints[0]
    return "fresh", None


def summarize(output_dir: Path, expected_run_names: list[str] | None = None) -> str:
    runs_root = output_dir / "runs"
    runs = []
    init_hashes_by_arm: dict[str, set[str]] = {}
    protocols_by_arm: dict[str, list[dict[str, Any]]] = {}
    hash_cache: dict[Path, str | None] = {}
    for run_dir in sorted(runs_root.iterdir() if runs_root.is_dir() else ()):
        config_path = run_dir / "config.json"
        if not run_dir.is_dir() or not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        args = config.get("args", {})
        arm_name = args.get("run_name") or config.get("run_name")
        if arm_name:
            protocols_by_arm.setdefault(str(arm_name), []).append(
                {field: args.get(field) for field in PROTOCOL_FIELDS}
            )
        init_path_value = args.get("init_checkpoint")
        init_hash = None
        if init_path_value:
            init_path = Path(init_path_value).expanduser()
            if not init_path.is_absolute():
                init_path = Path.cwd() / init_path
            if init_path not in hash_cache:
                hash_cache[init_path] = _sha256(init_path)
            init_hash = hash_cache[init_path]
            if init_hash:
                init_hashes_by_arm.setdefault(str(arm_name), set()).add(init_hash)
        events = _read_jsonl(run_dir / "metrics.jsonl")
        epochs = [event for event in events if event.get("event") == "step" and event.get("eval_loss") is not None]
        best = min(epochs, key=lambda event: float(event["eval_loss"])) if epochs else None
        terminal = next(
            (event for event in reversed(events) if event.get("event") in {"run_finished", "run_failed"}),
            None,
        )
        progress_path = run_dir / "progress.txt"
        progress = {}
        if progress_path.is_file():
            for line in progress_path.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator:
                    progress[key.strip()] = value.strip()
        status = terminal.get("status", "unknown") if terminal else progress.get("status", "incomplete")
        runs.append({
            "run_id": config.get("run_id", run_dir.name),
            "arm_name": arm_name,
            "variant": args.get("metadata_conditioning", "unknown"),
            "mapping": args.get("metadata_scale_mapping", "linear"),
            "shift": bool(args.get("metadata_shift", False)),
            "epoch": best.get("epoch") if best else None,
            "eval_loss": best.get("eval_loss") if best else None,
            "t2i": best.get("val_loss_t2i") if best else None,
            "ti2i": best.get("val_loss_ti2i") if best else None,
            "status": status,
            "init_hash": init_hash,
            "diagnostics": ({
                key: value for key, value in best.items()
                if key.startswith(("metadata_attn_", "metadata_ffn_"))
            } if best else {}),
            "run_dir": run_dir,
        })

    # A resumed attempt has no --init-checkpoint argument. Carry its original
    # initialization identity forward from an earlier attempt of the same arm.
    for run in runs:
        hashes = init_hashes_by_arm.get(str(run["arm_name"]), set())
        if run["init_hash"] is None and len(hashes) == 1:
            run["init_hash"] = next(iter(hashes))

    observed_names = {run["arm_name"] for run in runs if run["arm_name"]}
    completed_names = {
        run["arm_name"] for run in runs
        if run["arm_name"] and run["status"] == "completed"
    }
    expected_run_names = expected_run_names or []
    init_identities = set().union(*init_hashes_by_arm.values()) if init_hashes_by_arm else set()
    protocol_values: dict[str, set[str]] = {}
    for arm_name in (expected_run_names or list(protocols_by_arm)):
        for protocol in protocols_by_arm.get(arm_name, []):
            for field, value in protocol.items():
                protocol_values.setdefault(field, set()).add(
                    json.dumps(value, sort_keys=True, separators=(",", ":"))
                )
    protocol_mismatches = sorted(
        field for field, values in protocol_values.items() if len(values) > 1
    )
    lines = [
        "# VFP-DiT Simple Ada scale/shift comparison", "",
        "Lower validation flow-MSE is better. All arms are intended to use the same seed and a shared no-metadata initialization checkpoint.",
        "The `none` baseline has no metadata path. Ada arms apply the selected scale mapping to both attention and FFN inputs; shift, when enabled, is added to both branch inputs.",
        "", "| Run | Conditioning | Scale mapping | Shift | Best epoch | Val flow MSE | T2I MSE | TI2I MSE | Init SHA-256 | Status |", "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for run in runs:
        lines.append(
            f"| `{run['run_id']}` | {run['variant']} | {run['mapping']} | {int(run['shift'])} "
            f"| {_fmt(run['epoch'])} | {_fmt(run['eval_loss'])} | {_fmt(run['t2i'])} "
            f"| {_fmt(run['ti2i'])} | {run['init_hash'][:12] if run['init_hash'] else '—'} | {run['status']} |"
        )
    diagnostic_runs = [run for run in runs if run["diagnostics"]]
    if diagnostic_runs:
        lines.extend([
            "", "## Ada scale diagnostics at best validation epoch", "",
            "Scale statistics are metadata-derived and exclude the FFN timestep scale. Near-zero means `|scale| < 0.1`.",
            "", "| Run | Attn mean | Attn min | Attn max | Attn near-zero | Attn negative | FFN mean | FFN min | FFN max | FFN near-zero | FFN negative |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for run in diagnostic_runs:
            diagnostics = run["diagnostics"]
            lines.append(
                f"| `{run['run_id']}` "
                f"| {_fmt(diagnostics.get('metadata_attn_scale_mean'))} "
                f"| {_fmt(diagnostics.get('metadata_attn_scale_min'))} "
                f"| {_fmt(diagnostics.get('metadata_attn_scale_max'))} "
                f"| {_fmt_fraction(diagnostics.get('metadata_attn_scale_near_zero_fraction'))} "
                f"| {_fmt_fraction(diagnostics.get('metadata_attn_scale_negative_fraction'))} "
                f"| {_fmt(diagnostics.get('metadata_ffn_scale_mean'))} "
                f"| {_fmt(diagnostics.get('metadata_ffn_scale_min'))} "
                f"| {_fmt(diagnostics.get('metadata_ffn_scale_max'))} "
                f"| {_fmt_fraction(diagnostics.get('metadata_ffn_scale_near_zero_fraction'))} "
                f"| {_fmt_fraction(diagnostics.get('metadata_ffn_scale_negative_fraction'))} |"
            )
        if any("metadata_attn_shift_rms" in run["diagnostics"] for run in diagnostic_runs):
            lines.extend([
                "", "### Ada shift magnitude", "",
                "| Run | Attn shift RMS | Attn shift absmax | FFN shift RMS | FFN shift absmax |",
                "|---|---:|---:|---:|---:|",
            ])
            for run in diagnostic_runs:
                diagnostics = run["diagnostics"]
                lines.append(
                    f"| `{run['run_id']}` "
                    f"| {_fmt(diagnostics.get('metadata_attn_shift_rms'))} "
                    f"| {_fmt(diagnostics.get('metadata_attn_shift_absmax'))} "
                    f"| {_fmt(diagnostics.get('metadata_ffn_shift_rms'))} "
                    f"| {_fmt(diagnostics.get('metadata_ffn_shift_absmax'))} |"
                )
    if expected_run_names:
        missing = [name for name in expected_run_names if name not in observed_names]
        incomplete = [name for name in expected_run_names if name in observed_names and name not in completed_names]
        lines.extend([
            "", "## Arm coverage", "",
            f"Completed: {len(expected_run_names) - len(missing) - len(incomplete)}/{len(expected_run_names)}; "
            f"missing: {len(missing)}; incomplete: {len(incomplete)}.",
        ])
        if missing:
            lines.append("Missing: " + ", ".join(f"`{name}`" for name in missing))
        if incomplete:
            lines.append("Not completed: " + ", ".join(f"`{name}`" for name in incomplete))
    if init_identities:
        identity_note = (
            f"{len(init_identities)} distinct initialization checkpoint SHA-256 value(s) found "
            f"across {len(init_hashes_by_arm)} arm(s)."
        )
    else:
        identity_note = "No readable initialization checkpoint hashes were found in run configs."
    lines.extend(["", "## Initialization audit", "", identity_note])
    if len(init_identities) > 1:
        lines.append("Warning: initialization hashes differ across arms; this is not a matched-initialization comparison.")
    elif len(init_identities) == 1 and expected_run_names:
        unhashed = [name for name in expected_run_names if name in observed_names and not init_hashes_by_arm.get(name)]
        if unhashed:
            lines.append("Hash unavailable for: " + ", ".join(f"`{name}`" for name in unhashed))
    lines.extend(["", "## Protocol audit", ""])
    if not protocol_values:
        lines.append("No comparison run configs are available for protocol audit.")
    elif protocol_mismatches:
        lines.append(
            "Warning: protocol values differ for: "
            + ", ".join(f"`{field}`" for field in protocol_mismatches)
            + ". Review these run configs before interpreting the arms as matched."
        )
    else:
        lines.append(
            f"No differences found across the recorded comparison fields "
            f"({len(protocol_values)} fields checked)."
        )
    lines.extend([
        "", f"Runs found: {len(runs)}", "",
        "This is a screening report. Treat the validation subset and single-seed ranking as preliminary; inspect run-level reports and fixed-prompt samples before selecting a scale/bias design.", "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm-state", action="store_true",
                        help="Print fresh, completed, or resume:<checkpoint> for --run-name.")
    parser.add_argument("--run-name")
    parser.add_argument("--expected-run-name", action="append", default=[],
                        help="Arm expected by the launcher; may be repeated to audit matrix coverage.")
    args = parser.parse_args(argv)
    if args.arm_state:
        if not args.run_name:
            parser.error("--arm-state requires --run-name")
        state, checkpoint = arm_state(args.output_dir, args.run_name)
        print(f"resume:{checkpoint}" if state == "resume" else state)
        return
    report = summarize(args.output_dir, args.expected_run_name)
    path = args.output_dir / "ada_scale_shift_comparison.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Saved report: {path}")


if __name__ == "__main__":
    main()
