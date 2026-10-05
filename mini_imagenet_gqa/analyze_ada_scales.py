"""Measure trained AdaRMS effective scales at fixed bucket metadata on CPU."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from mini_imagenet_gqa.model import MiniImageNetGQAModel, _ada_scale_offset


DEFAULT_VARIANTS = (
    "ada_gated_gqa_silu_gated_ffn",
    "ada_1plus_silu_gated_gqa_silu_gated_ffn",
    "ada_softplus1_norm_gated_gqa_silu_gated_ffn",
)
CHECKPOINT_KEY = "mini_imagenet_gqa.checkpoint"


def parse_size(value: str) -> tuple[int, int]:
    try:
        height_text, width_text = value.lower().split("x", maxsplit=1)
        height, width = int(height_text), int(width_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("size must be HxW, for example 64x64") from error
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("size dimensions must be positive")
    return height, width


def _stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().to(device="cpu", dtype=torch.float64).flatten()
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "fraction_below_zero": float((values < 0.0).double().mean()),
        "fraction_zero_to_one": float(((values >= 0.0) & (values <= 1.0)).double().mean()),
        "fraction_above_one": float((values > 1.0).double().mean()),
    }


def _collect_run_dirs(output_dir: Path, variants: set[str]) -> list[Path]:
    """Select the highest-epoch best checkpoint per seed/variant."""
    candidates: dict[tuple[int, str], list[tuple[int, int, Path]]] = defaultdict(list)
    for config_path in output_dir.glob("runs/*/config.json"):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            args = config["args"]
            variant = args.get("variant")
            seed = int(args["seed"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        if variant not in variants:
            continue
        checkpoint_path = config_path.parent / "checkpoints" / "checkpoint_best.safetensors"
        if not checkpoint_path.is_file():
            continue
        try:
            with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
                record = json.loads((checkpoint.metadata() or {}).get(CHECKPOINT_KEY, "{}"))
            epoch = int(record.get("epoch", 0))
        except Exception:
            # Ignore stale/truncated attempts; a valid older checkpoint for the
            # same seed/variant may still be present in the output directory.
            epoch = 0
            try:
                with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
                    checkpoint.metadata()
            except Exception:
                continue
        candidates[(seed, variant)].append((epoch, checkpoint_path.stat().st_mtime_ns, config_path.parent))
    selected = []
    for key in sorted(candidates):
        selected.append(max(candidates[key], key=lambda item: (item[0], item[1]))[2])
    return selected


def _model_from_run(run_dir: Path) -> tuple[MiniImageNetGQAModel, dict]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    args = config["args"]
    model = MiniImageNetGQAModel(
        num_classes=int(args["num_classes"]),
        widths=tuple(args["widths"]),
        heads=int(args["heads"]),
        kv_heads=int(args["kv_heads"]),
        blocks_per_stage=int(args["blocks_per_stage"]),
        image_size=int(args["image_size"]),
        variant=str(args["variant"]),
        dropout=float(args["dropout"]),
        ff_mult=float(args["ff_mult"]),
    ).cpu().eval()
    state = load_file(str(run_dir / "checkpoints" / "checkpoint_best.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    return model, args


def analyze_checkpoint(run_dir: Path, bucket_size: tuple[int, int], optimizer: str) -> dict:
    model, args = _model_from_run(run_dir)
    height, width = bucket_size
    metadata = torch.tensor(
        [[math.log(math.sqrt(height * width)), math.log(width / height)]],
        dtype=torch.float32,
    )
    assert model.meta_embedding is not None
    with torch.inference_mode():
        condition = model.meta_embedding(metadata)
        attn_values = []
        ffn_values = []
        stage_rows = []
        for stage_index, stage in enumerate(model.stages):
            for block_index, block in enumerate(stage.blocks):
                if block.norm1_scale is not None:
                    attn = 1.0 + _ada_scale_offset(
                        block.norm1_scale(condition), block.ada_scale_mode,
                    )
                    ffn = 1.0 + _ada_scale_offset(
                        block.norm2_scale(condition), block.ada_scale_mode,
                    ) if block.norm2_scale is not None else None
                    attn_values.append(attn.reshape(-1))
                    if ffn is not None:
                        ffn_values.append(ffn.reshape(-1))
                    stage_rows.append({
                        "stage": stage_index,
                        "block": block_index,
                        "attention": _stats(attn),
                        "ffn": _stats(ffn) if ffn is not None else None,
                    })
    if not attn_values or not ffn_values:
        raise ValueError(f"No Ada scale projections found in {run_dir}")
    return {
        "optimizer": optimizer,
        "seed": int(args["seed"]),
        "variant": str(args["variant"]),
        "run_id": json.loads((run_dir / "config.json").read_text())["run_id"],
        "checkpoint": str(run_dir / "checkpoints" / "checkpoint_best.safetensors"),
        "bucket_height": height,
        "bucket_width": width,
        "metadata": metadata.flatten().tolist(),
        "attention": _stats(torch.cat(attn_values)),
        "ffn": _stats(torch.cat(ffn_values)),
        "blocks": stage_rows,
    }


def _write_report(path: Path, report: dict) -> None:
    lines = [
        "# Mini-ImageNet AdaRMS effective-scale analysis",
        "",
        f"- Fixed bucket: `{report['bucket_height']}x{report['bucket_width']}` (H×W)",
        f"- Metadata: `log(sqrt(HW))={report['metadata'][0]:.6f}`, `log(W/H)={report['metadata'][1]:.6f}`",
        "- Checkpoint: best-validation checkpoint for each seed/variant; duplicated attempts are reduced to the checkpoint with the greatest recorded epoch.",
        "- Scale definition: effective multiplier `1 + mapped Ada offset`, exactly as applied before the branch RMS normalization.",
        "- Statistics pool every channel across all Ada blocks within a checkpoint; standard deviation is population std.",
        "- Fractions use `<0`, inclusive `[0,1]`, and `>1`; the three bins partition finite scale values.",
        "- The first table averages statistics equally across seeds; min/max columns are means of per-seed extrema, not global extrema.",
        "- AdamW and AdamW-SF runs differ in schedule policy (AdamW cosine vs AdamW-SF schedule-free), so this is a trained-checkpoint comparison, not an isolated optimizer update-rule effect.",
        "",
        "| Optimizer | Variant | Seeds | Branch | Mean seed-min | Mean seed-max | Mean scale | Mean std | <0 | 0–1 | >1 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in report["runs"]:
        for branch in ("attention", "ffn"):
            grouped[(row["optimizer"], row["variant"], branch)].append(row[branch])
    for (optimizer, variant, branch), stats_rows in sorted(grouped.items()):
        # Aggregate individual run-level moments equally by seed; the raw per-seed
        # rows remain in JSON so this summary cannot hide seed variability.
        avg = {
            key: statistics.mean(row[key] for row in stats_rows)
            for key in (
                "min", "max", "mean", "std", "fraction_below_zero",
                "fraction_zero_to_one", "fraction_above_one",
            )
        }
        seeds = ", ".join(
            str(row["seed"]) for row in report["runs"]
            if row["optimizer"] == optimizer and row["variant"] == variant
        )
        lines.append(
            f"| {optimizer} | `{variant}` | {seeds} | {branch} | {avg['min']:.4f} | {avg['max']:.4f} "
            f"| {avg['mean']:.4f} | {avg['std']:.4f} | {avg['fraction_below_zero']:.3%} "
            f"| {avg['fraction_zero_to_one']:.3%} | {avg['fraction_above_one']:.3%} |"
        )
    lines.extend(["", "## Per-seed results", "", "| Optimizer | Seed | Variant | Branch | Mean ± std | Min–max | <0 | 0–1 | >1 |", "|---|---:|---|---|---:|---:|---:|---:|---:|"])
    for row in sorted(report["runs"], key=lambda item: (item["optimizer"], item["variant"], item["seed"])):
        a, f = row["attention"], row["ffn"]
        for branch, stats in (("attention", a), ("ffn", f)):
            lines.append(
                f"| {row['optimizer']} | {row['seed']} | `{row['variant']}` | {branch} "
                f"| {stats['mean']:.4f} ± {stats['std']:.4f} | [{stats['min']:.4f}, {stats['max']:.4f}] "
                f"| {stats['fraction_below_zero']:.3%} | {stats['fraction_zero_to_one']:.3%} "
                f"| {stats['fraction_above_one']:.3%} |"
            )
    by_optimizer_seed = {
        (row["optimizer"], row["variant"], row["seed"]): row
        for row in report["runs"]
    }
    if {row["optimizer"] for row in report["runs"]} >= {"AdamW", "AdamW-SF"}:
        lines.extend([
            "",
            "## Paired optimizer difference",
            "",
            "Values are AdamW-SF minus AdamW, averaged across matched seeds; percentages are percentage-point differences.",
            "",
            "| Variant | Branch | Seeds | Δ mean scale | Δ <0 | Δ 0–1 | Δ >1 |",
            "|---|---|---|---:|---:|---:|---:|",
        ])
        for variant in sorted({row["variant"] for row in report["runs"]}):
            seeds = sorted(
                seed for seed in {row["seed"] for row in report["runs"]}
                if ("AdamW", variant, seed) in by_optimizer_seed
                and ("AdamW-SF", variant, seed) in by_optimizer_seed
            )
            for branch in ("attention", "ffn"):
                differences = []
                for seed in seeds:
                    adamw = by_optimizer_seed[("AdamW", variant, seed)][branch]
                    adamw_sf = by_optimizer_seed[("AdamW-SF", variant, seed)][branch]
                    differences.append({key: adamw_sf[key] - adamw[key] for key in (
                        "mean", "fraction_below_zero", "fraction_zero_to_one", "fraction_above_one",
                    )})
                if not differences:
                    continue
                averages = {
                    key: statistics.mean(item[key] for item in differences)
                    for key in differences[0]
                }
                lines.append(
                    f"| `{variant}` | {branch} | {', '.join(map(str, seeds))} "
                    f"| {averages['mean']:+.4f} | {averages['fraction_below_zero'] * 100:+.2f} pp "
                    f"| {averages['fraction_zero_to_one'] * 100:+.2f} pp "
                    f"| {averages['fraction_above_one'] * 100:+.2f} pp |"
                )
    lines.extend(["", "## Interpretation boundary", "", "These are deterministic scale-projection outputs for one fixed metadata vector. They do not measure token activations after RMSNorm, data-weighted exposure across buckets, or causal impact on accuracy. Use the per-block JSON for layer-local inspection.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", action="append", default=None, metavar="OPTIMIZER=DIR",
        help="Optimizer label and run output directory; repeat for each optimizer.",
    )
    parser.add_argument("--bucket-size", type=parse_size, default=(64, 64), metavar="HxW")
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument(
        "--report-dir", type=Path,
        default=Path("mini_imagenet_gqa/output/bucketed/adamw-vs-sf-ada-seeds47-50/ada-scale-diagnostics"),
    )
    args = parser.parse_args()

    inputs = args.input or [
        "AdamW=mini_imagenet_gqa/output/bucketed/ada-scale-vs-no-ada-same-lr-10e-seeds47-50",
        "AdamW-SF=mini_imagenet_gqa/output/bucketed/adamw-sf-ada-seeds47-50",
    ]
    runs = []
    for spec in inputs:
        optimizer, separator, directory = spec.partition("=")
        if not separator or not optimizer or not directory:
            raise SystemExit(f"invalid --input {spec!r}; expected OPTIMIZER=DIR")
        output_dir = Path(directory)
        run_dirs = _collect_run_dirs(output_dir, set(args.variants))
        if not run_dirs:
            raise SystemExit(f"No best checkpoints found under {output_dir} for {args.variants}")
        runs.extend(
            analyze_checkpoint(run_dir, args.bucket_size, optimizer)
            for run_dir in run_dirs
        )
    report = {
        "bucket_height": args.bucket_size[0],
        "bucket_width": args.bucket_size[1],
        "metadata": runs[0]["metadata"],
        "runs": runs,
    }
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"effective_scales_{args.bucket_size[0]}x{args.bucket_size[1]}.json"
    md_path = report_dir / f"effective_scales_{args.bucket_size[0]}x{args.bucket_size[1]}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_report(md_path, report)
    print(f"Analyzed {len(runs)} checkpoints on CPU")
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
