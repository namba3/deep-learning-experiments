"""Pair AdaRMS scale/shift runs with existing same-seed scale-only controls."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .summarize_comparison import summarize


SEEDS = (47, 48, 49, 50)
CONDITIONS = {
    "no_ada": ("No Ada baseline", "gated_gqa_silu_gated_ffn", "scale_only"),
    "linear_scale": ("1+s", "ada_gated_gqa_silu_gated_ffn", "scale_only"),
    "linear_shift": ("1+s + Ada bias", "ada_shift_gated_gqa_silu_gated_ffn", "shift"),
    "softplus_scale": (
        "Normalized Softplus",
        "ada_softplus1_norm_gated_gqa_silu_gated_ffn",
        "scale_only",
    ),
    "softplus_shift": (
        "Normalized Softplus + Ada bias",
        "ada_softplus1_norm_shift_gated_gqa_silu_gated_ffn",
        "shift",
    ),
}


def _index_runs(report: dict) -> dict[tuple[str, int], dict]:
    return {
        (run["variant"], int(run["seed"])): run
        for run in report["runs"]
    }


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _comparable_protocol(protocol: dict) -> dict:
    """Drop optimizer-specific settings that cannot affect the recorded runs."""
    comparable = dict(protocol)
    if comparable.get("optimizer") != "APOLLO":
        for key in tuple(comparable):
            if key.startswith("apollo_"):
                comparable.pop(key)
    return comparable


def compare(scale_only_dir: Path, shift_dir: Path) -> dict:
    scale_report = summarize(scale_only_dir)
    shift_report = summarize(shift_dir)
    scale_protocol = _comparable_protocol(scale_report["protocol"])
    shift_protocol = _comparable_protocol(shift_report["protocol"])
    if scale_protocol != shift_protocol:
        differences = [
            f"{key}: scale-only={scale_protocol.get(key, '<missing>')!r}, "
            f"shift={shift_protocol.get(key, '<missing>')!r}"
            for key in sorted(set(scale_protocol) | set(shift_protocol))
            if scale_protocol.get(key) != shift_protocol.get(key)
        ]
        raise ValueError(
            "scale-only and shift runs have different training protocols: "
            + "; ".join(differences)
        )
    scale_runs = _index_runs(scale_report)
    shift_runs = _index_runs(shift_report)
    all_runs = {**scale_runs, **shift_runs}

    rows = {}
    missing = []
    for key, (label, variant, source) in CONDITIONS.items():
        values = {}
        for seed in SEEDS:
            run = all_runs.get((variant, seed))
            if run is None:
                missing.append({"condition": key, "variant": variant, "seed": seed})
                continue
            values[str(seed)] = {
                "test_top1": float(run["test_top1"]),
                "best_validation_top1": float(run["best_validation_top1"]),
                "run_id": run["run_id"],
            }
        scores = [entry["test_top1"] for entry in values.values()]
        mean, std = _mean_std(scores)
        baseline = {
            seed: all_runs.get((CONDITIONS["no_ada"][1], seed))
            for seed in SEEDS
        }
        deltas = [
            float(all_runs[(variant, seed)]["test_top1"])
            - float(baseline[seed]["test_top1"])
            for seed in SEEDS
            if (variant, seed) in all_runs and baseline[seed] is not None
        ]
        delta_mean, delta_std = _mean_std(deltas)
        rows[key] = {
            "label": label,
            "variant": variant,
            "source": source,
            "seeds": values,
            "n": len(scores),
            "test_top1_mean": mean,
            "test_top1_sample_std": std,
            "paired_delta_vs_no_ada_mean": delta_mean,
            "paired_delta_vs_no_ada_sample_std": delta_std,
            "paired_delta_n": len(deltas),
        }

    paired_shift = {}
    for name, scale_key, shift_key in (
        ("1+s", "linear_scale", "linear_shift"),
        ("Normalized Softplus", "softplus_scale", "softplus_shift"),
    ):
        scale_variant = CONDITIONS[scale_key][1]
        shift_variant = CONDITIONS[shift_key][1]
        deltas = [
            float(all_runs[(shift_variant, seed)]["test_top1"])
            - float(all_runs[(scale_variant, seed)]["test_top1"])
            for seed in SEEDS
            if (shift_variant, seed) in all_runs and (scale_variant, seed) in all_runs
        ]
        mean, std = _mean_std(deltas)
        paired_shift[name] = {
            "n": len(deltas),
            "mean": mean,
            "sample_std": std,
            "per_seed": {
                str(seed): float(all_runs[(shift_variant, seed)]["test_top1"])
                - float(all_runs[(scale_variant, seed)]["test_top1"])
                for seed in SEEDS
                if (shift_variant, seed) in all_runs and (scale_variant, seed) in all_runs
            },
        }
    return {
        "protocol": scale_protocol,
        "seeds_expected": list(SEEDS),
        "conditions": rows,
        "paired_shift_deltas": paired_shift,
        "missing_runs": missing,
    }


def write_report(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ada_bias_comparison.json"
    markdown_path = output_dir / "ada_bias_comparison.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    columns = ["Condition", *map(str, SEEDS), "Test top-1 mean ± SD", "Paired Δ vs No Ada"]
    rows = [
        "# AdaRMS scale and bias comparison",
        "",
        "Test top-1 is reported as mean ± sample SD across available seeds. Paired deltas use the same-seed No Ada baseline.",
        f"Expected seeds: {', '.join(map(str, SEEDS))}.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---", *(["---:"] * (len(columns) - 1))]) + " |",
    ]
    for key, row in report["conditions"].items():
        seed_cells = [
            f"{100 * row['seeds'][str(seed)]['test_top1']:.2f}%"
            if str(seed) in row["seeds"] else "—"
            for seed in SEEDS
        ]
        if row["test_top1_mean"] is None:
            stat = "—"
        elif row["n"] == 1:
            stat = f"{100 * row['test_top1_mean']:.2f}% (n=1)"
        else:
            stat = f"{100 * row['test_top1_mean']:.2f} ± {100 * row['test_top1_sample_std']:.2f}%"
        if row["paired_delta_vs_no_ada_mean"] is None:
            delta = "—"
        else:
            delta = (
                f"{100 * row['paired_delta_vs_no_ada_mean']:+.2f}"
                f" ± {100 * row['paired_delta_vs_no_ada_sample_std']:.2f} pp"
            )
        rows.append(f"| {row['label']} | " + " | ".join(seed_cells + [stat, delta]) + " |")
    rows.extend(["", "## Paired bias effect", ""])
    for label, result in report["paired_shift_deltas"].items():
        if result["mean"] is None:
            value = "no matched seeds yet"
        else:
            value = f"{100 * result['mean']:+.2f} ± {100 * result['sample_std']:.2f} pp (n={result['n']})"
        per_seed = ", ".join(
            f"{seed}: {100 * delta:+.2f} pp"
            for seed, delta in result["per_seed"].items()
        )
        rows.append(f"- **{label}**, shift minus scale-only: {value}. {per_seed}")
    if report["missing_runs"]:
        rows.extend(["", "## Missing runs", ""])
        for missing in report["missing_runs"]:
            rows.append(
                f"- `{missing['condition']}` / seed {missing['seed']} "
                f"(`{missing['variant']}`)"
            )
    rows.append("")
    markdown_path.write_text("\n".join(rows), encoding="utf-8")
    return json_path, markdown_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale-only-dir", type=Path, required=True)
    parser.add_argument("--shift-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare(args.scale_only_dir, args.shift_dir)
    json_path, markdown_path = write_report(report, args.output_dir)
    print(f"missing_runs={len(report['missing_runs'])}")
    print(f"wrote {markdown_path} and {json_path}")


if __name__ == "__main__":
    main()
