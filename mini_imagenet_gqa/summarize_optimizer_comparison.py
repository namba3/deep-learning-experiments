"""Compare completed Mini-ImageNet summaries across optimizer protocols."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


_MATCHED_PROTOCOL_KEYS = (
    "dataset", "image_size", "widths", "heads", "kv_heads", "blocks_per_stage",
    "dropout", "ff_mult", "epochs", "batch_size", "lr", "weight_decay", "amp",
    "bucket_sizes", "transform_degrees", "transform_shear", "steps_per_epoch",
    "eval_batches", "common_init", "deterministic", "apollo_sf_quant_block_size",
    "apollo_sf_delta_refresh_window", "apollo_update_proj_gap",
)
# ``apollo_sf_delta_refresh`` is an explicit comparison axis for the delta-
# refresh experiment. It remains in each input protocol and report, but is
# intentionally allowed to differ between paired arms.
_METRICS = (
    "best_validation_top1", "test_top1", "test_loss",
    "train_samples_per_second", "peak_allocated_mb", "peak_reserved_mb",
    "persistent_state_mib", "sf_delta_commit_count",
    "sf_delta_commit_norm_sum", "sf_delta_commit_norm_max",
    "sf_delta_blend_count", "sf_delta_blend_step_count",
    "sf_delta_blend_norm_sum",
)
_EXPECTED_SEEDS = (47, 48, 49, 50)
_EXPECTED_VARIANTS = (
    "gated_gqa_silu_gated_ffn",
    "ada_gated_gqa_silu_gated_ffn",
    "ada_1plus_silu_gated_gqa_silu_gated_ffn",
    "ada_softplus1_norm_gated_gqa_silu_gated_ffn",
)


def _parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("inputs must use LABEL=SUMMARY_DIR")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("input labels and paths cannot be empty")
    return label.strip(), Path(raw_path).expanduser()


def _load_summary(label: str, path: Path) -> dict:
    summary_path = path / "comparison_summary.json" if path.is_dir() else path
    if not summary_path.is_file():
        raise FileNotFoundError(f"{label}: summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = summary.get("runs", [])
    if not runs:
        raise ValueError(f"{label}: no completed runs in {summary_path}")
    return {
        "label": label,
        "path": summary_path,
        "protocol": summary.get("protocol", {}),
        "runs": runs,
    }


def _validate_protocols(inputs: list[dict]) -> dict:
    reference = inputs[0]
    for current in inputs[1:]:
        mismatches = [
            key for key in _MATCHED_PROTOCOL_KEYS
            if reference["protocol"].get(key) != current["protocol"].get(key)
        ]
        if mismatches:
            raise ValueError(
                f"matched training protocol differs between {reference['label']} and "
                f"{current['label']}: {', '.join(mismatches)}"
            )
    return {
        key: reference["protocol"].get(key)
        for key in _MATCHED_PROTOCOL_KEYS
        if key in reference["protocol"]
    }


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def compare(
    inputs: list[dict],
    reference_label: str,
    *,
    expected_seeds: tuple[int, ...] = _EXPECTED_SEEDS,
    expected_variants: tuple[str, ...] = _EXPECTED_VARIANTS,
) -> dict:
    labels = [item["label"] for item in inputs]
    if len(set(labels)) != len(labels):
        raise ValueError("optimizer labels must be unique")
    if reference_label not in labels:
        raise ValueError(f"reference optimizer {reference_label!r} is not included")
    matched_protocol = _validate_protocols(inputs)
    indexed = {}
    for item in inputs:
        by_key = {}
        for run in item["runs"]:
            key = (run["variant"], int(run["seed"]))
            if key in by_key:
                raise ValueError(f"{item['label']}: duplicate variant/seed {key}")
            by_key[key] = run
        indexed[item["label"]] = by_key

    expected_keys = {
        (variant, seed)
        for variant in expected_variants
        for seed in expected_seeds
    }
    coverage = {}
    for label, by_key in indexed.items():
        actual_keys = set(by_key)
        missing_keys = sorted(expected_keys - actual_keys)
        unexpected_keys = sorted(actual_keys - expected_keys)
        coverage[label] = {
            "expected_runs": len(expected_keys),
            "completed_expected_runs": len(expected_keys & actual_keys),
            "complete": not missing_keys and not unexpected_keys,
            "missing_runs": [
                {"variant": variant, "seed": seed}
                for variant, seed in missing_keys
            ],
            "unexpected_runs": [
                {"variant": variant, "seed": seed}
                for variant, seed in unexpected_keys
            ],
        }

    ref_runs = indexed[reference_label]
    rows = []
    for item in inputs:
        by_key = indexed[item["label"]]
        variants = sorted({variant for variant, _ in by_key})
        for variant in variants:
            variant_runs = {
                seed: run for (run_variant, seed), run in by_key.items()
                if run_variant == variant
            }
            ref_variant_runs = {
                seed: run for (run_variant, seed), run in ref_runs.items()
                if run_variant == variant
            }
            common_seeds = sorted(set(variant_runs) & set(ref_variant_runs))
            row = {
                "optimizer": item["label"],
                "variant": variant,
                "seeds": sorted(variant_runs),
                "runs": len(variant_runs),
                "matched_seeds": common_seeds,
                "initial_state_mismatch_seeds": [
                    seed for seed in common_seeds
                    if variant_runs[seed].get("initial_state_sha256")
                    != ref_variant_runs[seed].get("initial_state_sha256")
                ],
            }
            backend_names = sorted({
                backend
                for seed in variant_runs
                for backend in variant_runs[seed].get(
                    "parameter_counts_by_backend", {},
                )
            })
            row["parameter_counts_by_backend"] = {
                backend: {
                    "n": len(values := [
                        variant_runs[seed]["parameter_counts_by_backend"][backend]
                        for seed in variant_runs
                        if backend in variant_runs[seed].get(
                            "parameter_counts_by_backend", {},
                        )
                    ]),
                    "mean": statistics.fmean(values),
                }
                for backend in backend_names
            }
            row["parameter_numel_by_backend"] = {
                backend: {
                    "n": len(values := [
                        variant_runs[seed]["parameter_numel_by_backend"][backend]
                        for seed in variant_runs
                        if backend in variant_runs[seed].get(
                            "parameter_numel_by_backend", {},
                        )
                    ]),
                    "mean": statistics.fmean(values),
                }
                for backend in sorted({
                    backend
                    for seed in variant_runs
                    for backend in variant_runs[seed].get(
                        "parameter_numel_by_backend", {},
                    )
                })
            }
            for metric in _METRICS:
                values = [
                    float(run[metric]) for run in variant_runs.values()
                    if run.get(metric) is not None
                ]
                mean, std = _mean_std(values)
                row[metric] = {"n": len(values), "mean": mean, "std": std}
            paired_deltas = [] if item["label"] == reference_label else [
                float(variant_runs[seed]["test_top1"])
                - float(ref_variant_runs[seed]["test_top1"])
                for seed in common_seeds
                if variant_runs[seed].get("test_top1") is not None
                and ref_variant_runs[seed].get("test_top1") is not None
            ]
            delta_mean, delta_std = _mean_std(paired_deltas)
            row["paired_test_delta_vs_reference"] = {
                "reference": reference_label,
                "n": len(paired_deltas),
                "mean": delta_mean,
                "std": delta_std,
                "values": {} if item["label"] == reference_label else {
                    str(seed): (
                        float(variant_runs[seed]["test_top1"])
                        - float(ref_variant_runs[seed]["test_top1"])
                    )
                    for seed in common_seeds
                    if variant_runs[seed].get("test_top1") is not None
                    and ref_variant_runs[seed].get("test_top1") is not None
                },
                "seeds_above_reference": sum(value > 0.0 for value in paired_deltas),
            }
            rows.append(row)

    return {
        "reference_optimizer": reference_label,
        "matched_protocol": matched_protocol,
        "expected_seeds": list(expected_seeds),
        "expected_variants": list(expected_variants),
        "coverage": coverage,
        "optimizer_protocols": {
            item["label"]: {
                "summary_path": str(item["path"]),
                "protocol": item["protocol"],
                "completed_runs": len(item["runs"]),
            }
            for item in inputs
        },
        "rows": rows,
    }


def _fmt_percent(stat: dict) -> str:
    if stat["mean"] is None:
        return "—"
    return f"{stat['mean'] * 100:.2f} ± {stat['std'] * 100:.2f}% (n={stat['n']})"


def _fmt_delta(stat: dict) -> str:
    if stat["mean"] is None:
        return "—"
    values = ", ".join(
        f"{seed}:{delta * 100:+.2f}" for seed, delta in stat["values"].items()
    )
    return (
        f"{stat['mean'] * 100:+.2f} ± {stat['std'] * 100:.2f} pp "
        f"({stat['seeds_above_reference']}/{stat['n']} seeds)"
        + (f"; {values}" if values else "")
    )


def _fmt_number(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}"


def write_report(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "optimizer_comparison.json"
    markdown_path = output_dir / "optimizer_comparison.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Mini-ImageNet optimizer comparison", "",
        f"Paired test deltas use **{report['reference_optimizer']}** as the reference.",
        "Optimizer schedule policies are recorded separately; this report compares the tested protocols, not optimizer equations in isolation.", "",
        "| Optimizer | Variant | Seeds | Test top-1 mean ± SD | Paired Δ vs reference | Best validation | Images/s | Peak allocated MiB | Optimizer state MiB |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['optimizer']} | `{row['variant']}` | "
            f"{', '.join(map(str, row['seeds']))} | "
            f"{_fmt_percent(row['test_top1'])} | "
            f"{_fmt_delta(row['paired_test_delta_vs_reference'])} | "
            f"{_fmt_percent(row['best_validation_top1'])} | "
            f"{_fmt_number(row['train_samples_per_second']['mean'])} | "
            f"{_fmt_number(row['peak_allocated_mb']['mean'])} | "
            f"{_fmt_number(row['persistent_state_mib']['mean'])} |"
        )
    lines += ["", "## Run coverage", ""]
    for label, coverage in report["coverage"].items():
        status = "complete" if coverage["complete"] else "INCOMPLETE"
        lines.append(
            f"- **{label}**: {coverage['completed_expected_runs']}/"
            f"{coverage['expected_runs']} expected runs ({status})."
        )
        if coverage["missing_runs"]:
            missing = ", ".join(
                f"{item['variant']} seed {item['seed']}"
                for item in coverage["missing_runs"]
            )
            lines.append(f"  - Missing: {missing}.")
        if coverage["unexpected_runs"]:
            unexpected = ", ".join(
                f"{item['variant']} seed {item['seed']}"
                for item in coverage["unexpected_runs"]
            )
            lines.append(f"  - Unexpected: {unexpected}.")
    lines += ["", "## Protocol notes", ""]
    for label, details in report["optimizer_protocols"].items():
        protocol = details["protocol"]
        schedule = protocol.get("lr_schedule_mode")
        if schedule is None and label == "AdamW":
            schedule = "epoch_cosine (legacy trainer default)"
        lines.append(
            f"- **{label}**: {details['completed_runs']} completed runs; "
            f"optimizer={protocol.get('optimizer', label)}, "
            f"schedule={schedule or 'unspecified'}, "
            f"LR scheduler={protocol.get('lr_scheduler', 'unspecified')}, "
            f"warmup ratio={protocol.get('warmup_ratio', 'unspecified')}, "
            f"APOLLO rank={protocol.get('apollo_rank', 'n/a')}, "
            f"delta refresh={protocol.get('apollo_sf_delta_refresh', 'n/a')}."
        )
    lines += ["", "## Parameter allocation by optimizer backend", ""]
    for row in report["rows"]:
        backends = sorted(set(row["parameter_counts_by_backend"]) | set(
            row["parameter_numel_by_backend"]
        ))
        allocations = []
        for backend in backends:
            tensors = row["parameter_counts_by_backend"].get(backend)
            elements = row["parameter_numel_by_backend"].get(backend)
            tensor_text = (
                f"{tensors['mean']:.1f} tensors (n={tensors['n']})"
                if tensors else "tensor count unavailable"
            )
            element_text = (
                f"{elements['mean']:.0f} elements (n={elements['n']})"
                if elements else "element count unavailable"
            )
            allocations.append(f"{backend}: {tensor_text}, {element_text}")
        lines.append(
            f"- **{row['optimizer']} / {row['variant']}**: "
            f"{', '.join(allocations) or 'not recorded'}."
        )
    lines += [
        "", "`initial_state_mismatch_seeds` in the JSON report flags comparisons where common model initialization hashes differ.",
        "Paired deltas are computed only on seeds present in both optimizer reports.", "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", action="append", type=_parse_input, required=True,
        metavar="LABEL=SUMMARY_DIR", help="Completed comparison summary directory; repeat per optimizer.",
    )
    parser.add_argument("--reference", default="AdamW", help="Label for paired test-score deltas.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expected-seeds", nargs="+", type=int, default=list(_EXPECTED_SEEDS),
        help="Expected seed IDs for coverage checks. Defaults to 47 48 49 50.",
    )
    parser.add_argument(
        "--expected-variants", nargs="+", default=list(_EXPECTED_VARIANTS),
        help="Expected variant names. Defaults to the established four-variant matrix.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    inputs = [_load_summary(label, path) for label, path in args.input]
    report = compare(
        inputs,
        args.reference,
        expected_seeds=tuple(args.expected_seeds),
        expected_variants=tuple(args.expected_variants),
    )
    paths = write_report(report, Path(args.output_dir))
    print(f"compared_optimizer_protocols={len(inputs)} rows={len(report['rows'])}")
    print(f"json={paths[0]}")
    print(f"markdown={paths[1]}")


if __name__ == "__main__":
    main()
