"""Summarize text-LM trajectory-curvature diagnostic output."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _metric_values(
    cases: list[dict[str, object]],
    section: str,
    field: str,
) -> list[float]:
    values: list[float] = []
    for case in cases:
        records = case.get(section, [])
        if not isinstance(records, list):
            continue
        for record in records:
            if isinstance(record, dict) and isinstance(record.get(field), (int, float)):
                values.append(float(record[field]))
    return values


def _loss_metric_values(
    cases: list[dict[str, object]],
    field: str,
    sequence: str,
) -> list[float]:
    values: list[float] = []
    for case in cases:
        loss_curvature = case.get("loss_curvature")
        if not isinstance(loss_curvature, dict):
            continue
        loss_metrics = loss_curvature.get(sequence)
        if isinstance(loss_metrics, dict) and isinstance(loss_metrics.get(field), (int, float)):
            values.append(float(loss_metrics[field]))
    return values


def _text(values: list[float], *, divisor: float = 1.0, digits: int = 4) -> str:
    if not values:
        return "-"
    mean, std = _mean_std(values)
    return f"{mean / divisor:.{digits}g} ± {std / divisor:.{digits}g}"


def build_report(path: Path) -> str:
    result = json.loads(path.read_text())
    grouped: dict[str, list[dict[str, object]]] = {}
    for case in result.get("cases", []):
        if isinstance(case, dict):
            grouped.setdefault(str(case.get("optimizer", "unknown")), []).append(case)

    lines = [
        "# Text LM optimizer trajectory diagnostics",
        "",
        f"Source: `{path}`",
        "",
        "Diagnostic runs include CPU copies and are not speed benchmarks.",
        "",
        "| optimizer | runs | validation loss mean±std | state MiB mean±std | step ms mean±std | turning angle | direction change | normalized roughness | train loss Δ² abs mean | validation loss Δ² abs mean | status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for optimizer, cases in sorted(grouped.items()):
        losses = [float(case["final_validation_loss"]) for case in cases]
        states = [float(case["persistent_state_bytes"]) for case in cases]
        steps = [float(case["host_seconds_per_optimizer_step"]) * 1000.0 for case in cases]
        statuses = {str(case.get("status", "unknown")) for case in cases}
        status = "passed" if statuses == {"passed"} else ", ".join(sorted(statuses))
        lines.append(
            f"| {optimizer} | {len(cases)} | {_text(losses, digits=5)} "
            f"| {_text(states, divisor=2**20, digits=5)} | {_text(steps, digits=5)} "
            f"| {_text(_metric_values(cases, 'trajectory_curvature', 'turning_angle_mean'))} "
            f"| {_text(_metric_values(cases, 'trajectory_curvature', 'direction_change_mean'))} "
            f"| {_text(_metric_values(cases, 'trajectory_curvature', 'normalized_roughness'))} "
            f"| {_text(_loss_metric_values(cases, 'second_difference_abs_mean', 'train_step'))} "
            f"| {_text(_loss_metric_values(cases, 'second_difference_abs_mean', 'validation_step'))} | {status} |"
        )
    if not grouped:
        lines.extend(["", "No cases found."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    print(build_report(args.result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
