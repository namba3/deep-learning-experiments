"""Summarize paired APOLLO/APOLLO-Conf confidence sensitivity results."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot summarize an empty list")
    return statistics.fmean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


def _decode(value: str) -> str:
    return value.replace("p", ".")


def build_report(root: Path) -> str:
    rows: list[dict[str, object]] = []
    pattern = re.compile(
        r"confidence-beta-(?P<beta>.+)-alpha-(?P<alpha>.+)"
    )
    for path in sorted(root.glob("confidence-beta-*-alpha-*/result.json")):
        match = pattern.fullmatch(path.parent.name)
        if match is None:
            continue
        result = json.loads(path.read_text())
        grouped: dict[str, list[dict[str, object]]] = {}
        for case in result.get("cases", []):
            grouped.setdefault(str(case["optimizer"]), []).append(case)
        if not grouped:
            continue
        baseline = grouped.get("APOLLO", [])
        baseline_losses = [float(c["final_validation_loss"]) for c in baseline]
        baseline_steps = [
            float(c["host_seconds_per_optimizer_step"]) for c in baseline
        ]
        for optimizer, cases in sorted(grouped.items()):
            losses = [float(c["final_validation_loss"]) for c in cases]
            states = [float(c["persistent_state_bytes"]) for c in cases]
            steps = [
                float(c["host_seconds_per_optimizer_step"]) * 1000.0
                for c in cases
            ]
            loss_mean, loss_std = _mean_std(losses)
            state_mean, state_std = _mean_std(states)
            step_mean, step_std = _mean_std(steps)
            if optimizer == "APOLLO" or not baseline_losses:
                delta_text = "-"
                improved_text = "-"
            else:
                deltas = [
                    loss - base
                    for loss, base in zip(losses, baseline_losses)
                ]
                delta_mean, delta_std = _mean_std(deltas)
                delta_text = f"{delta_mean:.6g} ± {delta_std:.4g}"
                improved_text = (
                    f"{sum(delta < 0.0 for delta in deltas)}/{len(deltas)}"
                )
            if optimizer == "APOLLO" or not baseline_steps:
                step_delta_text = "-"
            else:
                step_deltas = [
                    step - base * 1000.0
                    for step, base in zip(steps, baseline_steps)
                ]
                step_delta_mean, step_delta_std = _mean_std(step_deltas)
                step_delta_text = (
                    f"{step_delta_mean:.3f} ± {step_delta_std:.3f} ms"
                )
            rows.append({
                "beta": _decode(match.group("beta")),
                "alpha": _decode(match.group("alpha")),
                "optimizer": optimizer,
                "runs": len(cases),
                "loss": f"{loss_mean:.6g} ± {loss_std:.4g}",
                "loss_delta": delta_text,
                "improved": improved_text,
                "state": f"{state_mean / 2**20:.3f} ± {state_std / 2**20:.3f} MiB",
                "step": f"{step_mean:.3f} ± {step_std:.3f} ms",
                "step_delta": step_delta_text,
                "status": result.get("status", "unknown"),
            })

    rows.sort(key=lambda row: (
        float(str(row["beta"])), float(str(row["alpha"])),
        str(row["optimizer"]),
    ))
    lines = [
        "# Text LM APOLLO confidence sensitivity",
        "",
        f"Source: `{root}`",
        "",
        "| confidence beta | alpha | optimizer | runs | validation loss mean±std | paired loss Δ vs APOLLO | improved seeds | state MiB mean±std | step ms mean±std | paired step Δ | status |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['beta']} | {row['alpha']} | {row['optimizer']} "
            f"| {row['runs']} | {row['loss']} | {row['loss_delta']} "
            f"| {row['improved']} | {row['state']} | {row['step']} "
            f"| {row['step_delta']} | {row['status']} |"
        )
    if not rows:
        lines.extend(["", "No completed confidence cells found."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    print(build_report(args.sweep_dir), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
