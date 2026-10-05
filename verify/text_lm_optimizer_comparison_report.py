"""Summarize grouped text-LM optimizer comparison results."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot summarize an empty list")
    return statistics.fmean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


def build_report(root: Path) -> str:
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("group-*/result.json")):
        result = json.loads(path.read_text())
        cases = result.get("cases", [])
        grouped: dict[str, list[dict[str, object]]] = {}
        for case in cases:
            grouped.setdefault(str(case["optimizer"]), []).append(case)
        for optimizer, optimizer_cases in sorted(grouped.items()):
            losses = [
                float(case["final_validation_loss"])
                for case in optimizer_cases
            ]
            states = [
                float(case["persistent_state_bytes"])
                for case in optimizer_cases
            ]
            peak_allocated = [
                float(case["peak_allocated_bytes"])
                for case in optimizer_cases
                if case.get("peak_allocated_bytes") is not None
            ]
            peak_reserved = [
                float(case["peak_reserved_bytes"])
                for case in optimizer_cases
                if case.get("peak_reserved_bytes") is not None
            ]
            steps = [
                float(case["host_seconds_per_optimizer_step"]) * 1000.0
                for case in optimizer_cases
            ]
            loss_mean, loss_std = _mean_std(losses)
            state_mean, state_std = _mean_std(states)
            step_mean, step_std = _mean_std(steps)

            def memory_text(values: list[float]) -> str:
                if not values:
                    return "not recorded"
                mean, std = _mean_std(values)
                return f"{mean / 2**30:.3f} ± {std / 2**30:.3f} GiB"

            rows.append({
                "group": path.parent.name.removeprefix("group-"),
                "optimizer": optimizer,
                "runs": len(optimizer_cases),
                "loss": f"{loss_mean:.6g} ± {loss_std:.4g}",
                "state": f"{state_mean / 2**20:.3f} ± {state_std / 2**20:.3f} MiB",
                "peak_allocated": memory_text(peak_allocated),
                "peak_reserved": memory_text(peak_reserved),
                "step": f"{step_mean:.3f} ± {step_std:.3f} ms",
                "status": result.get("status", "unknown"),
            })

    lines = [
        "# Text LM optimizer comparison",
        "",
        f"Source: `{root}`",
        "",
        "| group | optimizer | runs | validation loss mean±std | state MiB mean±std | peak allocated | peak reserved | step ms mean±std | status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['optimizer']} | {row['runs']} "
            f"| {row['loss']} | {row['state']} | {row['peak_allocated']} "
            f"| {row['peak_reserved']} | {row['step']} | {row['status']} |"
        )
    if not rows:
        lines.extend(["", "No completed result groups found."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_dir", type=Path)
    args = parser.parse_args()
    print(build_report(args.comparison_dir), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
