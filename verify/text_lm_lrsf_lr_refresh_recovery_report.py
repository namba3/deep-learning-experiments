"""Summarize AdamW-LRSF-LR refresh-recovery diagnostics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _text(values: list[float], *, divisor: float = 1.0, digits: int = 4) -> str:
    if not values:
        return "-"
    mean, std = _mean_std(values)
    return f"{mean / divisor:.{digits}g} ± {std / divisor:.{digits}g}"


def _recovery_deltas(case: dict[str, object]) -> list[float]:
    """Return pre-refresh to first post-refresh validation deltas.

    This is a coarse recovery proxy: validation is usually measured less
    frequently than refresh events, so it includes ordinary training progress.
    Negative values mean the first post-refresh validation loss is lower.
    """
    records = case.get("validation_step_loss_records", [])
    moments = case.get("lrsf_latent_moment_history", [])
    if not isinstance(records, list) or not isinstance(moments, list):
        return []
    validation = [
        (int(record["step"]), float(record["validation_loss"]))
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("step"), int)
        and isinstance(record.get("validation_loss"), (int, float))
    ]
    events = [
        int(record["step"])
        for record in moments
        if isinstance(record, dict)
        and record.get("refresh_event") is True
        and isinstance(record.get("step"), int)
    ]
    deltas: list[float] = []
    for event_step in events:
        before = [loss for step, loss in validation if step < event_step]
        after = [loss for step, loss in validation if step >= event_step]
        if before and after:
            deltas.append(after[0] - before[-1])
    return deltas


def _cases(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text())
    return [case for case in data.get("cases", []) if isinstance(case, dict)]


def build_report(directory: Path) -> str:
    paths = {
        name: directory / f"{name}.json"
        for name in ("frozen", "hard-reset", "hard-transport")
    }
    lines = [
        "# AdamW-LRSF-LR refresh recovery",
        "",
        f"Source: `{directory}`",
        "",
        "Diagnostic runs include CPU copies and are not speed benchmarks.",
        "The recovery proxy is the first post-refresh validation loss minus the last pre-refresh validation loss; negative is lower loss.",
        "",
        "| policy | runs | final validation loss mean±std | recovery proxy mean±std | refresh events mean | moment age at refresh | state MiB | peak allocated GiB | peak reserved GiB | status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    found = False
    for policy, path in paths.items():
        if not path.exists():
            continue
        found = True
        cases = _cases(path)
        losses = [float(case["final_validation_loss"]) for case in cases]
        recovery = [value for case in cases for value in _recovery_deltas(case)]
        event_counts = [
            sum(
                record.get("refresh_event") is True
                for record in case.get("lrsf_latent_moment_history", [])
                if isinstance(record, dict)
            )
            for case in cases
        ]
        event_ages = [
            float(record["moment_step_min"])
            for case in cases
            for record in case.get("lrsf_latent_moment_history", [])
            if isinstance(record, dict)
            and record.get("refresh_event") is True
            and isinstance(record.get("moment_step_min"), (int, float))
        ]
        states = [float(case["persistent_state_bytes"]) for case in cases]
        allocated = [float(case["peak_allocated_bytes"]) for case in cases]
        reserved = [float(case["peak_reserved_bytes"]) for case in cases]
        statuses = {str(case.get("status", "unknown")) for case in cases}
        status = "passed" if statuses == {"passed"} else ", ".join(sorted(statuses))
        lines.append(
            f"| {policy} | {len(cases)} | {_text(losses, digits=5)} "
            f"| {_text(recovery, digits=5)} | {_text(event_counts, digits=4)} "
            f"| {_text(event_ages, digits=4)} | {_text(states, divisor=2**20, digits=5)} "
            f"| {_text(allocated, divisor=2**30, digits=5)} "
            f"| {_text(reserved, divisor=2**30, digits=5)} | {status} |"
        )
    if not found:
        lines.extend(["", "No policy JSON files found."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(build_report(args.directory), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
