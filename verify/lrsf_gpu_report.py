"""Aggregate LRSF GPU validation JSON files by optimizer and refresh mode."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
import sys
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="+",
        type=Path,
        help="JSON files or directories containing lrsf-gpu-*.json files",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format. Default: markdown.",
    )
    return parser.parse_args(argv)


def _number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.{digits}g}"
    return str(value)


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _event_recovery_mean(case: dict[str, Any]) -> float | None:
    values: list[float] = []
    for key in ("refresh_events", "orthogonal_refresh_events"):
        events = case.get(key)
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            value = _as_number(event.get("recovery_steps_to_pre_refresh_loss"))
            if value is not None:
                values.append(value)
    mean, _ = _mean_std(values)
    return mean


def _event_metric_mean(case: dict[str, Any], key: str) -> float | None:
    values: list[float] = []
    events = case.get("orthogonal_refresh_events")
    if not isinstance(events, list):
        return None
    for event in events:
        if isinstance(event, dict):
            value = _as_number(event.get(key))
            if value is not None:
                values.append(value)
    mean, _ = _mean_std(values)
    return mean


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, sqrt(variance)


def _refresh_label(result: dict[str, Any], case: dict[str, Any]) -> str:
    refresh = result.get("refresh")
    if not isinstance(refresh, dict):
        refresh = {}
    mode = case.get("refresh_mode", refresh.get("mode"))
    mix = case.get("projection_refresh_mix", refresh.get("mix"))
    orthogonal_rate = case.get(
        "orthogonal_refresh_rate", refresh.get("orthogonal_rate", 0.0)
    )
    direction = case.get(
        "orthogonal_refresh_direction",
        refresh.get("orthogonal_direction", "random"),
    )
    signal = case.get(
        "orthogonal_refresh_signal",
        refresh.get("orthogonal_signal", "gradient"),
    )
    if mode == "none" and orthogonal_rate not in (None, 0, 0.0):
        parts = [f"orthogonal({orthogonal_rate:g})"]
        if direction != "random":
            parts.append(str(direction))
        if signal != "gradient":
            parts.append(str(signal))
        return "/".join(parts)
    if mode == "none":
        return "fixed/frozen"
    if mode == "smooth":
        return f"smooth/{mix}" if mix is not None else "smooth"
    return str(mode) if mode is not None else "-"


def _expand_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("lrsf-gpu-*.json")))
        else:
            expanded.append(path)
    unique = sorted(set(expanded))
    if not unique:
        raise FileNotFoundError("no lrsf-gpu-*.json files found")
    return unique


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _expand_paths(paths):
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist")
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        if not isinstance(result, dict) or not isinstance(result.get("cases"), dict):
            raise ValueError(f"{path}: expected a top-level 'cases' object")
        seed = result.get("seed")
        for label, raw_case in result["cases"].items():
            if not isinstance(raw_case, dict):
                raise ValueError(f"{path}: case {label!r} must be an object")
            rows.append({
                "file": path.name,
                "seed": seed,
                "optimizer": raw_case.get("optimizer", label),
                "rank": raw_case.get("rank"),
                "refresh": _refresh_label(result, raw_case),
                "status": raw_case.get("status", result.get("status", "-")),
                "validation_loss": _as_number(raw_case.get("final_validation_loss")),
                "state_bytes": _as_number(raw_case.get("persistent_state_bytes")),
                "peak_state_bytes": _as_number(
                    raw_case.get("peak_persistent_state_bytes")
                ),
                "peak_allocated_bytes": _as_number(
                    raw_case.get("peak_allocated_bytes")
                ),
                "peak_reserved_bytes": _as_number(
                    raw_case.get("peak_reserved_bytes")
                ),
                "step_seconds": _as_number(
                    raw_case.get("host_seconds_per_optimizer_step")
                ),
                "orthogonal_refresh_steps": _as_number(
                    raw_case.get("orthogonal_refresh_steps")
                ),
                "update_norm_mean": _as_number(raw_case.get("update_norm_mean")),
                "update_norm_variance": _as_number(
                    raw_case.get("update_norm_variance")
                ),
                "recovery_steps_mean": _event_recovery_mean(raw_case),
                "loss_lowering_proxy_decrease": _event_metric_mean(
                    raw_case, "loss_lowering_proxy_decrease",
                ),
            })
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fixed_losses = {
        (row["optimizer"], row["rank"], row["seed"]): row["validation_loss"]
        for row in rows
        if row["refresh"] == "fixed/frozen"
        and row["validation_loss"] is not None
    }
    for row in rows:
        baseline = fixed_losses.get(
            (row["optimizer"], row["rank"], row["seed"])
        )
        row["paired_loss_delta"] = (
            None
            if baseline is None or row["refresh"] == "fixed/frozen"
            or row["validation_loss"] is None
            else row["validation_loss"] - baseline
        )

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["optimizer"], row["rank"], row["refresh"])
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict[str, Any]] = []
    for (optimizer, rank, refresh), group in sorted(
        grouped.items(), key=lambda item: (str(item[0][0]), item[0][1] is None, item[0][1] or 0, item[0][2])
    ):
        values: dict[str, Any] = {
            "optimizer": optimizer,
            "rank": rank,
            "refresh": refresh,
            "runs": len(group),
            "seeds": sorted(
                row["seed"] for row in group if isinstance(row["seed"], int)
            ),
            "status": "passed" if all(row["status"] == "passed" for row in group) else "mixed",
        }
        for key in (
            "validation_loss",
            "paired_loss_delta",
            "state_bytes",
            "peak_state_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "step_seconds",
            "orthogonal_refresh_steps",
            "update_norm_mean",
            "update_norm_variance",
            "recovery_steps_mean",
            "loss_lowering_proxy_decrease",
        ):
            mean, std = _mean_std(
                [row[key] for row in group if row[key] is not None]
            )
            values[f"{key}_mean"] = mean
            values[f"{key}_std"] = std
        paired_deltas = [
            row["paired_loss_delta"]
            for row in group
            if row["paired_loss_delta"] is not None
        ]
        values["paired_improved_count"] = sum(
            delta < 0.0 for delta in paired_deltas
        )
        values["paired_comparison_count"] = len(paired_deltas)
        aggregated.append(values)
    return aggregated


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| optimizer | rank | refresh | runs | validation loss mean±std | paired loss Δ vs fixed | improved seeds | state bytes mean±std | peak state bytes | peak allocated bytes | peak reserved bytes | step seconds mean±std | orthogonal steps | update norm mean | update norm variance | recovery steps | loss-lowering proxy decrease | status |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        loss = f"{_number(row['validation_loss_mean'])} ± {_number(row['validation_loss_std'])}"
        state = f"{_number(row['state_bytes_mean'], 4)} ± {_number(row['state_bytes_std'], 4)}"
        step = f"{_number(row['step_seconds_mean'])} ± {_number(row['step_seconds_std'])}"
        paired_loss = (
            f"{_number(row['paired_loss_delta_mean'])} ± "
            f"{_number(row['paired_loss_delta_std'])}"
            if row["paired_loss_delta_mean"] is not None
            else "-"
        )
        improved = (
            f"{row['paired_improved_count']}/{row['paired_comparison_count']}"
            if row["paired_comparison_count"]
            else "-"
        )
        lines.append(
            f"| {row['optimizer']} | {_number(row['rank'], 4)} | {row['refresh']} | "
            f"{row['runs']} | {loss} | {paired_loss} | {improved} | {state} | "
            f"{_number(row['peak_state_bytes_mean'], 4)} | "
            f"{_number(row['peak_allocated_bytes_mean'], 4)} | "
            f"{_number(row['peak_reserved_bytes_mean'], 4)} | {step} | "
            f"{_number(row['orthogonal_refresh_steps_mean'], 4)} | "
            f"{_number(row['update_norm_mean_mean'], 6)} | "
            f"{_number(row['update_norm_variance_mean'], 6)} | "
            f"{_number(row['recovery_steps_mean_mean'], 4)} | "
            f"{_number(row['loss_lowering_proxy_decrease_mean'], 6)} | "
            f"{row['status']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = aggregate_rows(load_rows(args.results))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
