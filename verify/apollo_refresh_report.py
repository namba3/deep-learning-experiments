"""Render comparable rows from APOLLO refresh experiment JSON files.

This is a reporting helper, not an experiment runner. It accepts the JSON
emitted by ``verify.image_ae_cifar10_adamw_apollo`` or ``verify.optimizers``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument(
        "--format", choices=("markdown", "json"), default="markdown",
        help="Output format. Default: markdown.",
    )
    return parser.parse_args(argv)


def _first(case: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = case.get(key)
        if value is not None:
            return value
    return None


def _number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.{digits}g}"
    return str(value)


def _refresh_label(case: dict[str, Any]) -> str:
    if case.get("projection_mode") == "frozen":
        mode = "frozen"
        orthogonal_rate = case.get("orthogonal_refresh_rate")
        if orthogonal_rate not in (None, 0, 0.0):
            return f"{mode}/orth={_number(orthogonal_rate)}"
        return mode
    mode = case.get("projection_refresh_mode")
    mix = case.get("projection_refresh_mix")
    state = case.get("projection_refresh_state")
    orthogonal_rate = case.get("orthogonal_refresh_rate")
    parts = []
    if mode is not None:
        parts.append(str(mode))
    if mode == "smooth" and mix is not None:
        parts.append(str(mix))
    if state is not None and mode in {"hard", "smooth"}:
        parts.append(str(state))
    if orthogonal_rate not in (None, 0, 0.0):
        parts.append(f"orth={_number(orthogonal_rate)}")
    return "/".join(parts) if parts else "-"


def rows_from_result(path: Path, result: dict[str, Any]) -> list[dict[str, Any]]:
    cases = result.get("cases")
    if not isinstance(cases, dict):
        raise ValueError(f"{path}: expected a top-level 'cases' object")
    rows = []
    for label, raw_case in cases.items():
        if not isinstance(raw_case, dict):
            raise ValueError(f"{path}: case {label!r} must be an object")
        refresh_events = raw_case.get("projection_refresh_events")
        refresh_steps = raw_case.get("projection_refresh_steps")
        if not isinstance(refresh_events, list):
            refresh_events = []
        if refresh_steps is None:
            refresh_steps = len(refresh_events)
        rows.append({
            "file": path.name,
            "case": label,
            "status": raw_case.get("status", result.get("status", "-")),
            "optimizer": raw_case.get("optimizer", label),
            "rank": raw_case.get("rank"),
            "refresh": _refresh_label(raw_case),
            "validation_loss": raw_case.get("final_validation_loss"),
            "state_bytes": raw_case.get("persistent_state_bytes"),
            "step_seconds": _first(
                raw_case,
                "cuda_seconds_per_step",
                "host_seconds_per_optimizer_step",
                "host_seconds_per_step",
            ),
            "peak_allocated_bytes": raw_case.get("peak_allocated_bytes"),
            "peak_reserved_bytes": raw_case.get("peak_reserved_bytes"),
            "refresh_steps": refresh_steps,
            "active_steps": raw_case.get("projection_refresh_active_steps"),
            "orthogonal_steps": raw_case.get("orthogonal_refresh_steps"),
            "update_norm_mean": raw_case.get("update_norm_mean"),
            "update_norm_variance": raw_case.get("update_norm_variance"),
            "update_norm_variance_cap": raw_case.get(
                "update_norm_variance_cap"
            ),
            "update_norm_variance_capped_steps": raw_case.get(
                "update_norm_variance_capped_steps", 0
            ),
            "projection_change_max_abs": max(
                (
                    float(event["projection_change_max_abs"])
                    for event in refresh_events
                    if isinstance(event, dict)
                    and event.get("projection_change_max_abs") is not None
                ),
                default=None,
            ),
        })
    return rows


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} does not exist; run the corresponding probe first"
            )
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        if not isinstance(result, dict):
            raise ValueError(f"{path}: top-level JSON must be an object")
        rows.extend(rows_from_result(path, result))
    return rows


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| file | case | optimizer | rank | refresh | validation loss | state bytes | step s | peak allocated | peak reserved | refresh events | active steps | orthogonal steps | projection change | update norm mean | update norm variance | variance cap | capped steps | status |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {file} | {case} | {optimizer} | {rank} | {refresh} | {validation_loss} | "
            "{state_bytes} | {step_seconds} | {peak_allocated_bytes} | "
            "{peak_reserved_bytes} | {refresh_steps} | {active_steps} | "
                "{orthogonal_steps} | {projection_change_max_abs} | {update_norm_mean} | "
                "{update_norm_variance} | {update_norm_variance_cap} | "
                "{update_norm_variance_capped_steps} | {status} |".format(
                file=row["file"],
                case=row["case"],
                optimizer=row["optimizer"],
                rank=_number(row["rank"], 4),
                refresh=row["refresh"],
                validation_loss=_number(row["validation_loss"]),
                state_bytes=_number(row["state_bytes"], 4),
                step_seconds=_number(row["step_seconds"]),
                peak_allocated_bytes=_number(row["peak_allocated_bytes"], 4),
                peak_reserved_bytes=_number(row["peak_reserved_bytes"], 4),
                refresh_steps=_number(row["refresh_steps"], 4),
                active_steps=_number(row["active_steps"], 4),
                orthogonal_steps=_number(row["orthogonal_steps"], 4),
                projection_change_max_abs=_number(
                    row["projection_change_max_abs"]
                ),
                update_norm_mean=_number(row["update_norm_mean"]),
                update_norm_variance=_number(row["update_norm_variance"]),
                update_norm_variance_cap=_number(
                    row["update_norm_variance_cap"]
                ),
                update_norm_variance_capped_steps=_number(
                    row["update_norm_variance_capped_steps"], 4
                ),
                status=row["status"],
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = load_rows(args.results)
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
