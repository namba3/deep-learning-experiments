"""Summarize offline residual approximation metrics from text-LM probes."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.{digits}g}"
    return str(value)


def _result_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        else:
            files.append(path)
    unique = sorted(set(files))
    if not unique:
        raise FileNotFoundError("no JSON result files found")
    return unique


def _collect(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    for path in _result_files(paths):
        result = json.loads(path.read_text(encoding="utf-8"))
        cases = result.get("cases")
        if not isinstance(cases, list):
            continue
        for case in cases:
            if not isinstance(case, dict):
                continue
            optimizer = str(case.get("optimizer", "-"))
            loss = case.get("final_validation_loss")
            if isinstance(loss, (int, float)) and not isinstance(loss, bool):
                losses.append({
                    "optimizer": optimizer,
                    "seed": case.get("seed"),
                    "loss": float(loss),
                })
            trajectory_items = case.get("update_trajectory_pca")
            if not isinstance(trajectory_items, list):
                continue
            for trajectory in trajectory_items:
                if not isinstance(trajectory, dict):
                    continue
                for basis_mode in ("rolling", "fixed_basis"):
                    basis_metrics = trajectory.get(basis_mode)
                    if not isinstance(basis_metrics, dict):
                        continue
                    block_size = basis_metrics.get(
                        "residual_compression_block_size"
                    )
                    scale_mode = basis_metrics.get(
                        "residual_compression_scale_mode"
                    )
                    temporal_rank_data = basis_metrics.get(
                        "residual_approximation"
                    )
                    if not isinstance(temporal_rank_data, dict):
                        continue
                    for temporal_rank, temporal_metrics in temporal_rank_data.items():
                        if not isinstance(temporal_metrics, dict):
                            continue
                        for method, method_metrics in (
                            ("low_rank_factor", temporal_metrics.get("low_rank_factor")),
                            ("blockwise_int8", temporal_metrics.get("blockwise_int8")),
                            ("blockwise_int4", temporal_metrics.get("blockwise_int4")),
                            (
                                "blockwise_int4_error_feedback",
                                temporal_metrics.get("blockwise_int4_error_feedback"),
                            ),
                            (
                                "low_rank_plus_int8",
                                temporal_metrics.get("low_rank_plus_int8"),
                            ),
                        ):
                            if method in ("low_rank_factor", "low_rank_plus_int8"):
                                if not isinstance(method_metrics, dict):
                                    continue
                                method_items = method_metrics.items()
                            else:
                                method_items = (("-", method_metrics),)
                            for spatial_rank, metrics in method_items:
                                if not isinstance(metrics, dict):
                                    continue
                                rows.append({
                                    "optimizer": optimizer,
                                    "basis_mode": basis_mode,
                                    "seed": case.get("seed"),
                                    "parameter": trajectory.get("parameter", "-"),
                                    "temporal_rank": str(temporal_rank),
                                    "method": method,
                                    "spatial_rank": str(spatial_rank),
                                    "trajectory_samples": temporal_metrics.get(
                                        "samples"
                                    ),
                                    "block_size": (
                                        metrics.get("block_size", block_size)
                                        if method != "low_rank_factor"
                                        else None
                                    ),
                                    "scale_mode": (
                                        metrics.get("scale_mode", scale_mode)
                                        if method != "low_rank_factor"
                                        else None
                                    ),
                                    "storage_ratio": metrics.get(
                                        "storage_ratio_to_target_bf16"
                                    ),
                                    "feedback_storage_ratio": metrics.get(
                                        "feedback_storage_ratio_to_target_bf16"
                                    ),
                                    "update_cosine": metrics.get("update_cosine"),
                                    "update_norm_ratio": metrics.get("update_norm_ratio"),
                                    "decode_ms": metrics.get("decode_milliseconds"),
                                })
    return rows, losses


def build_report(*paths: Path) -> str:
    rows, losses = _collect(list(paths))
    grouped: dict[tuple[str, str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["optimizer"], row["basis_mode"], row["temporal_rank"],
            row["method"], row["spatial_rank"],
            "-" if row["block_size"] is None else str(row["block_size"]),
            "-" if row["scale_mode"] is None else str(row["scale_mode"]),
        )
        grouped.setdefault(key, []).append(row)

    loss_by_optimizer: dict[str, list[float]] = {}
    for item in losses:
        loss_by_optimizer.setdefault(item["optimizer"], []).append(item["loss"])

    lines = [
        "# Text LM residual approximation report",
        "",
        "Offline reconstruction only; optimizer updates and persistent state are unchanged.",
        "Temporal error-feedback requires at least two trajectory predictions per case;",
        "a row with one trajectory sample cannot show temporal accumulation.",
        "",
        "| optimizer | basis | temporal rank | method | spatial rank | block size | scale mode | cases | trajectory samples | validation loss mean | storage / target BF16 | feedback / target BF16 | update cosine | update norm ratio | decode ms |",
        "| --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(
        grouped,
        key=lambda item: (
            item[0], item[1], int(item[2]), item[3],
            -1 if item[4] == "-" else int(item[4]),
            -1 if item[5] == "-" else int(item[5]),
            item[6],
        ),
    ):
        optimizer, basis_mode, temporal_rank, method, spatial_rank, block_size, scale_mode = key
        items = grouped[key]

        def numeric(name: str) -> float | None:
            values = [
                float(item[name]) for item in items
                if isinstance(item.get(name), (int, float))
                and not isinstance(item.get(name), bool)
            ]
            return _mean(values)

        lines.append(
            f"| {optimizer} | {basis_mode} | {temporal_rank} | {method} | {spatial_rank} "
            f"| {block_size} | {scale_mode} | {len(items)} "
            f"| {_number(numeric('trajectory_samples'))} "
            f"| {_number(_mean(loss_by_optimizer.get(optimizer, [])))} "
            f"| {_number(numeric('storage_ratio'))} "
            f"| {_number(numeric('feedback_storage_ratio'))} "
            f"| {_number(numeric('update_cosine'))} "
            f"| {_number(numeric('update_norm_ratio'))} | {_number(numeric('decode_ms'))} |"
        )
    if not grouped:
        lines.extend(["", "No residual approximation metrics found."])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", nargs="+", type=Path,
        help="JSON result files or directories containing result JSON files",
    )
    args = parser.parse_args(argv)
    print(build_report(*args.results), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
