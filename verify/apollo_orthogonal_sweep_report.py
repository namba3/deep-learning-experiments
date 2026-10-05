"""Render an APOLLO orthogonal-rate sweep as a Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


_ORTHOGONAL_NAME = re.compile(r"apollo-refresh-orthogonal-seed(?P<seed>[0-9]+)\.json$")


def _case_by_optimizer(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(case["optimizer"]): case
        for case in data.get("cases", {}).values()
    }


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for orthogonal_path in sorted(root.rglob("apollo-refresh-orthogonal-seed*.json")):
        match = _ORTHOGONAL_NAME.fullmatch(orthogonal_path.name)
        if match is None:
            continue
        none_path = orthogonal_path.with_name(
            orthogonal_path.name.replace(
                "apollo-refresh-orthogonal-", "apollo-refresh-none-", 1,
            )
        )
        if not none_path.is_file():
            raise FileNotFoundError(f"matching none result not found: {none_path}")

        relative_parts = orthogonal_path.relative_to(root).parts
        if len(relative_parts) < 4:
            raise ValueError(
                "sweep result must be under <direction>/<rate>/seed-<N>"
            )
        direction, rate_dir = relative_parts[0], relative_parts[1]
        if not rate_dir.startswith("rate-"):
            raise ValueError(f"invalid sweep rate directory: {rate_dir}")
        rate = rate_dir.removeprefix("rate-").replace("p", ".")
        none_cases = _case_by_optimizer(none_path)
        orthogonal_cases = _case_by_optimizer(orthogonal_path)
        for optimizer in ("APOLLO", "APOLLO-CAME", "APOLLO-Mini"):
            baseline = none_cases[optimizer]
            candidate = orthogonal_cases[optimizer]
            baseline_step = float(baseline["host_seconds_per_optimizer_step"])
            candidate_step = float(candidate["host_seconds_per_optimizer_step"])
            baseline_variance = float(baseline.get("update_norm_variance", 0.0))
            candidate_variance = float(candidate.get("update_norm_variance", 0.0))
            rows.append({
                "direction": direction,
                "rate": rate,
                "seed": int(match.group("seed")),
                "optimizer": optimizer,
                "none_loss": float(baseline["final_validation_loss"]),
                "orthogonal_loss": float(candidate["final_validation_loss"]),
                "loss_delta": float(candidate["final_validation_loss"])
                - float(baseline["final_validation_loss"]),
                "step_ratio": candidate_step / baseline_step,
                "variance_ratio": (
                    candidate_variance / baseline_variance
                    if baseline_variance > 0.0 else None
                ),
                "orthogonal_steps": int(candidate.get("orthogonal_refresh_steps", 0)),
                "status": "passed" if (
                    baseline.get("status") == "passed"
                    and candidate.get("status") == "passed"
                ) else "error",
            })
    if not rows:
        raise ValueError(f"no orthogonal sweep results found below {root}")
    return rows


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.7g}"


def render(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| direction | rate | seed | optimizer | none loss | orthogonal loss | delta | step ratio | variance ratio | orthogonal steps | status |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {direction} | {rate} | {seed} | {optimizer} | {none_loss} | "
            "{orthogonal_loss} | {loss_delta} | {step_ratio} | {variance_ratio} | "
            "{orthogonal_steps} | {status} |".format(
                direction=row["direction"],
                rate=row["rate"],
                seed=row["seed"],
                optimizer=row["optimizer"],
                none_loss=_number(row["none_loss"]),
                orthogonal_loss=_number(row["orthogonal_loss"]),
                loss_delta=_number(row["loss_delta"]),
                step_ratio=_number(row["step_ratio"]),
                variance_ratio=_number(row["variance_ratio"]),
                orthogonal_steps=row["orthogonal_steps"],
                status=row["status"],
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    print(render(load_rows(args.root)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
