"""Render an APOLLO update-norm variance-cap sweep as Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


_CAP_NAME = re.compile(r"cap-(?P<cap>none|[0-9]+p[0-9]+)$")
_SEED_NAME = re.compile(r"seed-(?P<seed>[0-9]+)$")


def _seed_cell_sort_key(path: Path) -> tuple[int, float, int]:
    cap_match = _CAP_NAME.fullmatch(path.parent.name)
    seed_match = _SEED_NAME.fullmatch(path.name)
    if cap_match is None or seed_match is None:
        return (2, float("inf"), 0)
    cap_slug = cap_match.group("cap")
    cap_value = float("inf") if cap_slug == "none" else float(
        cap_slug.replace("p", ".")
    )
    return (
        0 if cap_slug == "none" else 1,
        cap_value,
        int(seed_match.group("seed")),
    )


def _case_by_optimizer(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(case["optimizer"]): case
        for case in data.get("cases", {}).values()
        if case.get("optimizer") in {"APOLLO", "APOLLO-CAME", "APOLLO-Mini"}
    }


def _cell_results(root: Path, cap_slug: str, seed: str) -> dict[str, dict[str, Any]]:
    path = root / f"cap-{cap_slug}" / f"seed-{seed}" / (
        f"apollo-refresh-none-seed{seed}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"result not found: {path}")
    return _case_by_optimizer(path)


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_dir in sorted(root.glob("cap-*/seed-*"), key=_seed_cell_sort_key):
        if not seed_dir.is_dir():
            continue
        cap_match = _CAP_NAME.fullmatch(seed_dir.parent.name)
        seed_match = _SEED_NAME.fullmatch(seed_dir.name)
        if cap_match is None or seed_match is None:
            continue
        cap_slug = cap_match.group("cap")
        cap = "none" if cap_slug == "none" else cap_slug.replace("p", ".")
        seed = seed_match.group("seed")
        candidate = _cell_results(root, cap_slug, seed)
        baseline = _cell_results(root, "none", seed)
        for optimizer in ("APOLLO", "APOLLO-CAME", "APOLLO-Mini"):
            base = baseline[optimizer]
            case = candidate[optimizer]
            base_step = float(base["host_seconds_per_optimizer_step"])
            case_step = float(case["host_seconds_per_optimizer_step"])
            rows.append({
                "cap": cap,
                "seed": int(seed),
                "optimizer": optimizer,
                "validation_loss": float(case["final_validation_loss"]),
                "loss_delta": float(case["final_validation_loss"])
                - float(base["final_validation_loss"]),
                "step_seconds": case_step,
                "step_ratio": case_step / base_step,
                "state_bytes": int(case["persistent_state_bytes"]),
                "peak_allocated_bytes": int(case.get("peak_allocated_bytes", 0)),
                "capped_steps": int(
                    case.get("update_norm_variance_capped_steps", 0)
                ),
                "update_norm_variance": float(
                    case.get("update_norm_variance", 0.0)
                ),
                "status": "passed" if (
                    base.get("status") == "passed"
                    and case.get("status") == "passed"
                ) else "error",
            })
    if not rows:
        raise ValueError(f"no variance-cap sweep results found below {root}")
    return rows


def _number(value: float | int | None) -> str:
    return "-" if value is None else f"{value:.7g}"


def render(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| cap | seed | optimizer | validation loss | loss delta vs none | step s | step ratio | state bytes | peak allocated | capped steps | update norm variance | status |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {cap} | {seed} | {optimizer} | {validation_loss} | {loss_delta} | "
            "{step_seconds} | {step_ratio} | {state_bytes} | "
            "{peak_allocated_bytes} | {capped_steps} | {update_norm_variance} | {status} |".format(
                cap=row["cap"],
                seed=row["seed"],
                optimizer=row["optimizer"],
                validation_loss=_number(row["validation_loss"]),
                loss_delta=_number(row["loss_delta"]),
                step_seconds=_number(row["step_seconds"]),
                step_ratio=_number(row["step_ratio"]),
                state_bytes=_number(row["state_bytes"]),
                peak_allocated_bytes=_number(row["peak_allocated_bytes"]),
                capped_steps=row["capped_steps"],
                update_norm_variance=_number(row["update_norm_variance"]),
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
