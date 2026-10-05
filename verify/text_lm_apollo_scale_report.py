"""Summarize the resumable text-LM APOLLO scale sweep."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _decode_slug(value: str) -> str:
    return value.replace("m", "-").replace("p", ".").replace("x", "+")


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot summarize an empty list")
    return statistics.fmean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


def build_report(root: Path) -> str:
    rows: list[dict[str, object]] = []
    paths = sorted(root.glob("lr-*/scale-*/limiter-*/result.json"))
    paths += sorted(root.glob("rank-*/lr-*/scale-*/limiter-*/result.json"))
    for path in paths:
        result = json.loads(path.read_text())
        relative = path.relative_to(root).parts
        if len(relative) == 4:
            rank = "-"
            learning_rate_part, scale_part, limiter_part = relative[0:3]
        elif len(relative) == 5:
            rank = relative[0].removeprefix("rank-")
            learning_rate_part, scale_part, limiter_part = relative[1:4]
        else:
            continue
        learning_rate = _decode_slug(learning_rate_part.removeprefix("lr-"))
        scale = _decode_slug(scale_part.removeprefix("scale-"))
        limiter = limiter_part.removeprefix("limiter-")
        grouped: dict[str, list[dict[str, object]]] = {}
        for case in result.get("cases", []):
            grouped.setdefault(str(case["optimizer"]), []).append(case)
        for optimizer, cases in sorted(grouped.items()):
            ranks = {str(case.get("rank")) for case in cases}
            if len(ranks) == 1:
                rank = next(iter(ranks))
            losses = [float(case["final_validation_loss"]) for case in cases]
            states = [float(case["persistent_state_bytes"]) for case in cases]
            steps = [
                float(case["host_seconds_per_optimizer_step"]) * 1000.0
                for case in cases
            ]
            loss_mean, loss_std = _mean_std(losses)
            state_mean, state_std = _mean_std(states)
            step_mean, step_std = _mean_std(steps)
            update_norms = [
                float(case["update_norm_mean"])
                for case in cases
                if "update_norm_mean" in case
            ]
            norm_text = "not recorded"
            if update_norms:
                norm_mean, norm_std = _mean_std(update_norms)
                norm_text = f"{norm_mean:.6g} ± {norm_std:.4g}"
            rows.append({
                "learning_rate": learning_rate,
                "rank": rank,
                "scale": scale,
                "limiter": limiter,
                "optimizer": optimizer,
                "runs": len(cases),
                "loss": f"{loss_mean:.6g} ± {loss_std:.4g}",
                "state": f"{state_mean / 2**20:.3f} ± {state_std / 2**20:.3f} MiB",
                "step": f"{step_mean:.3f} ± {step_std:.3f} ms",
                "update_norm": norm_text,
                "status": result.get("status", "unknown"),
            })

    rows.sort(key=lambda row: (
        str(row["learning_rate"]), str(row["scale"]),
        str(row["limiter"]), str(row["optimizer"]),
    ))
    lines = [
        "# Text LM APOLLO scale sweep",
        "",
        f"Source: `{root}`",
        "",
        "| rank | learning rate | scale | limiter | optimizer | runs | validation loss mean±std | state MiB mean±std | step ms mean±std | update norm mean±std | status |",
        "| ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['learning_rate']} | {row['scale']} "
            f"| {row['limiter']} | {row['optimizer']} | {row['runs']} | {row['loss']} "
            f"| {row['state']} | {row['step']} | {row['update_norm']} "
            f"| {row['status']} |"
        )
    if not rows:
        lines.extend(["", "No completed result cells found."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    print(build_report(args.sweep_dir), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
