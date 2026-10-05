"""Shared training progress and TensorBoard metric conventions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_standard_progress_rows(
    *,
    step: int,
    total_steps: int,
    global_step: int | None = None,
    loss: str | None = None,
    learning_rate: str | None = None,
    step_seconds: str | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Build the common Rich status rows used by simple train scripts."""
    rows = {"step": f"{step}/{total_steps}"}
    if global_step is not None:
        rows["global_step"] = str(global_step)
    if loss is not None:
        rows["loss"] = loss
    if learning_rate is not None:
        rows["lr"] = learning_rate
    if step_seconds is not None:
        rows["step_time"] = step_seconds
    if extra:
        rows.update({str(key): str(value) for key, value in extra.items()})
    return rows


def write_standard_training_metrics(
    writer: Any,
    *,
    step: int,
    train_loss: float,
    eval_loss: float | None = None,
    learning_rate: float | None = None,
    scheduled_learning_rate: float | None = None,
    steps_per_second: float | None = None,
    samples_per_second: float | None = None,
    extra: Mapping[str, float] | None = None,
) -> None:
    """Write canonical epoch-level tags while preserving script-specific tags."""
    writer.add_scalar("train/loss/total", float(train_loss), step)
    if eval_loss is not None:
        writer.add_scalar("eval/loss/total", float(eval_loss), step)
    if learning_rate is not None:
        writer.add_scalar("train/lr/effective", float(learning_rate), step)
    if scheduled_learning_rate is not None:
        writer.add_scalar(
            "train/lr/scheduled", float(scheduled_learning_rate), step,
        )
    if steps_per_second is not None:
        writer.add_scalar(
            "train/performance/steps_per_second",
            float(steps_per_second),
            step,
        )
    if samples_per_second is not None:
        writer.add_scalar(
            "train/performance/samples_per_second",
            float(samples_per_second),
            step,
        )
    if extra:
        for name, value in extra.items():
            writer.add_scalar(str(name), float(value), step)
