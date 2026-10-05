"""Common reporting helpers for no-training validation runs."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any

import torch


class ValidationTimer:
    """Measure validation wall time and optional CUDA peak memory."""

    def __init__(self, device: Any) -> None:
        self.device = device
        self.started_at = perf_counter()
        self.cuda_enabled = (
            getattr(device, "type", str(device)) == "cuda"
            and torch.cuda.is_available()
        )
        if self.cuda_enabled:
            torch.cuda.reset_peak_memory_stats(device)

    def finish(self) -> dict[str, Any]:
        """Return elapsed time and CUDA peak metrics collected so far."""
        if self.cuda_enabled:
            torch.cuda.synchronize(self.device)
        measurements: dict[str, Any] = {
            "validation_seconds": perf_counter() - self.started_at,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
        if self.cuda_enabled:
            measurements.update(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
            )
        return measurements


def build_validation_report(
    *,
    script: str,
    device: Any,
    dtype: Any,
    train_examples: int,
    eval_examples: int | None,
    model_parameters: int,
    trainable_parameters: int,
    steps_per_epoch: int,
    measurements: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and print a serializable summary for ``--validate-only``."""
    if train_examples <= 0:
        raise ValueError(f"train_examples must be > 0 (got {train_examples})")
    if eval_examples is not None and eval_examples < 0:
        raise ValueError(f"eval_examples must be >= 0 (got {eval_examples})")
    if model_parameters <= 0:
        raise ValueError(f"model_parameters must be > 0 (got {model_parameters})")
    if trainable_parameters < 0 or trainable_parameters > model_parameters:
        raise ValueError(
            "trainable_parameters must be between 0 and model_parameters "
            f"(got {trainable_parameters}/{model_parameters})"
        )
    if steps_per_epoch <= 0:
        raise ValueError(f"steps_per_epoch must be > 0 (got {steps_per_epoch})")

    report: dict[str, Any] = {
        "script": script,
        "device": str(device),
        "dtype": str(dtype),
        "train_examples": int(train_examples),
        "eval_examples": None if eval_examples is None else int(eval_examples),
        "model_parameters": int(model_parameters),
        "trainable_parameters": int(trainable_parameters),
        "steps_per_epoch": int(steps_per_epoch),
    }
    if extra:
        report.update(dict(extra))
    if measurements:
        report.update(dict(measurements))
    print(
        "Validation: "
        f"script={script} train={report['train_examples']} "
        f"eval={report['eval_examples']} parameters={report['model_parameters']:,} "
        f"trainable={report['trainable_parameters']:,} "
        f"steps_per_epoch={report['steps_per_epoch']} "
        f"elapsed={report.get('validation_seconds', 0.0):.3f}s"
    )
    if report.get("peak_allocated_bytes") is not None:
        print(
            "Validation memory: "
            f"peak_allocated={report['peak_allocated_bytes']} bytes "
            f"peak_reserved={report['peak_reserved_bytes']} bytes"
        )
    return report
