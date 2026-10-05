"""Common lightweight checks and reporting before training starts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def build_training_preflight(
    *,
    script: str,
    device: Any,
    dtype: Any,
    output_dir: str | Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    seed: int | None = None,
    resume: str | Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate shared training arguments and return a serializable summary."""
    errors: list[str] = []
    if epochs <= 0:
        errors.append(f"epochs must be > 0 (got {epochs})")
    if batch_size <= 0:
        errors.append(f"batch_size must be > 0 (got {batch_size})")
    if num_workers < 0:
        errors.append(f"num_workers must be >= 0 (got {num_workers})")
    if seed is not None and seed < 0:
        errors.append(f"seed must be >= 0 (got {seed})")
    if not str(output_dir):
        errors.append("output_dir must not be empty")
    if resume is not None and not str(resume):
        errors.append("resume must not be empty when specified")
    if errors:
        raise ValueError(
            f"Preflight validation failed for {script}:\n"
            + "\n".join(f"- {error}" for error in errors)
        )

    summary: dict[str, Any] = {
        "script": script,
        "device": str(device),
        "dtype": str(dtype),
        "output_dir": str(output_dir),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "seed": seed,
        "resume": None if resume is None else str(resume),
    }
    if extra:
        summary.update(dict(extra))
    print(
        "Preflight: "
        f"script={script} device={summary['device']} dtype={summary['dtype']} "
        f"epochs={summary['epochs']} batch_size={summary['batch_size']} "
        f"num_workers={summary['num_workers']} output={summary['output_dir']}"
    )
    return summary
