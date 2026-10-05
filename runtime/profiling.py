"""Low-overhead opt-in wall-clock timing for model components."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from collections.abc import Iterator

import torch


@contextmanager
def component_timer(enabled: bool, device: torch.device) -> Iterator[dict[str, float]]:
    """Measure a component, synchronizing CUDA only when profiling is enabled."""
    timing = {"seconds": 0.0}
    if not enabled:
        yield timing
        return

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    try:
        yield timing
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing["seconds"] = perf_counter() - started
