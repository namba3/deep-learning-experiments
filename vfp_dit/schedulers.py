"""Compatibility wrapper for the shared flow-sampling schedules."""

from __future__ import annotations

from flow_sampling import SCHEDULERS, build_timesteps as _build_timesteps

__all__ = ["SCHEDULERS", "build_timesteps"]


def build_timesteps(
    steps: int,
    *,
    scheduler: str = "flow_match_euler",
    flow_shift: float = 1.0,
    start_t: float = 1.0,
) -> tuple[float, ...]:
    """Return VFP-DiT model times using the shared FP32 schedule builder."""
    times = _build_timesteps(
        steps,
        scheduler=scheduler,
        shift=flow_shift,
        start_t=start_t,
    )
    return tuple(float(value) for value in times.tolist())
