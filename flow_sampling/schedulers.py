"""Timestep schedules for flow-matching models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WarpedTimeGrid:
    """Aligned solver progress, descending model times, and d tau / d r."""

    solver_times: torch.Tensor
    model_timesteps: torch.Tensor
    derivatives: torch.Tensor
    warp_type: str
    warp_parameter: float


SCHEDULERS = ("uniform", "flow_match_euler")


def available_schedulers() -> tuple[str, ...]:
    """Return the timestep schedule names currently implemented here."""
    return SCHEDULERS


def build_timesteps(
    num_steps: int,
    *,
    scheduler: str = "flow_match_euler",
    shift: float = 1.0,
    start_t: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a descending flow time grid with num_steps + 1 points.

    uniform is linear in time. flow_match_euler applies the static Diffusers
    FlowMatch shift s*t / (1 + (s-1)*t) to a linear grid. Arithmetic is
    performed in FP32 before the requested output cast.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if scheduler not in SCHEDULERS:
        raise ValueError(f"scheduler must be one of {SCHEDULERS}, got {scheduler!r}")
    if not math.isfinite(shift) or shift <= 0.0:
        raise ValueError(f"shift must be finite and > 0, got {shift}")
    if scheduler == "uniform" and shift != 1.0:
        raise ValueError("shift only applies to the flow_match_euler scheduler")
    if not math.isfinite(start_t) or not 0.0 < start_t <= 1.0:
        raise ValueError(f"start_t must be finite and in (0, 1], got {start_t}")
    if not dtype.is_floating_point:
        raise ValueError(f"dtype must be floating point, got {dtype}")

    times = torch.linspace(start_t, 0.0, num_steps + 1, device=device, dtype=torch.float32)
    if scheduler == "flow_match_euler":
        times = shift * times / (1.0 + (shift - 1.0) * times)
    times[-1] = 0.0
    return times.to(dtype=dtype)


def build_warped_timesteps(
    num_steps: int,
    *,
    warp: str = "identity",
    shift: float = 1.0,
    q_min: float = 1e-3,
    q_max: float = 1e3,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> WarpedTimeGrid:
    """Build a uniform tau grid and its aligned RF model times and q=d tau/dr.

    Sampling progress r increases from noise (0) to data (1), while returned
    model times t=1-r are strictly descending. RationalWarp uses
    phi(r)=shift*r/(1+(shift-1)*r); no runtime derivative clamp is applied.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if warp not in {"identity", "rational"}:
        raise ValueError("warp must be 'identity' or 'rational'")
    if not math.isfinite(shift) or shift <= 0.0:
        raise ValueError(f"shift must be finite and positive, got {shift}")
    if warp == "identity" and shift != 1.0:
        raise ValueError("shift only applies to the rational warp")
    if not math.isfinite(q_min) or not math.isfinite(q_max) or q_min <= 0.0 or q_max < q_min:
        raise ValueError("q bounds must be finite, positive, and ordered")
    if not dtype.is_floating_point:
        raise ValueError(f"dtype must be floating point, got {dtype}")

    tau = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=torch.float32)
    if warp == "identity":
        progress = tau
        q = torch.ones_like(tau)
    else:
        # Invert tau=shift*r/(1+(shift-1)*r) and evaluate its exact derivative.
        progress = tau / (shift - (shift - 1.0) * tau)
        q = shift / (1.0 + (shift - 1.0) * progress).square()
        if min(shift, 1.0 / shift) < q_min or max(shift, 1.0 / shift) > q_max:
            raise ValueError(
                f"rational warp derivative leaves configured bounds [{q_min}, {q_max}]"
            )
    model_times = 1.0 - progress
    tau[0], tau[-1] = 0.0, 1.0
    model_times[0], model_times[-1] = 1.0, 0.0
    if not bool(torch.isfinite(q).all()) or not bool(torch.all(q > 0.0)):
        raise ValueError("time warp produced a non-finite or non-positive derivative")
    if not bool(torch.all(model_times[:-1] > model_times[1:])):
        raise ValueError("time warp does not produce strictly descending model times")
    if float(q.min()) < q_min or float(q.max()) > q_max:
        raise ValueError("time warp derivative violates configured q bounds")
    solver_times = tau.to(dtype=dtype)
    model_timesteps = model_times.to(dtype=dtype)
    derivatives = q.to(dtype=dtype)
    if not bool(torch.all(solver_times[1:] > solver_times[:-1])):
        raise ValueError(f"{dtype} cannot represent a strictly ascending solver grid")
    if not bool(torch.all(model_timesteps[:-1] > model_timesteps[1:])):
        raise ValueError(f"{dtype} cannot represent strictly descending model times")
    if not bool(torch.isfinite(derivatives).all()) or not bool(torch.all(derivatives > 0.0)):
        raise ValueError(f"{dtype} cannot represent positive finite warp derivatives")
    return WarpedTimeGrid(
        solver_times=solver_times,
        model_timesteps=model_timesteps,
        derivatives=derivatives,
        warp_type=warp,
        warp_parameter=shift,
    )



def warp_timesteps(
    model_timesteps: torch.Tensor,
    *,
    warp: str = "identity",
    shift: float = 1.0,
    q_min: float = 1e-3,
    q_max: float = 1e3,
) -> WarpedTimeGrid:
    """Pair an existing descending model-time grid with warped progress data.

    Unlike :func:`build_warped_timesteps`, this preserves the supplied model
    times and applies the time change to their progress coordinates. This is
    useful when a host scheduler owns the model-time spacing.
    """
    if model_timesteps.ndim != 1 or model_timesteps.numel() < 2:
        raise ValueError("model_timesteps must be 1D with at least two entries")
    if not model_timesteps.is_floating_point():
        raise ValueError("model_timesteps must be floating point")
    if not bool(torch.isfinite(model_timesteps).all()):
        raise ValueError("model_timesteps must contain only finite values")
    if not bool(torch.all(model_timesteps[:-1] > model_timesteps[1:])):
        raise ValueError("model_timesteps must be strictly descending")
    if float(model_timesteps[0]) != 1.0 or float(model_timesteps[-1]) != 0.0:
        raise ValueError("model_timesteps must span [1, 0]")
    if warp not in {"identity", "rational"}:
        raise ValueError("warp must be 'identity' or 'rational'")
    if not math.isfinite(shift) or shift <= 0.0:
        raise ValueError("shift must be finite and positive")
    if warp == "identity" and shift != 1.0:
        raise ValueError("shift only applies to the rational warp")
    if not math.isfinite(q_min) or not math.isfinite(q_max) or q_min <= 0.0 or q_max < q_min:
        raise ValueError("q bounds must be finite, positive, and ordered")

    times = model_timesteps.to(dtype=torch.float32)
    progress = 1.0 - times
    if warp == "identity":
        solver_times = progress
        derivatives = torch.ones_like(progress)
    else:
        denominator = 1.0 + (shift - 1.0) * progress
        solver_times = shift * progress / denominator
        derivatives = shift / denominator.square()
    solver_times[0], solver_times[-1] = 0.0, 1.0
    if not bool(torch.isfinite(solver_times).all()) or not bool(torch.all(solver_times[1:] > solver_times[:-1])):
        raise ValueError("warp must produce finite, strictly ascending solver times")
    if not bool(torch.isfinite(derivatives).all()) or not bool(torch.all(derivatives > 0.0)):
        raise ValueError("warp derivatives must be finite and positive")
    if float(derivatives.min()) < q_min or float(derivatives.max()) > q_max:
        raise ValueError(f"warp derivative leaves configured bounds [{q_min}, {q_max}]")

    dtype = model_timesteps.dtype
    solver_times = solver_times.to(dtype=dtype)
    derivatives = derivatives.to(dtype=dtype)
    if not bool(torch.all(solver_times[1:] > solver_times[:-1])):
        raise ValueError(f"{dtype} cannot represent a strictly ascending solver grid")
    if not bool(torch.isfinite(derivatives).all()) or not bool(torch.all(derivatives > 0.0)):
        raise ValueError(f"{dtype} cannot represent finite, positive warp derivatives")
    return WarpedTimeGrid(
        solver_times=solver_times,
        model_timesteps=model_timesteps,
        derivatives=derivatives,
        warp_type=warp,
        warp_parameter=shift,
    )


def flow_match_timesteps(
    num_steps: int,
    *,
    shift: float = 1.0,
    start_t: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compatibility wrapper for the static FlowMatch schedule."""
    return build_timesteps(
        num_steps,
        scheduler="flow_match_euler",
        shift=shift,
        start_t=start_t,
        device=device,
        dtype=dtype,
    )
