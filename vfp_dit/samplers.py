"""VFP-DiT adapter for the shared flow-sampling package."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from flow_sampling import (
    SamplingInfo,
    available_schedulers,
    available_solvers,
    build_timesteps,
    build_warped_timesteps,
    sample,
)

SOLVERS = available_solvers()
SCHEDULERS = available_schedulers()

VelocityFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def sample_flow_matching(
    predict_velocity: VelocityFunction,
    initial_noise: torch.Tensor,
    *,
    steps: int,
    solver: str = "euler",
    scheduler: str = "flow_match_euler",
    flow_shift: float = 1.0,
    generator: torch.Generator | None = None,
    er_sde_sigma_max: float = 80.0,
    rf_er_sde_eta: float = 0.2,
    rf_er_sde_cutoff: float = 0.9,
    rf_2m_warp_type: str = "identity",
    rf_2m_warp_shift: float = 1.0,
    rf_er_sde_warp_type: str = "identity",
    rf_er_sde_warp_shift: float = 1.0,
    rf_er_sde_warp_trust_lambda: float = 4.0,
    rf_er_sde_warp_trust_error_c: float = 0.1,
    rf_er_sde_warp_trust_epsilon: float = 1e-8,
    rf_trust_lambda: float = 4.0,
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, SamplingInfo]:
    """Sample x_t=(1-t)x0+t*epsilon from noise toward data.

    Model-specific conditioning and classifier-free guidance stay inside the
    callback. ER-SDE starts at its existing finite sigma endpoint and retains
    the caller's torch.Generator stream.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {', '.join(SOLVERS)}")
    if scheduler not in SCHEDULERS:
        raise ValueError(f"scheduler must be one of {', '.join(SCHEDULERS)}")
    if not math.isfinite(flow_shift) or flow_shift <= 0.0:
        raise ValueError("flow_shift must be finite and positive")
    if not math.isfinite(er_sde_sigma_max) or er_sde_sigma_max <= 0.0:
        raise ValueError("er_sde_sigma_max must be finite and positive")
    if solver == "rf_trust_region" and (
        not math.isfinite(rf_trust_lambda) or rf_trust_lambda < 0.0
    ):
        raise ValueError("rf_trust_lambda must be finite and non-negative")
    warped_solvers = {
        "rf_2m_warp", "rf_er_sde_warp_1", "rf_er_sde_warp_2m",
        "rf_er_sde_warp_trust",
    }
    if solver in warped_solvers:
        if flow_shift != 1.0:
            raise ValueError(f"{solver} owns its time transform; set flow_shift=1")
        if solver == "rf_2m_warp":
            warp_type, warp_shift = rf_2m_warp_type, rf_2m_warp_shift
        else:
            warp_type, warp_shift = rf_er_sde_warp_type, rf_er_sde_warp_shift
        grid = build_warped_timesteps(
            steps,
            warp=warp_type,
            shift=warp_shift,
            device=initial_noise.device,
        )
        return sample(
            predict_velocity,
            initial_noise,
            grid.model_timesteps,
            solver=solver,
            return_info=return_info,
            generator=generator,
            rf_er_sde_eta=rf_er_sde_eta,
            rf_er_sde_cutoff=rf_er_sde_cutoff,
            rf_er_sde_warp_trust_lambda=rf_er_sde_warp_trust_lambda,
            rf_er_sde_warp_trust_error_c=rf_er_sde_warp_trust_error_c,
            rf_er_sde_warp_trust_epsilon=rf_er_sde_warp_trust_epsilon,
            solver_times=grid.solver_times,
            warp_derivatives=grid.derivatives,
        )

    t_start = (
        er_sde_sigma_max / (1.0 + er_sde_sigma_max)
        if solver == "er_sde" else 1.0
    )
    timesteps = build_timesteps(
        steps,
        scheduler=scheduler,
        shift=flow_shift,
        start_t=t_start,
        device=initial_noise.device,
    )
    return sample(
        predict_velocity,
        initial_noise,
        timesteps,
        solver=solver,
        return_info=return_info,
        generator=generator,
        rf_er_sde_eta=rf_er_sde_eta,
        rf_er_sde_cutoff=rf_er_sde_cutoff,
        rf_er_sde_warp_trust_lambda=rf_er_sde_warp_trust_lambda,
        rf_er_sde_warp_trust_error_c=rf_er_sde_warp_trust_error_c,
        rf_er_sde_warp_trust_epsilon=rf_er_sde_warp_trust_epsilon,
        rf_trust_lambda=rf_trust_lambda,
    )
