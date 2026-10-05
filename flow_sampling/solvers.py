"""Inference solvers for velocity fields trained with flow matching."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

VelocityFn = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class SamplingInfo:
    """Small amount of accounting useful for sampler comparisons."""

    solver: str
    num_steps: int
    num_model_evaluations: int


def available_solvers() -> tuple[str, ...]:
    """Return the solver names currently implemented by this package."""
    return (
        "euler",
        "fireflow",
        "abm2",
        "er_sde",
        "rf_ab2",
        "rf_2m_warp",
        "rf_trust_region",
        "rf_er_sde_1",
        "rf_er_sde_2m",
        "rf_er_sde_trust",
        "rf_er_sde_warp_1",
        "rf_er_sde_warp_2m",
        "rf_er_sde_warp_trust",
    )


def _validate_inputs(initial: Tensor, timesteps: Tensor, solver: str) -> None:
    if initial.ndim < 2:
        raise ValueError(f"initial must have a batch axis and data axes, got {initial.shape}")
    if not initial.is_floating_point():
        raise ValueError(f"initial must be floating point, got {initial.dtype}")
    if timesteps.ndim != 1 or timesteps.numel() < 2:
        raise ValueError("timesteps must be a 1D tensor with at least two entries")
    if not timesteps.is_floating_point():
        raise ValueError(f"timesteps must be floating point, got {timesteps.dtype}")
    if not bool(torch.isfinite(timesteps).all()):
        raise ValueError("timesteps must contain only finite values")
    if not bool(torch.all(timesteps[:-1] > timesteps[1:])):
        raise ValueError("timesteps must be strictly descending")
    if solver not in available_solvers():
        raise ValueError(f"unknown solver {solver!r}; choose from {available_solvers()}")


def _accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    # Preserve double precision; use FP32 accumulation for all lower precision.
    return torch.float64 if dtype == torch.float64 else torch.float32


def _time_batch(time: Tensor, batch_size: int, device: torch.device) -> Tensor:
    return time.reshape(1).expand(batch_size)


def _evaluate(
    velocity_fn: VelocityFn,
    state: Tensor,
    time: Tensor,
    model_dtype: torch.dtype,
) -> Tensor:
    # Keep the solver's master state in FP32 while presenting the callback with
    # the input dtype, which is typically the model/autocast boundary dtype.
    model_state = state.to(dtype=model_dtype)
    velocity = velocity_fn(model_state, _time_batch(time, state.shape[0], state.device))
    if not isinstance(velocity, Tensor):
        raise TypeError("velocity_fn must return a torch.Tensor")
    if velocity.shape != model_state.shape:
        raise ValueError(
            "velocity_fn must return a tensor with the same shape as its input; "
            f"got {tuple(velocity.shape)} for input {tuple(model_state.shape)}"
        )
    if velocity.device != state.device:
        raise ValueError(f"velocity_fn returned {velocity.device}, expected {state.device}")
    if not velocity.is_floating_point():
        raise ValueError(f"velocity_fn must return a floating tensor, got {velocity.dtype}")
    return velocity.to(dtype=state.dtype)


def _add_scaled(state: Tensor, scale: Tensor, velocity: Tensor) -> Tensor:
    # state and velocity are in the solver accumulation dtype, and scale is a
    # scalar tensor in the same dtype. The caller casts only at model boundaries.
    return state + scale * velocity


@torch.no_grad()
def sample(
    velocity_fn: VelocityFn,
    initial: Tensor,
    timesteps: Tensor,
    *,
    solver: str = "euler",
    return_info: bool = False,
    generator: torch.Generator | None = None,
    rf_er_sde_eta: float = 0.2,
    rf_er_sde_cutoff: float = 0.9,
    rf_er_sde_warp_trust_lambda: float = 4.0,
    rf_er_sde_warp_trust_error_c: float = 0.1,
    rf_er_sde_warp_trust_epsilon: float = 1e-8,
    rf_er_sde_trust_lambda: float = 4.0,
    rf_er_sde_trust_error_c: float = 0.1,
    rf_er_sde_trust_epsilon: float = 1e-8,
    solver_times: Tensor | None = None,
    warp_derivatives: Tensor | None = None,
    rf_trust_lambda: float = 4.0,
) -> Tensor | tuple[Tensor, SamplingInfo]:
    """Integrate a flow-matching velocity field over a descending time grid.

    velocity_fn(x, t_batch) returns dx/dt. Its result must match x in shape
    and device; its floating dtype may differ (for example, a BF16 model under
    autocast), and is converted to the solver's accumulation dtype. The solver
    keeps an FP32 master state for FP16/BF16/FP32 inputs and casts back to the
    input dtype only when evaluating the callback and returning the sample.

    The common training path x_t=(1-t)*data+t*noise is integrated from t=1
    toward t=0. ER-SDE uses sigma=t/(1-t), so its grid must start below one.
    RF solvers require times in [0, 1] and advance in positive progress
    r=1-t. rf_ab2 is raw variable-step AB2; rf_2m_warp applies AB2 to the
    transformed velocity on an ascending solver-coordinate grid.
    rf_trust_region gates raw AB2 correction per sample using the prior-step
    velocity prediction error. rf_er_sde_1 is Euler-Maruyama; rf_er_sde_2m
    extrapolates only sampling-direction velocity. rf_er_sde_trust gates the
    variable-step AB2 correction and stochastic envelope from a per-sample
    velocity prediction error on the input progress grid. The _warp variants
    apply those stochastic updates in an ascending warped-progress coordinate.
    rf_er_sde_warp_trust additionally gates warped 2M and the SDE scale using
    the per-sample transformed-velocity prediction error.
    """
    _validate_inputs(initial, timesteps, solver)
    warped_solvers = {
        "rf_2m_warp", "rf_er_sde_warp_1", "rf_er_sde_warp_2m",
        "rf_er_sde_warp_trust",
    }
    if solver in warped_solvers:
        if solver_times is None or warp_derivatives is None:
            raise ValueError(f"{solver} requires solver_times and warp_derivatives")
        if solver_times.ndim != 1 or solver_times.numel() != timesteps.numel():
            raise ValueError("solver_times must be 1D and aligned with timesteps")
        if warp_derivatives.ndim != 1 or warp_derivatives.numel() != timesteps.numel():
            raise ValueError("warp_derivatives must be 1D and aligned with timesteps")
        if not solver_times.is_floating_point() or not warp_derivatives.is_floating_point():
            raise ValueError("solver_times and warp_derivatives must be floating point")
        if not bool(torch.isfinite(solver_times).all()):
            raise ValueError("solver_times must contain only finite values")
        if not bool(torch.isfinite(warp_derivatives).all()) or not bool(torch.all(warp_derivatives > 0.0)):
            raise ValueError("warp_derivatives must be finite and positive")
        if not bool(torch.all(solver_times[1:] > solver_times[:-1])):
            raise ValueError("solver_times must be strictly ascending")
        if float(solver_times[0]) != 0.0 or float(solver_times[-1]) != 1.0:
            raise ValueError(f"{solver} solver_times must span [0, 1]")
        if float(timesteps[0]) != 1.0 or float(timesteps[-1]) != 0.0:
            raise ValueError(f"{solver} model timesteps must span [1, 0]")
    elif solver_times is not None or warp_derivatives is not None:
        raise ValueError("solver_times and warp_derivatives require a warped solver")
    rf_solvers = {
        "rf_ab2", "rf_2m_warp", "rf_trust_region", "rf_er_sde_1",
        "rf_er_sde_2m", "rf_er_sde_trust", "rf_er_sde_warp_1",
        "rf_er_sde_warp_2m", "rf_er_sde_warp_trust",
    }
    if solver in rf_solvers:
        if float(timesteps[0]) > 1.0 or float(timesteps[-1]) < 0.0:
            raise ValueError("RF residual/SDE solvers require times in [0, 1]")
    if solver == "rf_trust_region" and (
        not math.isfinite(rf_trust_lambda) or rf_trust_lambda < 0.0
    ):
        raise ValueError("rf_trust_lambda must be finite and non-negative")
    rf_er_sde_solvers = {
        "rf_er_sde_1", "rf_er_sde_2m", "rf_er_sde_trust",
        "rf_er_sde_warp_1", "rf_er_sde_warp_2m", "rf_er_sde_warp_trust",
    }
    if solver in rf_er_sde_solvers:
        if not math.isfinite(rf_er_sde_eta) or rf_er_sde_eta < 0.0:
            raise ValueError("rf_er_sde_eta must be finite and non-negative")
        if (
            not math.isfinite(rf_er_sde_cutoff)
            or not 0.0 <= rf_er_sde_cutoff < 1.0
        ):
            raise ValueError("rf_er_sde_cutoff must be finite and in [0, 1)")
    if solver == "rf_er_sde_trust":
        if not math.isfinite(rf_er_sde_trust_lambda) or rf_er_sde_trust_lambda < 0.0:
            raise ValueError("rf_er_sde_trust_lambda must be finite and non-negative")
        if not math.isfinite(rf_er_sde_trust_error_c) or rf_er_sde_trust_error_c <= 0.0:
            raise ValueError("rf_er_sde_trust_error_c must be finite and positive")
        if not math.isfinite(rf_er_sde_trust_epsilon) or rf_er_sde_trust_epsilon <= 0.0:
            raise ValueError("rf_er_sde_trust_epsilon must be finite and positive")
    if solver == "rf_er_sde_warp_trust":
        if (
            not math.isfinite(rf_er_sde_warp_trust_lambda)
            or rf_er_sde_warp_trust_lambda < 0.0
        ):
            raise ValueError("rf_er_sde_warp_trust_lambda must be finite and non-negative")
        if (
            not math.isfinite(rf_er_sde_warp_trust_error_c)
            or rf_er_sde_warp_trust_error_c <= 0.0
        ):
            raise ValueError("rf_er_sde_warp_trust_error_c must be finite and positive")
        if (
            not math.isfinite(rf_er_sde_warp_trust_epsilon)
            or rf_er_sde_warp_trust_epsilon <= 0.0
        ):
            raise ValueError("rf_er_sde_warp_trust_epsilon must be finite and positive")
    if solver == "er_sde" and float(timesteps[0]) >= 1.0:
        raise ValueError("er_sde requires a starting timestep below 1")

    accumulation_dtype = _accumulation_dtype(initial.dtype)
    times = timesteps.to(device=initial.device, dtype=accumulation_dtype)
    x = initial.to(dtype=accumulation_dtype)
    model_dtype = initial.dtype
    evaluations = 0
    trust_signal_finite: Tensor | None = None

    if solver == "rf_2m_warp":
        tau = solver_times.to(device=initial.device, dtype=accumulation_dtype)
        q = warp_derivatives.to(device=initial.device, dtype=accumulation_dtype)
        previous_transformed_velocity: Tensor | None = None
        previous_solver_step: Tensor | None = None
        for i in range(times.numel() - 1):
            velocity = _evaluate(velocity_fn, x, times[i], model_dtype)
            evaluations += 1
            transformed_velocity = -velocity / q[i]
            solver_step = tau[i + 1] - tau[i]
            if previous_transformed_velocity is None:
                effective_velocity = transformed_velocity
            else:
                assert previous_solver_step is not None
                ratio = solver_step / previous_solver_step
                effective_velocity = transformed_velocity + (0.5 * ratio) * (
                    transformed_velocity - previous_transformed_velocity
                )
            x = _add_scaled(x, solver_step, effective_velocity)
            previous_transformed_velocity = transformed_velocity
            previous_solver_step = solver_step
    elif solver == "rf_ab2":
        previous_sampling_velocity: Tensor | None = None
        previous_progress_step: Tensor | None = None
        for i in range(times.numel() - 1):
            velocity = _evaluate(velocity_fn, x, times[i], model_dtype)
            evaluations += 1
            sampling_velocity = -velocity
            progress_step = times[i] - times[i + 1]
            if previous_sampling_velocity is None:
                effective_velocity = sampling_velocity
            else:
                assert previous_progress_step is not None
                ratio = progress_step / previous_progress_step
                effective_velocity = sampling_velocity + (0.5 * ratio) * (
                    sampling_velocity - previous_sampling_velocity
                )
            x = _add_scaled(x, progress_step, effective_velocity)
            previous_sampling_velocity = sampling_velocity
            previous_progress_step = progress_step
    elif solver == "rf_trust_region":
        previous_velocity: Tensor | None = None
        previous_previous_velocity: Tensor | None = None
        previous_step: Tensor | None = None
        previous_previous_step: Tensor | None = None
        feature_axes = tuple(range(1, x.ndim))
        broadcast_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        for i in range(times.numel() - 1):
            velocity = _evaluate(velocity_fn, x, times[i], model_dtype)
            evaluations += 1
            sampling_velocity = -velocity
            step = times[i] - times[i + 1]
            if previous_velocity is None:
                effective_velocity = sampling_velocity
            else:
                assert previous_step is not None
                ratio = step / previous_step
                correction = (0.5 * ratio) * (sampling_velocity - previous_velocity)
                if previous_previous_velocity is None:
                    trust = sampling_velocity.new_ones((x.shape[0],))
                else:
                    assert previous_previous_step is not None
                    predicted_velocity = previous_velocity + (
                        previous_step / previous_previous_step
                    ) * (previous_velocity - previous_previous_velocity)
                    prediction_rms = torch.sqrt(torch.mean(
                        (sampling_velocity - predicted_velocity).square(),
                        dim=feature_axes,
                    ))
                    velocity_rms = torch.sqrt(torch.mean(
                        sampling_velocity.square(), dim=feature_axes,
                    ))
                    prediction_error = prediction_rms / (velocity_rms + 1e-8)
                    current_trust_signal_finite = torch.isfinite(prediction_error).all()
                    trust_signal_finite = (
                        current_trust_signal_finite if trust_signal_finite is None
                        else trust_signal_finite & current_trust_signal_finite
                    )
                    trust = torch.exp(-rf_trust_lambda * prediction_error)
                effective_velocity = sampling_velocity + trust.reshape(
                    broadcast_shape
                ) * correction
            x = _add_scaled(x, step, effective_velocity)
            previous_previous_velocity = previous_velocity
            previous_velocity = sampling_velocity
            previous_previous_step = previous_step
            previous_step = step
    elif solver == "rf_er_sde_trust":
        feature_axes = tuple(range(1, x.ndim))
        broadcast_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        previous_u: Tensor | None = None
        previous_previous_u: Tensor | None = None
        previous_step: Tensor | None = None
        previous_previous_step: Tensor | None = None
        for i in range(times.numel() - 1):
            timestep = times[i]
            step = timestep - times[i + 1]
            velocity = _evaluate(velocity_fn, x, timestep, model_dtype)
            evaluations += 1
            transformed_velocity = -velocity

            if previous_u is None:
                gamma = x.new_ones((x.shape[0],))
                prediction_error: Tensor | None = None
                effective_velocity = transformed_velocity
            else:
                assert previous_step is not None
                residual = transformed_velocity - previous_u
                ab2_scale = 0.5 * step / previous_step
                if previous_previous_u is None:
                    gamma = x.new_ones((x.shape[0],))
                    prediction_error = None
                else:
                    assert previous_previous_step is not None
                    predicted_velocity = previous_u + (
                        previous_step / previous_previous_step
                    ) * (previous_u - previous_previous_u)
                    prediction_rms = torch.sqrt(torch.mean(
                        (transformed_velocity - predicted_velocity).square(),
                        dim=feature_axes,
                    ))
                    velocity_rms = torch.sqrt(torch.mean(
                        transformed_velocity.square(), dim=feature_axes,
                    ))
                    prediction_error = prediction_rms / velocity_rms.clamp_min(
                        rf_er_sde_trust_epsilon
                    )
                    current_error_finite = torch.isfinite(prediction_error).all()
                    trust_signal_finite = (
                        current_error_finite if trust_signal_finite is None
                        else trust_signal_finite & current_error_finite
                    )
                    gamma = torch.exp(
                        -rf_er_sde_trust_lambda * prediction_error
                    )
                effective_velocity = transformed_velocity + (
                    ab2_scale * gamma.reshape(broadcast_shape) * residual
                )

            progress = 1.0 - timestep
            base_diffusion = rf_er_sde_eta * torch.sin(
                torch.pi * progress
            ).clamp_min(0.0)
            if i < 2 or bool(progress >= rf_er_sde_cutoff):
                diffusion = x.new_zeros((x.shape[0],))
            else:
                assert prediction_error is not None
                diffusion = base_diffusion * prediction_error / (
                    prediction_error + rf_er_sde_trust_error_c
                )
            diffusion_view = diffusion.reshape(broadcast_shape)

            drift = effective_velocity
            if bool(torch.any(diffusion > 0.0)):
                # This RF score identity is valid at interior model times t>0.
                score = -(x + (1.0 - timestep) * velocity) / timestep
                drift = drift + 0.5 * diffusion_view.square() * score
                noise = torch.randn(
                    x.shape,
                    dtype=x.dtype,
                    device=x.device,
                    generator=generator,
                )
                x = _add_scaled(x, step, drift)
                x = x + diffusion_view * torch.sqrt(step) * noise
            else:
                x = _add_scaled(x, step, drift)

            previous_previous_u = previous_u
            previous_u = transformed_velocity
            previous_previous_step = previous_step
            previous_step = step
    elif solver == "rf_er_sde_warp_trust":
        tau = solver_times.to(device=initial.device, dtype=accumulation_dtype)
        q = warp_derivatives.to(device=initial.device, dtype=accumulation_dtype)
        feature_axes = tuple(range(1, x.ndim))
        broadcast_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        previous_u: Tensor | None = None
        previous_previous_u: Tensor | None = None
        previous_step: Tensor | None = None
        previous_previous_step: Tensor | None = None
        for i in range(times.numel() - 1):
            timestep = times[i]
            step = tau[i + 1] - tau[i]
            velocity = _evaluate(velocity_fn, x, timestep, model_dtype)
            evaluations += 1
            transformed_velocity = -velocity / q[i]

            if previous_u is None:
                gamma = x.new_ones((x.shape[0],))
                prediction_error: Tensor | None = None
                effective_velocity = transformed_velocity
            else:
                assert previous_step is not None
                residual = transformed_velocity - previous_u
                ab2_scale = 0.5 * step / previous_step
                if previous_previous_u is None:
                    # No two-step prediction exists at startup: keep AB2 ungated.
                    gamma = x.new_ones((x.shape[0],))
                    prediction_error = None
                else:
                    assert previous_previous_step is not None
                    predicted_velocity = previous_u + (
                        previous_step / previous_previous_step
                    ) * (previous_u - previous_previous_u)
                    prediction_rms = torch.sqrt(torch.mean(
                        (transformed_velocity - predicted_velocity).square(),
                        dim=feature_axes,
                    ))
                    velocity_rms = torch.sqrt(torch.mean(
                        transformed_velocity.square(), dim=feature_axes,
                    ))
                    prediction_error = prediction_rms / velocity_rms.clamp_min(
                        rf_er_sde_warp_trust_epsilon
                    )
                    current_error_finite = torch.isfinite(prediction_error).all()
                    trust_signal_finite = (
                        current_error_finite if trust_signal_finite is None
                        else trust_signal_finite & current_error_finite
                    )
                    gamma = torch.exp(
                        -rf_er_sde_warp_trust_lambda * prediction_error
                    )
                effective_velocity = transformed_velocity + (
                    ab2_scale * gamma.reshape(broadcast_shape) * residual
                )

            progress = 1.0 - timestep
            base_diffusion = rf_er_sde_eta * torch.sin(
                torch.pi * progress
            ).clamp_min(0.0)
            if i < 2 or bool(progress >= rf_er_sde_cutoff):
                diffusion = x.new_zeros((x.shape[0],))
            else:
                assert prediction_error is not None
                diffusion = base_diffusion * prediction_error / (
                    prediction_error + rf_er_sde_warp_trust_error_c
                )
            diffusion_view = diffusion.reshape(broadcast_shape)

            drift = effective_velocity
            if bool(torch.any(diffusion > 0.0)):
                # RF score uses model-time t as the noise coefficient.
                score = -(x + (1.0 - timestep) * velocity) / timestep
                drift = drift + (
                    0.5 * diffusion_view.square() * score / q[i]
                )
                noise = torch.randn(
                    x.shape,
                    dtype=x.dtype,
                    device=x.device,
                    generator=generator,
                )
                x = _add_scaled(x, step, drift)
                x = x + (diffusion_view / torch.sqrt(q[i])) * torch.sqrt(step) * noise
            else:
                x = _add_scaled(x, step, drift)

            previous_previous_u = previous_u
            previous_u = transformed_velocity
            previous_previous_step = previous_step
            previous_step = step
    elif solver in rf_er_sde_solvers:
        warped = solver in {"rf_er_sde_warp_1", "rf_er_sde_warp_2m"}
        use_history = solver in {"rf_er_sde_2m", "rf_er_sde_warp_2m"}
        tau = (
            solver_times.to(device=initial.device, dtype=accumulation_dtype)
            if warped else None
        )
        q = (
            warp_derivatives.to(device=initial.device, dtype=accumulation_dtype)
            if warped else None
        )
        previous_sampling_velocity: Tensor | None = None
        previous_solver_step: Tensor | None = None
        for i in range(times.numel() - 1):
            timestep = times[i]
            step = (tau[i + 1] - tau[i]) if warped else (timestep - times[i + 1])
            velocity = _evaluate(velocity_fn, x, timestep, model_dtype)
            evaluations += 1

            progress = 1.0 - timestep
            diffusion = torch.sin(torch.pi * progress).clamp_min(0.0)
            diffusion = rf_er_sde_eta * diffusion
            if bool(progress >= rf_er_sde_cutoff):
                diffusion = diffusion.new_zeros(())

            if warped:
                assert q is not None
                sampling_velocity = -velocity / q[i]
                diffusion_in_solver_time = diffusion / torch.sqrt(q[i])
                score_scale = 1.0 / q[i]
            else:
                sampling_velocity = -velocity
                diffusion_in_solver_time = diffusion
                score_scale = diffusion.new_ones(())

            if use_history and previous_sampling_velocity is not None:
                assert previous_solver_step is not None
                ratio = step / previous_solver_step
                sampling_velocity = sampling_velocity + (0.5 * ratio) * (
                    sampling_velocity - previous_sampling_velocity
                )

            drift = sampling_velocity
            if bool(diffusion > 0.0):
                # In this parameterization the noise coefficient is t. The
                # exact linear-RF score is -[x + (1-t)u] / t.
                score = -(x + (1.0 - timestep) * velocity) / timestep
                drift = drift + (0.5 * diffusion.square()) * score_scale * score
                noise = torch.randn(
                    x.shape,
                    dtype=x.dtype,
                    device=x.device,
                    generator=generator,
                )
                x = _add_scaled(x, step, drift)
                x = x + diffusion_in_solver_time * torch.sqrt(step) * noise
            else:
                x = _add_scaled(x, step, drift)

            previous_sampling_velocity = (
                -velocity / q[i] if warped and q is not None else -velocity
            )
            previous_solver_step = step
    elif solver == "euler":
        for i in range(times.numel() - 1):
            velocity = _evaluate(velocity_fn, x, times[i], model_dtype)
            evaluations += 1
            dt = times[i + 1] - times[i]
            x = _add_scaled(x, dt, velocity)
    elif solver == "fireflow":
        cached_midpoint_velocity: Tensor | None = None
        for i in range(times.numel() - 1):
            dt = times[i + 1] - times[i]
            if cached_midpoint_velocity is None:
                predictor_velocity = _evaluate(velocity_fn, x, times[i], model_dtype)
                evaluations += 1
            else:
                predictor_velocity = cached_midpoint_velocity
            midpoint_x = _add_scaled(x, 0.5 * dt, predictor_velocity)
            midpoint_t = 0.5 * (times[i] + times[i + 1])
            cached_midpoint_velocity = _evaluate(
                velocity_fn, midpoint_x, midpoint_t, model_dtype,
            )
            evaluations += 1
            x = _add_scaled(x, dt, cached_midpoint_velocity)
    elif solver == "abm2":
        first_dt = times[1] - times[0]
        previous_velocity = _evaluate(velocity_fn, x, times[0], model_dtype)
        evaluations += 1
        euler_predictor = _add_scaled(x, first_dt, previous_velocity)
        next_velocity = _evaluate(velocity_fn, euler_predictor, times[1], model_dtype)
        evaluations += 1
        x = x + (0.5 * first_dt) * (previous_velocity + next_velocity)

        # Variable-step AB2 predictor followed by a trapezoidal corrector.
        # The corrected endpoint derivative is approximated by the derivative
        # already evaluated at the predictor, so this PECE variant adds one NFE
        # per later interval. It is a fixed-grid method, not adaptive ABM.
        for i in range(1, times.numel() - 1):
            dt = times[i + 1] - times[i]
            previous_dt = times[i] - times[i - 1]
            ratio = dt / previous_dt
            predictor_velocity = (
                (1.0 + 0.5 * ratio) * next_velocity
                - (0.5 * ratio) * previous_velocity
            )
            predictor_x = _add_scaled(x, dt, predictor_velocity)
            current_velocity = _evaluate(
                velocity_fn, predictor_x, times[i + 1], model_dtype,
            )
            evaluations += 1
            x = x + (0.5 * dt) * (next_velocity + current_velocity)
            previous_velocity, next_velocity = next_velocity, current_velocity
    else:  # er_sde
        for i in range(times.numel() - 1):
            timestep = times[i]
            next_timestep = times[i + 1]
            velocity = _evaluate(velocity_fn, x, timestep, model_dtype)
            evaluations += 1
            x0 = x - timestep * velocity
            sigma = timestep / (1.0 - timestep)
            next_sigma = next_timestep / (1.0 - next_timestep)
            ratio = next_sigma / sigma if bool(sigma > 0.0) else sigma.new_zeros(())
            transition = ratio * ratio
            variance = torch.clamp(
                next_sigma * next_sigma - sigma * sigma * transition * transition,
                min=0.0,
            )

            ve_state = x / (1.0 - timestep)
            next_ve_state = transition * ve_state + (1.0 - transition) * x0
            if bool(variance > 0.0):
                noise = torch.randn(
                    x.shape,
                    dtype=x.dtype,
                    device=x.device,
                    generator=generator,
                )
                next_ve_state = next_ve_state + torch.sqrt(variance) * noise
            x = (1.0 - next_timestep) * next_ve_state

    if trust_signal_finite is not None and not bool(trust_signal_finite):
        if solver == "rf_trust_region":
            raise ValueError("rf_trust_region produced a non-finite trust signal")
        if solver == "rf_er_sde_trust":
            raise ValueError("rf_er_sde_trust produced a non-finite prediction error")
        raise ValueError("rf_er_sde_warp_trust produced a non-finite prediction error")
    result = x.to(dtype=initial.dtype)
    if return_info:
        return result, SamplingInfo(solver, times.numel() - 1, evaluations)
    return result
