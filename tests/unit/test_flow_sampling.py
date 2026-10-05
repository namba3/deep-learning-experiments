import math
from typing import Any

import pytest
import torch

from flow_sampling import (
    available_schedulers,
    available_solvers,
    build_warped_timesteps,
    build_timesteps,
    flow_match_timesteps,
    sample,
)


def test_schedule_separates_uniform_and_static_flow_shift():
    uniform = build_timesteps(4, scheduler="uniform")
    shifted = flow_match_timesteps(4, shift=2.0)

    assert uniform.shape == shifted.shape == (5,)
    assert torch.allclose(uniform, torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]))
    expected = 2.0 * uniform / (1.0 + uniform)
    assert torch.allclose(shifted, expected)
    assert shifted[0] == 1.0
    assert shifted[-1] == 0.0
    assert torch.all(shifted[:-1] > shifted[1:])
    assert available_schedulers() == ("uniform", "flow_match_euler")
    assert available_solvers() == (
        "euler", "fireflow", "abm2", "er_sde", "rf_ab2", "rf_2m_warp",
        "rf_trust_region", "rf_er_sde_1", "rf_er_sde_2m",
        "rf_er_sde_trust", "rf_er_sde_warp_1", "rf_er_sde_warp_2m",
        "rf_er_sde_warp_trust",
    )


def test_schedule_supports_finite_partial_start_and_fp32_construction():
    times = build_timesteps(
        4, scheduler="flow_match_euler", start_t=0.9, shift=3.0,
        dtype=torch.bfloat16,
    )

    assert times.dtype == torch.bfloat16
    assert times.shape == (5,)
    assert float(times[0].float().item()) == pytest.approx(3.0 * 0.9 / (1.0 + 2.0 * 0.9), abs=0.01)
    assert times[-1] == 0
    assert torch.all(times[:-1] > times[1:])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_steps": 0}, "num_steps"),
        ({"num_steps": 2, "scheduler": "bad"}, "scheduler"),
        ({"num_steps": 2, "shift": float("nan")}, "shift"),
        ({"num_steps": 2, "shift": 0.0}, "shift"),
        ({"num_steps": 2, "start_t": float("inf")}, "start_t"),
        ({"num_steps": 2, "dtype": torch.int64}, "dtype"),
        ({"num_steps": 2, "scheduler": "uniform", "shift": 2.0}, "only applies"),
    ],
)
def test_schedule_rejects_invalid_configuration(kwargs, match):
    defaults: dict[str, Any] = {"num_steps": 2}
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=match):
        build_timesteps(**defaults)


@pytest.mark.parametrize(
    ("solver", "nfe"),
    [("euler", 8), ("fireflow", 9), ("abm2", 9)],
)
def test_deterministic_solvers_integrate_constant_velocity_exactly(solver, nfe):
    initial = torch.randn(2, 3, 5, 7)
    velocity = torch.full_like(initial, 0.25)
    calls = []

    def field(state, batch_times):
        calls.append((state.shape, batch_times.shape, batch_times.dtype))
        return velocity.to(state.dtype)

    result, info = sample(
        field, initial, build_timesteps(8), solver=solver, return_info=True,
    )

    assert torch.allclose(result, initial - 0.25, atol=1e-6, rtol=0)
    assert info.num_steps == 8
    assert info.num_model_evaluations == nfe
    assert len(calls) == nfe
    assert all(shape == initial.shape for shape, _, _ in calls)
    assert all(batch_shape == (initial.shape[0],) for _, batch_shape, _ in calls)
    assert all(dtype == torch.float32 for _, _, dtype in calls)


@pytest.mark.parametrize("solver", ["euler", "fireflow", "abm2"])
@pytest.mark.parametrize("shift", [1.0, 3.0])
def test_fixed_grid_solver_errors_decrease_on_linear_ode(solver, shift):
    errors = []
    for steps in (8, 16, 32):
        def field(state, batch_times):
            return state

        result = sample(
            field, torch.ones(1, 1), flow_match_timesteps(steps, shift=shift),
            solver=solver,
        )
        errors.append(abs(float(result.item()) - math.exp(-1.0)))

    assert errors[2] < errors[1] < errors[0]
    threshold = 1.8 if solver == "euler" else 3.0
    assert errors[0] / errors[1] > threshold
    assert errors[1] / errors[2] > threshold


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("solver", ["euler", "fireflow", "abm2"])
def test_solver_preserves_latent_dtype_and_uses_fp32_time_and_accumulation(dtype, solver):
    initial = torch.ones(2, 3, 4, 6, dtype=dtype)
    calls = []

    def field(state, batch_times):
        calls.append((state.dtype, batch_times.dtype))
        return torch.full_like(state, 0.125)

    result, info = sample(
        field, initial, build_timesteps(4, dtype=torch.float32),
        solver=solver, return_info=True,
    )

    assert result.dtype == dtype
    assert torch.isfinite(result).all()
    assert torch.allclose(result.float(), torch.full_like(initial.float(), 0.875), atol=0.01)
    assert info.num_model_evaluations == len(calls)
    assert all(state_dtype == dtype for state_dtype, _ in calls)
    assert all(time_dtype == torch.float32 for _, time_dtype in calls)


def test_solver_rejects_increasing_and_repeated_times():
    initial = torch.ones(1, 2)
    def field(x, t):
        return torch.ones_like(x)

    for times in (torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.5, 0.5])):
        with pytest.raises(ValueError, match="strictly descending"):
            sample(field, initial, times)


def test_er_sde_is_seeded_and_counts_one_velocity_call_per_interval():
    initial = torch.ones(2, 3, 4, 5)
    times = build_timesteps(5, start_t=80.0 / 81.0)
    calls = []

    def field(state, batch_times):
        calls.append(batch_times)
        return torch.zeros_like(state)

    first, info = sample(
        field, initial, times, solver="er_sde",
        generator=torch.Generator().manual_seed(42), return_info=True,
    )
    second = sample(
        field, initial, times, solver="er_sde",
        generator=torch.Generator().manual_seed(42),
    )

    assert torch.equal(first, second)
    assert info.num_model_evaluations == 5
    assert len(calls) == 10
    assert torch.isfinite(first).all()


def test_er_sde_requires_a_finite_starting_time():
    with pytest.raises(ValueError, match="starting timestep below 1"):
        sample(
            lambda x, t: torch.zeros_like(x),
            torch.ones(1, 2),
            build_timesteps(2),
            solver="er_sde",
        )


@pytest.mark.parametrize(
    ("solver", "warped"),
    [
        ("rf_ab2", False),
        ("rf_trust_region", False),
        ("rf_2m_warp", True),
        ("rf_er_sde_1", False),
        ("rf_er_sde_2m", False),
        ("rf_er_sde_trust", False),
        ("rf_er_sde_warp_1", True),
        ("rf_er_sde_warp_2m", True),
        ("rf_er_sde_warp_trust", True),
    ],
)
def test_rf_solvers_integrate_constant_velocity_on_identity_grid(solver, warped):
    initial = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    velocity = torch.full_like(initial, 0.25)
    calls = []

    def field(state, batch_times):
        calls.append((state.shape, batch_times.shape))
        return velocity

    if warped:
        grid = build_warped_timesteps(8, warp="identity")
        times = grid.model_timesteps
        warp_kwargs = {
            "solver_times": grid.solver_times,
            "warp_derivatives": grid.derivatives,
        }
    else:
        times = build_timesteps(8, scheduler="uniform")
        warp_kwargs = {}

    result, info = sample(
        field,
        initial,
        times,
        solver=solver,
        rf_er_sde_eta=0.0,
        generator=torch.Generator().manual_seed(17),
        return_info=True,
        **warp_kwargs,
    )

    assert torch.allclose(result, initial - 0.25, atol=1e-6, rtol=0)
    assert info.num_model_evaluations == 8
    assert len(calls) == 8
    assert all(shape == initial.shape for shape, _ in calls)
    assert all(batch_shape == (initial.shape[0],) for _, batch_shape in calls)


@pytest.mark.parametrize(
    "solver",
    ["rf_2m_warp", "rf_er_sde_warp_1", "rf_er_sde_warp_2m", "rf_er_sde_warp_trust"],
)
def test_warped_rf_first_step_scales_by_derivative(solver):
    initial = torch.tensor([[1.0]])
    velocity = torch.tensor([[0.4]])
    grid = build_warped_timesteps(1, warp="rational", shift=2.0)

    result = sample(
        lambda state, batch_times: velocity,
        initial,
        grid.model_timesteps,
        solver=solver,
        solver_times=grid.solver_times,
        warp_derivatives=grid.derivatives,
        rf_er_sde_eta=0.0,
    )

    # For one interval, Δtau=1 and q(0)=2, so Δx=Δtau*(-u/q)=-0.2.
    assert torch.allclose(result, torch.tensor([[0.8]]), atol=1e-6, rtol=0)


def _reference_rf_trust_region(
    field, initial, timesteps, trust_lambda, *, clamp_velocity_rms=False,
):
    """Small reference for the documented gated variable-step AB2 update."""
    state = initial.clone()
    previous_velocity = None
    previous_previous_velocity = None
    previous_step = None
    previous_previous_step = None
    for index in range(timesteps.numel() - 1):
        batch_time = timesteps[index].expand(state.shape[0])
        sampling_velocity = -field(state, batch_time)
        step = timesteps[index] - timesteps[index + 1]
        if previous_velocity is None:
            effective_velocity = sampling_velocity
        else:
            correction = 0.5 * (step / previous_step) * (
                sampling_velocity - previous_velocity
            )
            if previous_previous_velocity is None:
                trust = torch.ones(state.shape[0], dtype=state.dtype)
            else:
                assert previous_previous_step is not None
                predicted_velocity = previous_velocity + (
                    previous_step / previous_previous_step
                ) * (previous_velocity - previous_previous_velocity)
                feature_axes = tuple(range(1, state.ndim))
                prediction_rms = torch.sqrt(torch.mean(
                    (sampling_velocity - predicted_velocity).square(), dim=feature_axes,
                ))
                velocity_rms = torch.sqrt(torch.mean(
                    sampling_velocity.square(), dim=feature_axes,
                ))
                denominator = (
                    velocity_rms.clamp_min(1e-8)
                    if clamp_velocity_rms
                    else velocity_rms + 1e-8
                )
                prediction_error = prediction_rms / denominator
                trust = torch.exp(-trust_lambda * prediction_error)
            effective_velocity = sampling_velocity + trust.reshape(
                (state.shape[0],) + (1,) * (state.ndim - 1)
            ) * correction
        state = state + step * effective_velocity
        previous_previous_velocity = previous_velocity
        previous_velocity = sampling_velocity
        previous_previous_step = previous_step
        previous_step = step
    return state


def test_rf_trust_solvers_match_hand_reference_and_eta_zero_degeneration():
    initial = torch.tensor([[0.7, -0.3], [1.2, 0.5]], dtype=torch.float64)
    times = torch.tensor([1.0, 0.72, 0.31, 0.0], dtype=torch.float64)

    def field(state, batch_times):
        return 0.3 * state + batch_times[:, None].square()

    expected_trust_region = _reference_rf_trust_region(
        field, initial, times, trust_lambda=2.5,
    )
    expected_sde_trust = _reference_rf_trust_region(
        field, initial, times, trust_lambda=2.5, clamp_velocity_rms=True,
    )
    deterministic = sample(
        field, initial, times, solver="rf_trust_region", rf_trust_lambda=2.5,
    )
    stochastic_eta_zero = sample(
        field,
        initial,
        times,
        solver="rf_er_sde_trust",
        rf_er_sde_eta=0.0,
        rf_er_sde_trust_lambda=2.5,
    )

    assert torch.allclose(deterministic, expected_trust_region, atol=1e-12, rtol=1e-12)
    assert torch.allclose(stochastic_eta_zero, expected_sde_trust, atol=1e-12, rtol=1e-12)


def test_rf_er_sde_uses_score_drift_and_seeded_noise_on_first_stochastic_step():
    initial = torch.tensor([[1.0]])
    velocity = torch.tensor([[0.2]])
    times = torch.tensor([0.75, 0.5])
    eta = 0.2
    seed = 17

    result = sample(
        lambda state, batch_times: velocity,
        initial,
        times,
        solver="rf_er_sde_1",
        rf_er_sde_eta=eta,
        generator=torch.Generator().manual_seed(seed),
    )

    step = times[0] - times[1]
    progress = 1.0 - times[0]
    diffusion = eta * torch.sin(torch.pi * progress)
    score = -(initial + progress * velocity) / times[0]
    drift = -velocity + 0.5 * diffusion.square() * score
    noise = torch.randn(initial.shape, generator=torch.Generator().manual_seed(seed))
    expected = initial + step * drift + diffusion * torch.sqrt(step) * noise

    assert torch.allclose(result, expected, atol=1e-6, rtol=1e-6)
