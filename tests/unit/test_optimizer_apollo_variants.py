import math
from typing import Any

import pytest

import torch

from optimizers.apollo import APOLLO, APOLLOADAMW, APOLLOADAMWAutoSchedule, APOLLOLion, DualRotAPOLLO, RotAPOLLO, APOLLOCAME, APOLLOCAMEAutoSchedule

def _assign_gradient(parameter, value=1.0):
    parameter.grad = torch.full_like(parameter, value)

def test_rot_apollo_handles_rows_less_than_columns_and_keeps_rows_orthonormal():
    parameter = torch.nn.Parameter(torch.randn(2, 2, 3, dtype=torch.bfloat16))
    optimizer = RotAPOLLO(
        [parameter],
        lr=0.01,
        rank=2,
        weight_decay=0.0,
        rotation_frequency=2,
        rotation_rate=0.1,
        exploration_ratio=0.2,
        update_proj_gap=100,
        fallback={"small_matrix": "apollo"},
    )
    for _ in range(3):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

    state = optimizer.state[parameter]
    projection = state["projection"]
    assert projection.shape == (2, 2)
    assert torch.allclose(
        projection.mm(projection.t()),
        torch.eye(2),
        atol=2e-4,
        rtol=0.0,
    )
    assert state["exp_avg"].shape == (2, 6)
    assert torch.isfinite(parameter).all()

def test_apollo_came_foreach_vector_path_updates_multiple_vectors():
    parameters = [
        torch.nn.Parameter(torch.randn(8, dtype=torch.bfloat16))
        for _ in range(3)
    ]
    optimizer = APOLLOCAME(
        parameters, lr=0.01, weight_decay=0.01, norm_growth_limiter=True,
        fallback="came",
    )

    for _ in range(2):
        for parameter in parameters:
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()

    for parameter in parameters:
        state = optimizer.state[parameter]
        assert "fallback_exp_avg" in state
        assert "fallback_exp_avg_sq" in state
        assert "fallback_update" not in state
        assert "scaled_grad_norm" in state
        assert torch.isfinite(parameter).all()
        assert all(
            not torch.is_tensor(value) or torch.isfinite(value).all()
            for value in state.values()
        )

def test_apollo_came_foreach_vector_path_matches_individual_groups():
    initial = [torch.randn(8, dtype=torch.bfloat16) for _ in range(2)]
    foreach_parameters = [torch.nn.Parameter(value.clone()) for value in initial]
    individual_parameters = [
        torch.nn.Parameter(value.clone()) for value in initial
    ]
    common: dict[str, Any] = dict(
        lr=0.01,
        scale=1.21,
        scale_front=True,
        weight_decay=0.01,
        norm_growth_limiter=True,
        fallback="came",
    )
    foreach_optimizer = APOLLOCAME(foreach_parameters, **common)
    individual_optimizer = APOLLOCAME(
        [{"params": [parameter]} for parameter in individual_parameters],
        **common,
    )

    for step in range(2):
        gradients = [
            torch.full_like(parameter, 0.5 + step)
            for parameter in foreach_parameters
        ]
        for foreach_parameter, individual_parameter, gradient in zip(
            foreach_parameters, individual_parameters, gradients
        ):
            foreach_parameter.grad = gradient.clone()
            individual_parameter.grad = gradient.clone()
        foreach_optimizer.step()
        individual_optimizer.step()

    for foreach_parameter, individual_parameter in zip(
        foreach_parameters, individual_parameters
    ):
        torch.testing.assert_close(
            foreach_parameter,
            individual_parameter,
            atol=2e-3,
            rtol=0.0,
        )

def test_apollo_came_fallback_matches_reference_update():
    parameter = torch.nn.Parameter(torch.tensor([0.5, -0.25, 0.75]))
    gradient = torch.tensor([1.0, -2.0, 0.5])
    optimizer = APOLLO(
        [parameter], lr=0.01, weight_decay=0.0, fallback="came",
    )
    parameter.grad = gradient.clone()
    before = parameter.detach().clone()

    optimizer.step()

    state = optimizer.state[parameter]
    assert state["fallback_exp_avg"].dtype == parameter.dtype
    assert state["fallback_exp_avg"].shape == parameter.shape
    # Ordinary APOLLO applies only scalar factors after the fallback update,
    # so it can use the EMA state directly without a full-size work copy.
    assert "fallback_update" not in state

    beta1, beta2 = (0.9, 0.999)
    eps_square = 1e-30
    update = gradient.square().add(eps_square)
    exp_avg_sq = update * (1.0 - beta2)
    update = exp_avg_sq.rsqrt() * gradient
    update = update / update.square().mean().sqrt().clamp_min(1.0)
    exp_avg = update * (1.0 - beta1)
    expected = before - 0.01 * exp_avg
    assert torch.allclose(parameter, expected, atol=1e-7, rtol=1e-6)

def test_apollo_adamw_uses_existing_apollo_matrix_path_and_came_fallback():
    matrix_a = torch.nn.Parameter(torch.randn(4, 3, dtype=torch.bfloat16))
    matrix_b = torch.nn.Parameter(matrix_a.detach().clone())
    vector = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    optimizer = APOLLOADAMW(
        [matrix_a, vector], lr=0.01, weight_decay=0.0,
        update_proj_gap=100,
        fallback={"one_dimensional": "came", "small_matrix": "apollo"},
    )
    reference = APOLLO(
        [matrix_b], lr=0.01, weight_decay=0.0, update_proj_gap=100,
        fallback={"small_matrix": "apollo"},
    )
    matrix_a.grad = torch.randn_like(matrix_a)
    matrix_b.grad = matrix_a.grad.detach().clone()
    vector.grad = torch.ones_like(vector)
    optimizer.step()
    reference.step()

    assert torch.equal(matrix_a, matrix_b)
    assert "fallback_exp_avg" in optimizer.state[vector]
    assert "fallback_exp_avg_sq" in optimizer.state[vector]

def test_apollo_adamw_auto_schedule_runs_one_step():
    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    optimizer = APOLLOADAMWAutoSchedule(
        [parameter], lr=0.01, weight_decay=0.0,
        auto_schedule_warmup_steps=0,
    )
    _assign_gradient(parameter)
    optimizer.step()
    assert torch.isfinite(parameter).all()
    assert optimizer.param_groups[0]["_auto_schedule_step"] == 1

def test_apollo_came_auto_schedule_runs_and_updates_controller():
    parameter = torch.nn.Parameter(torch.randn(4, 3, dtype=torch.bfloat16))
    optimizer = APOLLOCAMEAutoSchedule(
        [parameter],
        lr=0.01,
        weight_decay=0.0,
        came_backend="torch",
        auto_schedule_warmup_steps=0,
        fallback={"small_matrix": "apollo"},
    )

    for _ in range(2):
        _assign_gradient(parameter)
        optimizer.step()

    group = optimizer.param_groups[0]
    assert group["_auto_schedule_step"] == 2
    assert group["_auto_schedule_has_ratio"]
    assert group["_auto_schedule_has_multiscale_ema"]
    assert group["_auto_schedule_fast_ema"] > 0.0
    assert group["_auto_schedule_slow_ema"] > 0.0
    assert group["_auto_schedule_has_confidence"]
    assert math.isfinite(group["lr"])
    assert torch.isfinite(parameter).all()

def test_apollo_came_confidence_stats_are_weighted_by_state_elements():
    parameter = torch.nn.Parameter(torch.randn(4, 3))
    optimizer = APOLLOCAMEAutoSchedule(
        [parameter], lr=0.01, weight_decay=0.0, came_backend="torch",
    )
    stats = optimizer._auto_schedule_new_stats(parameter)
    optimizer._add_auto_schedule_low_rank_stats(
        stats,
        {
            "exp_avg": torch.ones(1, 1),
            "exp_avg_res_row": torch.zeros(1),
            "exp_avg_res_col": torch.zeros(1),
        },
    )
    optimizer._add_auto_schedule_low_rank_stats(
        stats,
        {
            "exp_avg": torch.ones(2, 2),
            "exp_avg_res_row": torch.ones(2),
            "exp_avg_res_col": torch.ones(2),
        },
    )

    assert stats["moment_norm_sq"].item() == pytest.approx(5.0)
    assert stats["noise_norm_sq"].item() == pytest.approx(4.0)

def test_apollo_lion_uses_one_low_rank_momentum_and_came_fallback():
    matrix = torch.nn.Parameter(torch.randn(4, 3, dtype=torch.bfloat16))
    vector = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    optimizer = APOLLOLion(
        [matrix, vector], lr=0.01, weight_decay=0.0,
        update_proj_gap=100,
        fallback={"one_dimensional": "came", "small_matrix": "apollo"},
    )
    matrix.grad = torch.randn_like(matrix)
    vector.grad = torch.ones_like(vector)
    optimizer.step()

    matrix_state = optimizer.state[matrix]
    assert "exp_avg" in matrix_state
    assert "exp_avg_sq" not in matrix_state
    assert matrix_state["exp_avg"].dtype == torch.float32
    assert "fallback_exp_avg" in optimizer.state[vector]
    assert torch.isfinite(matrix).all()
    assert torch.isfinite(vector).all()

def test_rot_apollo_rotates_orthonormal_projection_and_transports_state():
    parameter = torch.nn.Parameter(torch.randn(6, 4, dtype=torch.bfloat16))
    optimizer = RotAPOLLO(
        [parameter], lr=0.01, weight_decay=0.0,
        rotation_frequency=2, rotation_rate=0.1,
        exploration_ratio=0.2, update_proj_gap=100,
        fallback={"small_matrix": "apollo"},
    )
    initial_projection = None
    for _ in range(3):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        if initial_projection is None:
            initial_projection = optimizer.state[parameter]["projection"].clone()

    state = optimizer.state[parameter]
    projection = state["projection"]
    assert torch.allclose(
        projection.t().mm(projection),
        torch.eye(projection.shape[1]), atol=2e-4, rtol=0.0,
    )
    assert not torch.allclose(projection, initial_projection)
    assert torch.isfinite(parameter).all()
    assert torch.isfinite(state["exp_avg"]).all()

def test_dual_rot_apollo_keeps_two_branches_and_rotates_stable_branch():
    parameter = torch.nn.Parameter(torch.randn(6, 4, dtype=torch.bfloat16))
    optimizer = DualRotAPOLLO(
        [parameter], lr=0.01, weight_decay=0.0,
        rotation_frequency=2, rotation_rate=0.1,
        exploration_ratio=0.2, update_proj_gap=100,
        fallback={"small_matrix": "apollo"},
    )
    projections = None
    for _ in range(3):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        if projections is None:
            projections = [
                branch["projection"].clone()
                for branch in optimizer.state[parameter]["branches"]
            ]

    branches = optimizer.state[parameter]["branches"]
    assert len(branches) == 2
    assert projections is not None
    for branch in branches:
        projection = branch["projection"]
        assert torch.allclose(
            projection.t().mm(projection),
            torch.eye(projection.shape[1]), atol=2e-4, rtol=0.0,
        )
        assert torch.isfinite(branch["exp_avg"]).all()
    assert any(
        not torch.allclose(branch["projection"], initial)
        for branch, initial in zip(branches, projections)
    )
    assert torch.isfinite(parameter).all()

def test_rot_apollo_transports_low_rank_state_by_projection_overlap():
    angle = 0.6
    old_projection = torch.tensor([[1.0], [0.0]])
    new_projection = torch.tensor([[torch.cos(torch.tensor(angle))], [
        torch.sin(torch.tensor(angle)),
    ]])
    state = {
        "exp_avg": torch.tensor([[1.0], [2.0]]),
        "exp_avg_sq": torch.tensor([[3.0], [4.0]]),
    }
    expected_overlap = old_projection.t().mm(new_projection)
    expected_exp_avg = state["exp_avg"].mm(expected_overlap)
    expected_exp_avg_sq = state["exp_avg_sq"].mm(expected_overlap.square())

    RotAPOLLO._transport_state(
        state, old_projection, new_projection, rows_ge_cols=True,
    )

    assert torch.allclose(state["exp_avg"], expected_exp_avg)
    assert torch.allclose(state["exp_avg_sq"], expected_exp_avg_sq)

def test_dual_rot_apollo_first_step_matches_manual_equal_branch_mixture():
    initial = torch.tensor([[0.5, -0.25], [0.75, -1.0]])
    gradient = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    parameter = torch.nn.Parameter(initial.clone())
    beta1, beta2 = 0.7, 0.8
    learning_rate, weight_decay, eps = 0.01, 0.1, 1e-8
    optimizer = DualRotAPOLLO(
        [parameter],
        lr=learning_rate,
        rank=1,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
        norm_growth_limiter=False,
        seed=123,
    )
    parameter.grad = gradient.clone()
    optimizer.step()

    state = optimizer.state[parameter]
    expected_scalings = []
    for branch in state["branches"]:
        projection = branch["projection"].float()
        low_rank_gradient = gradient.mm(projection)
        expected_exp_avg = (1.0 - beta1) * low_rank_gradient
        expected_exp_avg_sq = (1.0 - beta2) * low_rank_gradient.square()
        normalized = (
            expected_exp_avg / (1.0 - beta1)
            / ((expected_exp_avg_sq / (1.0 - beta2)).sqrt() + eps)
        )
        expected_scalings.append(
            normalized.norm(dim=1)
            / (low_rank_gradient.norm(dim=1) + eps)
        )
        assert torch.allclose(
            branch["exp_avg"], expected_exp_avg, atol=1e-6, rtol=1e-6,
        )
        assert torch.allclose(
            branch["exp_avg_sq"], expected_exp_avg_sq, atol=1e-6, rtol=1e-6,
        )

    expected_scaling = torch.stack(expected_scalings).mean(dim=0)
    expected_parameter = (
        initial * (1.0 - learning_rate * weight_decay)
        - learning_rate * gradient * expected_scaling.reshape(-1, 1)
    )
    assert torch.allclose(parameter, expected_parameter, atol=1e-6, rtol=1e-6)

def test_apollo_fallback_rms_without_square_buffer_matches_reference():
    tensor = torch.tensor([-2.0, 1.0, 3.0, -4.0])
    expected = tensor.square().mean().sqrt()
    actual = APOLLO._rms_without_square_buffer(tensor)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)

def test_apollo_matrix_step_matches_manual_projected_adam_scaling():
    initial = torch.tensor([[0.5, -0.25], [0.75, -1.0]])
    gradient = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    parameter = torch.nn.Parameter(initial.clone())
    beta1, beta2 = 0.8, 0.9
    learning_rate, weight_decay, eps = 0.01, 0.1, 1e-8
    optimizer = APOLLO(
        [parameter],
        lr=learning_rate,
        rank=1,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
        norm_growth_limiter=False,
        update_proj_gap=100,
        seed=123,
    )
    parameter.grad = gradient.clone()
    optimizer.step()

    state = optimizer.state[parameter]
    projection = state["projection"].float()
    projected_gradient = gradient.mm(projection)
    expected_exp_avg = (1.0 - beta1) * projected_gradient
    expected_exp_avg_sq = (1.0 - beta2) * projected_gradient.square()
    normalized = expected_exp_avg / (expected_exp_avg_sq.sqrt() + eps)
    normalized *= (1.0 - beta2) ** 0.5 / (1.0 - beta1)
    scaling = normalized.norm(dim=1) / (
        projected_gradient.norm(dim=1) + eps
    )
    expected_update = gradient * scaling.reshape(-1, 1)
    expected_parameter = (
        initial * (1.0 - learning_rate * weight_decay)
        - learning_rate * expected_update
    )

    assert torch.allclose(parameter, expected_parameter, atol=1e-6, rtol=1e-6)
    assert torch.allclose(state["exp_avg"], expected_exp_avg, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        state["exp_avg_sq"], expected_exp_avg_sq, atol=1e-6, rtol=1e-6,
    )

def test_apollo_came_matrix_step_matches_manual_projected_came_scaling():
    initial = torch.tensor([[0.5, -0.25], [0.75, -1.0]])
    gradient = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    parameter = torch.nn.Parameter(initial.clone())
    beta1, beta2, beta3 = 0.8, 0.9, 0.95
    eps_square, eps_instability = 1e-30, 1e-16
    learning_rate, weight_decay, eps = 0.01, 0.1, 1e-16
    optimizer = APOLLOCAME(
        [parameter],
        lr=learning_rate,
        rank=1,
        betas=(beta1, beta2, beta3),
        eps=(eps_square, eps_instability),
        clip_threshold=1.0,
        weight_decay=weight_decay,
        norm_growth_limiter=False,
        update_proj_gap=100,
        seed=123,
        came_backend="torch",
        fallback={"small_matrix": "apollo"},
    )
    parameter.grad = gradient.clone()
    optimizer.step()

    def approx_sq_grad(row, col):
        row_factor = (
            row / row.mean(dim=-1, keepdim=True).clamp_min(1e-30)
        ).rsqrt().unsqueeze(-1)
        col_factor = col.clamp_min(1e-30).rsqrt().unsqueeze(-2)
        return row_factor * col_factor

    state = optimizer.state[parameter]
    projection = state["projection"].float()
    projected_gradient = gradient.mm(projection)
    second_moment = projected_gradient.square() + eps_square
    expected_row = (1.0 - beta2) * second_moment.mean(dim=-1)
    expected_col = (1.0 - beta2) * second_moment.mean(dim=-2)
    normalized = approx_sq_grad(expected_row, expected_col) * projected_gradient
    normalized = normalized / normalized.square().mean().sqrt().clamp_min(1.0)
    expected_exp_avg = (1.0 - beta1) * normalized
    residual = (normalized - expected_exp_avg).square() + eps_instability
    expected_res_row = (1.0 - beta3) * residual.mean(dim=-1)
    expected_res_col = (1.0 - beta3) * residual.mean(dim=-2)
    came_update = (
        approx_sq_grad(expected_res_row, expected_res_col) * expected_exp_avg
    )
    scaling = came_update.norm(dim=1) / (
        projected_gradient.norm(dim=1) + eps
    )
    expected_parameter = (
        initial * (1.0 - learning_rate * weight_decay)
        - learning_rate * gradient * scaling.reshape(-1, 1)
    )

    assert torch.allclose(parameter, expected_parameter, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        state["came_low_rank_grad"], projected_gradient, atol=1e-6, rtol=1e-6,
    )
    assert torch.allclose(
        state["exp_avg_sq_row"], expected_row, atol=1e-6, rtol=1e-6,
    )
    assert torch.allclose(
        state["exp_avg_sq_col"], expected_col, atol=1e-6, rtol=1e-6,
    )
    assert torch.allclose(state["exp_avg"], expected_exp_avg, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        state["exp_avg_res_row"], expected_res_row, atol=1e-6, rtol=1e-6,
    )
    assert torch.allclose(
        state["exp_avg_res_col"], expected_res_col, atol=1e-6, rtol=1e-6,
    )

def test_apollo_lion_matrix_step_matches_manual_sign_scaling():
    initial = torch.tensor([[0.5, -0.25], [0.75, -1.0]])
    gradient = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    parameter = torch.nn.Parameter(initial.clone())
    beta1, beta2 = 0.7, 0.8
    learning_rate, weight_decay, eps = 0.01, 0.1, 1e-8
    optimizer = APOLLOLion(
        [parameter],
        lr=learning_rate,
        rank=1,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
        norm_growth_limiter=False,
        update_proj_gap=100,
        seed=123,
        fallback={"small_matrix": "apollo"},
    )
    parameter.grad = gradient.clone()
    optimizer.step()

    state = optimizer.state[parameter]
    projected_gradient = gradient.mm(state["projection"].float())
    expected_exp_avg = (1.0 - beta2) * projected_gradient
    lion_update = ((1.0 - beta1) * projected_gradient).sign()
    scaling = lion_update.norm(dim=1) / (
        projected_gradient.norm(dim=1) + eps
    )
    expected_parameter = (
        initial * (1.0 - learning_rate * weight_decay)
        - learning_rate * gradient * scaling.reshape(-1, 1)
    )

    assert torch.allclose(parameter, expected_parameter, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        state["exp_avg"], expected_exp_avg, atol=1e-6, rtol=1e-6,
    )
    assert "exp_avg_sq" not in state
