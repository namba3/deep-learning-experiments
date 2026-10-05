from contextlib import contextmanager

import pytest

import torch

from optimizers.apollo import APOLLO, APOLLOAutoSchedule, APOLLOCAME, APOLLOCAMEAutoSchedule, APOLLOMini

def _assign_gradient(parameter, value=1.0):
    parameter.grad = torch.full_like(parameter, value)

@pytest.mark.parametrize("matrix_shape", [(4, 3), (3, 4)])
def test_apollo_project_into_matches_matmul(matrix_shape):
    rows, cols = matrix_shape
    rank = 2
    matrix = torch.randn(rows, cols)
    projection_shape = (cols, rank) if rows >= cols else (rank, rows)
    projection = torch.randn(projection_shape)
    expected = (
        matrix.matmul(projection)
        if rows >= cols
        else projection.matmul(matrix)
    )
    output = torch.empty_like(expected)

    actual = APOLLO._project_into(matrix, projection, output)

    assert actual is output
    assert torch.equal(actual, expected)

@pytest.mark.parametrize(
    "optimizer_type",
    [APOLLO, APOLLOCAME, APOLLOAutoSchedule, APOLLOCAMEAutoSchedule],
)
def test_apollo_family_defaults_to_rank_eight(optimizer_type):
    parameter = torch.nn.Parameter(torch.randn(4, 4))
    optimizer = optimizer_type([parameter])

    assert optimizer.param_groups[0]["rank"] == 8

def test_apollo_mini_remains_rank_one():
    parameter = torch.nn.Parameter(torch.randn(4, 4))
    optimizer = APOLLOMini([parameter], rank=8)

    assert optimizer.param_groups[0]["rank"] == 1

@pytest.mark.parametrize("optimizer_type", [APOLLO, APOLLOCAME])
@pytest.mark.parametrize(
    ("parameter_shape", "requested_rank"),
    [
        ((4, 3), 1),
        ((4, 3), 2),
        ((3, 4), 1),
        ((3, 4), 2),
        ((2, 2, 3), 1),
        ((2, 2, 3), 8),
    ],
)
def test_apollo_low_rank_state_shapes_cover_orientation_rank_and_flattening(
    optimizer_type, parameter_shape, requested_rank,
):
    parameter = torch.nn.Parameter(torch.randn(*parameter_shape, dtype=torch.bfloat16))
    optimizer = optimizer_type(
        [parameter],
        lr=0.01,
        rank=requested_rank,
        weight_decay=0.0,
        update_proj_gap=100,
        fallback={"small_matrix": "apollo"},
    )
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()

    rows = parameter.shape[0]
    cols = parameter.numel() // rows
    rank = min(requested_rank, rows, cols)
    state = optimizer.state[parameter]
    expected_projection_shape = (cols, rank) if rows >= cols else (rank, rows)
    expected_low_rank_shape = (rows, rank) if rows >= cols else (rank, cols)

    assert tuple(state["projection"].shape) == expected_projection_shape
    assert tuple(state["exp_avg"].shape) == expected_low_rank_shape
    assert torch.isfinite(state["projection"]).all()
    assert torch.isfinite(state["exp_avg"]).all()

@pytest.mark.parametrize("optimizer_type", [APOLLO, APOLLOMini, APOLLOCAME])
def test_apollo_family_updates_bfloat16_matrix_and_keeps_state(optimizer_type):
    parameter = torch.nn.Parameter(torch.randn(4, 3, dtype=torch.bfloat16))
    kwargs = {
        "lr": 0.01,
        "rank": 1,
        "update_proj_gap": 2,
        "weight_decay": 0.0,
        "norm_growth_limiter": True,
        "fallback": {"small_matrix": "apollo"},
    }
    optimizer = optimizer_type([parameter], **kwargs)

    for _ in range(2):
        _assign_gradient(parameter)
        optimizer.step()

    state = optimizer.state[parameter]
    assert parameter.dtype == torch.bfloat16
    assert state["projection"].dtype == torch.float32
    assert state["scaled_grad_norm"].dtype == torch.float32
    assert all(
        torch.isfinite(value).all()
        for value in state.values()
        if torch.is_tensor(value)
    )

def test_apollo_performance_hook_reports_parameter_shape_metrics():
    class PerformanceSink:
        def __init__(self):
            self.metrics = {}

        def add_metric(self, name, value=1):
            self.metrics[name] = self.metrics.get(name, 0) + value

        @contextmanager
        def measure(self, name):
            yield

    parameter = torch.nn.Parameter(torch.randn(4, 3, dtype=torch.bfloat16))
    optimizer = APOLLO(
        [parameter], lr=0.01, update_proj_gap=2,
        fallback={"small_matrix": "apollo"},
    )
    _assign_gradient(parameter)
    performance = PerformanceSink()
    optimizer.step_with_performance(performance)

    assert performance.metrics["apollo_parameter_tensors"] == 1
    assert performance.metrics["apollo_matrix_parameters"] == 1
    assert performance.metrics["apollo_gradient_elements"] == parameter.numel()
    # The public default rank is now 8; the 4x3 matrix clamps it to 3, so
    # the projection has shape (3, 3) and contains 9 elements.
    assert performance.metrics["apollo_projection_elements"] == 9

def test_apollo_weight_decay_is_applied_once():
    parameter = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16))
    optimizer = APOLLO(
        [parameter],
        lr=0.1,
        weight_decay=0.1,
        fallback="sgd",
    )
    _assign_gradient(parameter, value=0.0)

    optimizer.step()

    assert torch.allclose(
        parameter.float(),
        torch.full_like(parameter.float(), 0.99),
        atol=2e-3,
        rtol=0.0,
    )

@pytest.mark.parametrize(
    ("target_update_ratio", "expected_multiplier"),
    [(0.01, 0.5), (0.2, 2.0)],
)
def test_auto_schedule_uses_target_and_per_step_factor_limits(
    target_update_ratio, expected_multiplier,
):
    parameter = torch.nn.Parameter(torch.randn(4, 3))
    optimizer = APOLLOAutoSchedule(
        [parameter],
        lr=0.01,
        weight_decay=0.0,
        auto_schedule_warmup_steps=0,
        auto_schedule_target_update_ratio=target_update_ratio,
        auto_schedule_trust_alpha=1.0,
        auto_schedule_gain=1.0,
        auto_schedule_controller_rate=1.0,
        auto_schedule_max_increase=2.0,
        auto_schedule_max_decrease=0.5,
    )
    group = optimizer.param_groups[0]
    group["_auto_schedule_step"] = 1
    group["_auto_schedule_has_multiscale_ema"] = True
    group["_auto_schedule_fast_ema"] = 0.1
    group["_auto_schedule_slow_ema"] = 0.1
    group["_auto_schedule_ema_ratio"] = 0.1
    stats = optimizer._auto_schedule_new_stats(parameter)
    stats["parameter_norm_sq"].fill_(1.0)
    stats["update_norm_sq"].fill_(0.01)

    optimizer._auto_schedule_finish_group(group, stats)

    assert group["_auto_schedule_multiplier"] == expected_multiplier
