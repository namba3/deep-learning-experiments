from typing import Any

import pytest

import torch

from came_pytorch import CAME as ReferenceCAME

from optimizers.came import CAME, CAMEAutoSchedule

def _assign_gradient(parameter, value=1.0):
    parameter.grad = torch.full_like(parameter, value)

@pytest.mark.parametrize("optimizer_type", [CAMEAutoSchedule])
def test_came_auto_schedule_runs_one_bfloat16_step(optimizer_type):
    parameter = torch.nn.Parameter(torch.randn(4, 3, dtype=torch.bfloat16))
    optimizer = optimizer_type(
        [parameter],
        lr=0.01,
        weight_decay=0.0,
        auto_schedule_warmup_steps=0,
    )
    _assign_gradient(parameter)
    optimizer.step()

    assert torch.isfinite(parameter).all()
    assert optimizer.param_groups[0]["_auto_schedule_step"] == 1

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_local_came_matches_reference_for_one_step(dtype):
    torch.manual_seed(123)
    initial = torch.randn(4, 3, dtype=dtype)
    gradient = torch.randn_like(initial)
    local_parameter = torch.nn.Parameter(initial.clone())
    reference_parameter = torch.nn.Parameter(initial.clone())
    kwargs: dict[str, Any] = dict(
        lr=0.01, eps=(1e-30, 1e-16), clip_threshold=1.0,
        betas=(0.9, 0.999, 0.9999), weight_decay=0.01,
    )
    local = CAME([local_parameter], **kwargs)
    reference = ReferenceCAME([reference_parameter], **kwargs)
    local_parameter.grad = gradient.clone()
    reference_parameter.grad = gradient.clone()
    local.step()
    reference.step()
    assert torch.allclose(
        local_parameter.float(), reference_parameter.float(),
        # Parameter-dtype moments intentionally differ from the
        # reference's FP32 moments for BF16 parameters.
        atol=2e-3 if dtype is torch.bfloat16 else 1e-6,
        rtol=2e-3 if dtype is torch.bfloat16 else 1e-6,
    )

def test_came_factored_step_matches_manual_row_col_and_residual_update():
    initial = torch.tensor([[0.5, -0.25], [0.75, -1.0]])
    gradient = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    parameter = torch.nn.Parameter(initial.clone())
    beta1, beta2, beta3 = 0.9, 0.99, 0.9
    eps_square, eps_instability = 1e-30, 1e-16
    learning_rate, weight_decay = 0.01, 0.1
    optimizer = CAME(
        [parameter],
        lr=learning_rate,
        eps=(eps_square, eps_instability),
        clip_threshold=1.0,
        betas=(beta1, beta2, beta3),
        weight_decay=weight_decay,
        backend="torch",
    )
    parameter.grad = gradient.clone()
    optimizer.step()

    def approx_sq_grad(row, col):
        row_factor = (
            row / row.mean(dim=-1, keepdim=True).clamp_min(1e-30)
        ).rsqrt().unsqueeze(-1)
        col_factor = col.clamp_min(1e-30).rsqrt().unsqueeze(-2)
        return row_factor * col_factor

    second_moment = gradient.square() + eps_square
    expected_row = (1.0 - beta2) * second_moment.mean(dim=-1)
    expected_col = (1.0 - beta2) * second_moment.mean(dim=-2)
    normalized = approx_sq_grad(expected_row, expected_col) * gradient
    normalized = normalized / normalized.square().mean().sqrt().clamp_min(1.0)
    expected_exp_avg = (1.0 - beta1) * normalized
    residual = (normalized - expected_exp_avg).square() + eps_instability
    expected_res_row = (1.0 - beta3) * residual.mean(dim=-1)
    expected_res_col = (1.0 - beta3) * residual.mean(dim=-2)
    expected_update = (
        approx_sq_grad(expected_res_row, expected_res_col) * expected_exp_avg
    )
    expected_parameter = (
        initial * (1.0 - learning_rate * weight_decay)
        - learning_rate * expected_update
    )

    state = optimizer.state[parameter]
    assert torch.allclose(parameter, expected_parameter, atol=1e-6, rtol=1e-6)
    assert torch.allclose(state["exp_avg"], expected_exp_avg, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        state["exp_avg_sq_row"], expected_row, atol=1e-6, rtol=1e-6,
    )
    assert torch.allclose(
        state["exp_avg_sq_col"], expected_col, atol=1e-6, rtol=1e-6,
    )
    assert torch.allclose(
        state["exp_avg_res_row"], expected_res_row, atol=1e-6, rtol=1e-6,
    )
    assert torch.allclose(
        state["exp_avg_res_col"], expected_res_col, atol=1e-6, rtol=1e-6,
    )
