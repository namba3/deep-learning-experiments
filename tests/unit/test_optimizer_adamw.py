from typing import Any

import pytest

import torch

from optimizers.adamw import AdamWAutoSchedule, AdamWFP32State

from optimizers.adamw_lr_ema import AdamWLowRankGradientEMA, AdamWLowRankGradientEMAConfidence

from optimizers.adamw_lrsf import AdamWLRSF

from optimizers.adamw_lrsf_lr import AdamWLRSLowRankPreconditioner

from optimizers.adamw_sf_lr import AdamWSFLowRankPreconditioner

from optimizers.lr_scheduler import LearningRateSchedule

from optimizers.muon_variants import AdaMuon, NorMuon

from optimizers.soap import SOAP

from optimizers.schedulefree import AdamWScheduleFree, RAdamScheduleFree

def _assign_gradient(parameter, value=1.0):
    parameter.grad = torch.full_like(parameter, value)

def test_adamw_state_follows_model_dtype_for_full_size_state():
    parameter = torch.nn.Parameter(torch.ones(4, 3, dtype=torch.bfloat16))
    optimizer = AdamWFP32State([parameter], lr=0.01)
    _assign_gradient(parameter)

    before = parameter.detach().clone()
    optimizer.step()

    state = optimizer.state[parameter]
    assert parameter.dtype == torch.bfloat16
    assert state["exp_avg"].dtype == parameter.dtype
    assert state["exp_avg_sq"].dtype == parameter.dtype
    assert not torch.equal(before, parameter)

def test_adamw_lr_ema_uses_low_rank_projected_gradient_state():
    parameter = torch.nn.Parameter(torch.randn(16, 8, dtype=torch.bfloat16))
    optimizer = AdamWLowRankGradientEMA(
        [parameter], lr=0.01, rank=4, ema_beta=0.9, weight_decay=0.0,
    )
    _assign_gradient(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert state["backend"] == "lr_ema"
    assert state["lr_ema_projection"].shape == (8, 4)
    assert state["lr_ema_grad"].shape == (16, 4)
    assert state["lr_ema_projection"].dtype == torch.float32
    assert state["lr_ema_grad"].dtype == torch.float32
    assert torch.isfinite(parameter).all()

def test_adamw_lr_ema_falls_back_to_full_gradient_ema_for_vectors():
    parameter = torch.nn.Parameter(torch.randn(8, dtype=torch.bfloat16))
    optimizer = AdamWLowRankGradientEMA(
        [parameter], lr=0.01, rank=4, ema_beta=0.9, weight_decay=0.0,
    )
    _assign_gradient(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert state["backend"] == "full"
    assert state["lr_ema_grad"].shape == parameter.shape
    assert state["lr_ema_grad"].dtype == parameter.dtype

def test_adamw_lr_ema_supports_wide_matrix_layout():
    parameter = torch.nn.Parameter(torch.randn(8, 16, dtype=torch.bfloat16))
    optimizer = AdamWLowRankGradientEMA(
        [parameter], lr=0.01, rank=4, ema_beta=0.9, weight_decay=0.0,
    )
    _assign_gradient(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert state["backend"] == "lr_ema"
    assert state["lr_ema_projection"].shape == (4, 8)
    assert state["lr_ema_grad"].shape == (4, 16)
    assert torch.isfinite(parameter).all()

def test_adamw_lr_ema_confidence_keeps_latent_innovation_variance():
    parameter = torch.nn.Parameter(torch.randn(16, 8, dtype=torch.bfloat16))
    optimizer = AdamWLowRankGradientEMAConfidence(
        [parameter],
        lr=0.01,
        rank=4,
        ema_beta=0.9,
        confidence_beta=0.99,
        confidence_alpha=1e-3,
        weight_decay=0.0,
    )
    _assign_gradient(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert state["backend"] == "lr_ema_confidence"
    assert state["lr_ema_residual_sq"].shape == (16, 4)
    assert state["lr_ema_residual_sq"].dtype == torch.float32
    assert torch.isfinite(state["lr_ema_residual_sq"]).all()
    assert torch.isfinite(parameter).all()

def test_adamw_matches_manual_two_step_update_order():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = AdamWFP32State(
        [parameter],
        lr=0.1,
        betas=(0.9, 0.99),
        eps=0.01,
        weight_decay=0.2,
        backend="torch",
    )
    beta1, beta2 = 0.9, 0.99
    learning_rate, weight_decay, eps = 0.1, 0.2, 0.01
    first_gradient = torch.tensor([0.5, -1.0])
    second_gradient = torch.tensor([-0.25, 0.5])
    expected_parameter = parameter.detach().clone()
    expected_exp_avg = torch.zeros_like(expected_parameter)
    expected_exp_avg_sq = torch.zeros_like(expected_parameter)

    for step, gradient in enumerate((first_gradient, second_gradient), start=1):
        parameter.grad = gradient.clone()
        optimizer.step()

        expected_exp_avg = (
            expected_exp_avg * beta1 + gradient * (1.0 - beta1)
        )
        expected_exp_avg_sq = (
            expected_exp_avg_sq * beta2 + gradient.square() * (1.0 - beta2)
        )
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        denominator = (
            expected_exp_avg_sq.sqrt() / bias_correction2**0.5
        ) + eps
        expected_parameter = (
            expected_parameter * (1.0 - learning_rate * weight_decay)
            - learning_rate / bias_correction1
            * expected_exp_avg / denominator
        )

        state = optimizer.state[parameter]
        assert torch.allclose(parameter, expected_parameter, atol=1e-6, rtol=1e-6)
        assert torch.allclose(state["exp_avg"], expected_exp_avg, atol=1e-6, rtol=1e-6)
        assert torch.allclose(
            state["exp_avg_sq"], expected_exp_avg_sq, atol=1e-6, rtol=1e-6,
        )

@pytest.mark.parametrize("optimizer_type", [SOAP, NorMuon, AdaMuon])
def test_modern_matrix_optimizers_use_parameter_dtype_for_full_size_state(optimizer_type):
    parameter = torch.nn.Parameter(torch.randn(4, 3, dtype=torch.bfloat16))
    kwargs: dict[str, Any] = {"lr": 0.01, "weight_decay": 0.0}
    if optimizer_type is SOAP:
        kwargs["precondition_frequency"] = 2
    optimizer = optimizer_type([parameter], **kwargs)
    for _ in range(2):
        _assign_gradient(parameter)
        optimizer.step()
    assert parameter.dtype == torch.bfloat16
    assert torch.isfinite(parameter).all()
    state = optimizer.state[parameter]
    full_size_keys = {"exp_avg", "exp_avg_sq", "momentum_buffer"}
    assert all(
        state[key].dtype == parameter.dtype
        for key in full_size_keys
        if key in state
    )
    assert all(
        value.dtype == torch.float32
        for key, value in state.items()
        if key not in full_size_keys and torch.is_tensor(value)
    )

@pytest.mark.parametrize("optimizer_type", [SOAP, NorMuon, AdaMuon])
def test_modern_optimizers_use_adam_fallback_for_vectors(optimizer_type):
    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    optimizer = optimizer_type([parameter], lr=0.01, weight_decay=0.0)
    _assign_gradient(parameter)
    optimizer.step()
    assert torch.isfinite(parameter).all()
    assert optimizer.state[parameter]["exp_avg"].dtype == parameter.dtype

@pytest.mark.parametrize("optimizer_type", [AdamWScheduleFree, RAdamScheduleFree])
def test_schedulefree_full_size_state_follows_parameter_dtype(optimizer_type):
    parameter = torch.nn.Parameter(torch.randn(4, 3, dtype=torch.bfloat16))
    optimizer = optimizer_type(
        [parameter], lr=0.01, weight_decay=0.0, backend="torch",
    )
    optimizer.train()
    _assign_gradient(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert state["z"].dtype == parameter.dtype
    assert state["exp_avg_sq"].dtype == parameter.dtype

def test_adamw_lrsf_keeps_full_size_preconditioner_parameter_dtype_and_low_rank_fp32():
    parameter = torch.nn.Parameter(torch.randn(16, 8, dtype=torch.bfloat16))
    optimizer = AdamWLRSF(
        [parameter], lr=0.01, rank=4, weight_decay=0.0, backend="torch",
    )
    optimizer.train()
    _assign_gradient(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert state["exp_avg_sq"].dtype == parameter.dtype
    assert state["lrsf_delta"].dtype == torch.float32
    assert state["lrsf_projection"].dtype == torch.float32

def test_adamw_sf_lr_uses_latent_second_moment_for_matrix_parameters():
    parameter = torch.nn.Parameter(torch.randn(16, 8, dtype=torch.bfloat16))
    optimizer = AdamWSFLowRankPreconditioner(
        [parameter], lr=0.01, rank=4, weight_decay=0.0, backend="torch",
    )
    optimizer.train()
    _assign_gradient(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert state["backend"] == "sf_lr"
    assert "exp_avg_sq" not in state
    assert state["z"].dtype == parameter.dtype
    assert state["sf_lr_projection"].shape == (8, 4)
    assert state["sf_lr_exp_avg_sq"].shape == (16, 4)
    assert state["sf_lr_projection"].dtype == torch.float32
    assert state["sf_lr_exp_avg_sq"].dtype == torch.float32

def test_adamw_sf_lr_supports_wide_matrix_layout():
    parameter = torch.nn.Parameter(torch.randn(8, 16, dtype=torch.bfloat16))
    optimizer = AdamWSFLowRankPreconditioner(
        [parameter], lr=0.01, rank=4, weight_decay=0.0, backend="torch",
    )
    optimizer.train()
    _assign_gradient(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert state["backend"] == "sf_lr"
    assert state["sf_lr_projection"].shape == (4, 8)
    assert state["sf_lr_exp_avg_sq"].shape == (4, 16)
    assert torch.isfinite(parameter).all()

def test_adamw_lrsf_lr_shares_low_rank_drift_and_preconditioner_state():
    parameter = torch.nn.Parameter(torch.randn(16, 8, dtype=torch.bfloat16))
    optimizer = AdamWLRSLowRankPreconditioner(
        [parameter], lr=0.01, rank=4, weight_decay=0.0, backend="torch",
    )
    optimizer.train()
    _assign_gradient(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert state["backend"] == "lrsf"
    assert "z" not in state
    assert "exp_avg_sq" not in state
    assert state["lrsf_projection"].shape == (8, 4)
    assert state["lrsf_delta"].shape == (16, 4)
    assert state["lrsf_exp_avg_sq"].shape == (16, 4)
    assert torch.isfinite(parameter).all()

def test_adamw_lrsf_lr_hard_refresh_transports_latent_second_moment():
    parameter = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16))
    optimizer = AdamWLRSLowRankPreconditioner(
        [parameter],
        lr=0.01,
        rank=2,
        weight_decay=0.0,
        projection_refresh={"mode": "hard", "interval": 1},
        backend="torch",
    )
    optimizer.train()
    _assign_gradient(parameter)
    optimizer.step()
    old_projection = optimizer.state[parameter]["lrsf_projection"].clone()
    _assign_gradient(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert state["refresh_count"] == 1
    assert not torch.equal(old_projection, state["lrsf_projection"])
    assert torch.isfinite(state["lrsf_exp_avg_sq"]).all()

def test_adamw_lrsf_lr_hard_refresh_reset_restarts_moment_age():
    parameter = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16))
    optimizer = AdamWLRSLowRankPreconditioner(
        [parameter],
        lr=0.01,
        rank=2,
        weight_decay=0.0,
        projection_refresh={"mode": "hard", "interval": 1},
        projection_refresh_state="reset",
        backend="torch",
    )
    optimizer.train()
    _assign_gradient(parameter)
    optimizer.step()
    _assign_gradient(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert state["lrsf_moment_step"] == 1
    assert torch.isfinite(state["lrsf_exp_avg_sq"]).all()

def test_adamw_lrsf_lr_shadow_refresh_updates_both_state_branches():
    parameter = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16))
    optimizer = AdamWLRSLowRankPreconditioner(
        [parameter],
        lr=0.01,
        rank=2,
        weight_decay=0.0,
        projection_refresh={"mode": "shadow", "interval": 1},
        backend="torch",
    )
    optimizer.train()
    _assign_gradient(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert state["shadow_active"] is True
    assert torch.linalg.vector_norm(state["lrsf_shadow_exp_avg_sq"]) > 0
    active_projection = state["lrsf_projection"].clone()

    _assign_gradient(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert state["refresh_count"] == 1
    assert not torch.equal(active_projection, state["lrsf_projection"])
    assert state["lrsf_shadow_projection"].shape == active_projection.shape
    assert torch.linalg.vector_norm(state["lrsf_shadow_delta"]) > 0
    assert torch.linalg.vector_norm(state["lrsf_shadow_exp_avg_sq"]) > 0
    assert torch.isfinite(parameter).all()

def test_adamw_sf_lr_uses_exact_full_state_for_vectors_and_small_matrices():
    vector = torch.nn.Parameter(torch.randn(8, dtype=torch.bfloat16))
    small_matrix = torch.nn.Parameter(torch.randn(2, 2, dtype=torch.bfloat16))
    optimizer = AdamWSFLowRankPreconditioner(
        [vector, small_matrix], lr=0.01, rank=4, weight_decay=0.0,
        backend="torch",
    )
    optimizer.train()
    _assign_gradient(vector)
    _assign_gradient(small_matrix)
    optimizer.step()

    assert optimizer.state[vector]["backend"] == "sf_full"
    assert optimizer.state[small_matrix]["backend"] == "sf_full"
    assert optimizer.state[vector]["exp_avg_sq"].shape == vector.shape
    assert optimizer.state[small_matrix]["exp_avg_sq"].shape == small_matrix.shape

def test_adamw_sf_lr_full_rank_fallback_matches_schedulefree_adamw():
    torch.manual_seed(0)
    reference_parameter = torch.nn.Parameter(torch.randn(2, 2))
    candidate_parameter = torch.nn.Parameter(reference_parameter.detach().clone())
    kwargs: dict[str, Any] = dict(lr=1e-3, weight_decay=0.01, warmup_steps=2)
    reference = AdamWScheduleFree(
        [reference_parameter], betas=(0.9, 0.999), backend="torch", **kwargs,
    )
    candidate = AdamWSFLowRankPreconditioner(
        [candidate_parameter], rank=4, sf_beta1=0.9, beta2=0.999,
        backend="torch", **kwargs,
    )

    for _ in range(3):
        gradient = torch.randn_like(reference_parameter)
        reference_parameter.grad = gradient.clone()
        candidate_parameter.grad = gradient.clone()
        reference.train()
        candidate.train()
        reference.step()
        candidate.step()
        torch.testing.assert_close(reference_parameter, candidate_parameter, rtol=0, atol=0)
        reference.eval()
        candidate.eval()
        torch.testing.assert_close(reference_parameter, candidate_parameter, rtol=0, atol=0)

def test_adamw_auto_schedule_initializes_and_reports_effective_lr():
    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    optimizer = AdamWAutoSchedule(
        [parameter],
        lr=0.01,
        weight_decay=0.0,
        auto_schedule_warmup_steps=0,
    )
    schedule = LearningRateSchedule(
        optimizer,
        name="constant",
        total_steps=4,
    )
    _assign_gradient(parameter)
    schedule.step(1)
    optimizer.step()

    formatted = schedule.format_effective_lrs()
    assert formatted.startswith("main=")
    assert optimizer.param_groups[0]["_auto_schedule_step"] == 1

def test_auto_schedule_preview_does_not_initialize_missing_group_state():
	parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
	optimizer = AdamWAutoSchedule(
		[parameter],
		lr=0.01,
		weight_decay=0.0,
		auto_schedule_warmup_steps=0,
	)
	schedule = LearningRateSchedule(
		optimizer,
		name="constant",
		total_steps=4,
	)
	group = optimizer.param_groups[0]
	controller_keys = {
		key for key in group if key.startswith("_auto_schedule_")
	}
	for key in controller_keys:
		group.pop(key)
	keys_before = set(group)

	assert schedule.format_effective_lrs().startswith("main=")
	assert schedule.format_auto_schedule_state().startswith("main=")

	assert set(group) == keys_before

def test_adamw_lrsf_full_rank_fallback_matches_schedulefree_adamw():
    torch.manual_seed(0)
    reference_parameter = torch.nn.Parameter(torch.randn(2, 2))
    candidate_parameter = torch.nn.Parameter(reference_parameter.detach().clone())
    kwargs: dict[str, Any] = dict(
        lr=1e-3, weight_decay=0.01, warmup_steps=2,
        r=0.0, weight_lr_power=2.0,
    )
    reference = AdamWScheduleFree(
        [reference_parameter], betas=(0.9, 0.999), backend="torch", **kwargs,
    )
    candidate = AdamWLRSF(
        [candidate_parameter], rank=4, sf_beta1=0.9, beta2=0.999, **kwargs,
    )

    for _ in range(3):
        gradient = torch.randn_like(reference_parameter)
        reference_parameter.grad = gradient.clone()
        candidate_parameter.grad = gradient.clone()
        reference.train()
        candidate.train()
        reference.step()
        candidate.step()
        torch.testing.assert_close(reference_parameter, candidate_parameter, rtol=0, atol=0)
        reference.eval()
        candidate.eval()
        torch.testing.assert_close(reference_parameter, candidate_parameter, rtol=0, atol=0)
