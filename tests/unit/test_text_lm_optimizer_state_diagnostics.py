from typing import Any, cast

import pytest
import torch
from optimizers.adamw_lrsf import AdamWLRSF
from verify.text_lm_optimizer_convergence import _confidence_diagnostic_snapshot, _decode_projected_matrix, _lrsf_latent_moment_snapshot, _matrix_rank_metrics, _rank_analysis_role, _state_rank_snapshot, _state_trajectory_snapshot, _schedulefree_trajectory_snapshot, _schedulefree_trajectory_gap_snapshot, _update_sf_delta_ema


def test_lrsf_latent_moment_snapshot_exposes_reset_age_and_refresh_event():
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    optimizer.state[parameter]["lrsf_exp_avg_sq"] = torch.ones(2, 1)
    optimizer.state[parameter]["lrsf_moment_step"] = 1
    optimizer.state[parameter]["refresh_count"] = 1
    previous = {}

    first = _lrsf_latent_moment_snapshot(optimizer, 25, previous)
    assert first is not None
    assert first["refresh_event"] is True
    assert first["moment_step_min"] == 1
    assert first["latent_moment_norm"] == pytest.approx(2.0 ** 0.5)

    optimizer.state[parameter]["lrsf_moment_step"] = 2
    second = _lrsf_latent_moment_snapshot(optimizer, 26, previous)
    assert second is not None
    assert second["refresh_event"] is False
    assert second["moment_step_min"] == 2

def test_state_rank_metrics_report_low_rank_energy():
    tensor = torch.zeros(6, 4)
    tensor[:, 0] = 1.0
    metrics = cast(dict[str, Any] | None, _matrix_rank_metrics(tensor, max_elements=100))

    assert metrics is not None
    assert metrics["rank_90"] == 1
    assert metrics["rank_95"] == 1
    assert metrics["retained_energy"]["1"] == 1.0
    assert metrics["effective_rank"] == 1.0

def test_decode_projected_matrix_supports_apollo_tall_layout():
    latent = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    projection = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    decoded = _decode_projected_matrix(latent, projection, (4, 3))

    assert decoded is not None
    assert decoded.shape == (4, 3)
    assert torch.equal(decoded, latent.matmul(projection.transpose(0, 1)))

def test_state_rank_snapshot_labels_apollo_latent_and_decoded_states():
    parameter = torch.nn.Parameter(torch.zeros(4, 3))
    model = torch.nn.Linear(3, 4, bias=False)
    model.weight = parameter
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    projection = torch.randn(3, 2)
    optimizer.state[parameter].update({
        "backend": "apollo",
        "projection": projection,
        "exp_avg": torch.randn(4, 2),
        "exp_avg_sq": torch.randn(4, 2),
        "scaled_grad_norm": torch.tensor(1.0),
    })

    snapshots = _state_rank_snapshot(
        model,
        optimizer,
        max_elements=100,
        max_tensors=1,
        parameter_substring=None,
    )
    sources = {item["source"] for item in snapshots}

    assert "apollo_R_update" in sources
    assert "apollo_exp_avg_latent" in sources
    assert "apollo_exp_avg_decoded" in sources
    assert "apollo_exp_avg_sq_latent" in sources
    assert "apollo_exp_avg_sq_decoded" in sources
    assert "apollo_scaled_grad_norm" in sources
    roles = {item["source"]: item["analysis_role"] for item in snapshots}
    assert roles["apollo_exp_avg_latent"] == "latent_utilization"
    assert roles["apollo_exp_avg_decoded"] == "projection_constrained_reference"

def test_state_rank_snapshot_labels_confidence_ema_states():
    parameter = torch.nn.Parameter(torch.zeros(4, 3))
    model = torch.nn.Linear(3, 4, bias=False)
    model.weight = parameter
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    optimizer.state[parameter].update({
        "backend": "lr_ema_confidence",
        "lr_ema_projection": torch.randn(3, 2),
        "lr_ema_grad": torch.randn(4, 2),
        "lr_ema_residual_sq": torch.rand(4, 2),
    })

    snapshots = _state_rank_snapshot(
        model,
        optimizer,
        max_elements=100,
        max_tensors=1,
        parameter_substring=None,
    )
    sources = {item["source"] for item in snapshots}

    assert "lr_ema_projection" in sources
    assert "lr_ema_grad" in sources
    assert "lr_ema_residual_sq" in sources
    roles = {item["source"]: item["analysis_role"] for item in snapshots}
    assert roles["lr_ema_grad"] == "compressed_state"

def test_confidence_diagnostic_snapshot_reports_bounded_signal_ratio():
    parameter = torch.nn.Parameter(torch.zeros(4, 3))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    optimizer.param_groups[0].update({
        "lr_ema_beta": 0.9,
        "lr_ema_confidence_beta": 0.99,
        "lr_ema_confidence_alpha": 1e-3,
        "lr_ema_eps": 1e-8,
    })
    optimizer.state[parameter].update({
        "backend": "lr_ema_confidence",
        "step": 1,
        "lr_ema_grad": torch.ones(4, 2),
        "lr_ema_residual_sq": torch.ones(4, 2),
    })

    metrics = _confidence_diagnostic_snapshot(optimizer)

    assert metrics is not None
    assert metrics["metric_type"] == "confidence_normalization"
    assert metrics["elements"] == 8
    assert metrics["confidence_mean"] == 0.5
    assert metrics["confidence_std"] == 0.0

def test_confidence_diagnostic_snapshot_accepts_integrated_lrsf_counter():
    parameter = torch.nn.Parameter(torch.zeros(4, 3))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    optimizer.param_groups[0].update({
        "lr_ema_beta": 0.9,
        "lr_ema_confidence_beta": 0.99,
        "lr_ema_confidence_alpha": 1e-3,
        "lr_ema_eps": 1e-8,
    })
    optimizer.state[parameter].update({
        "backend": "lrsf",
        "lr_ema_step": 1,
        "lr_ema_confidence_projection": torch.randn(3, 2),
        "lr_ema_grad": torch.ones(4, 2),
        "lr_ema_residual_sq": torch.ones(4, 2),
    })

    metrics = _confidence_diagnostic_snapshot(optimizer)

    assert metrics is not None
    assert metrics["step"] == 1
    assert metrics["elements"] == 8

def test_rank_analysis_role_separates_full_state_candidates_from_apollo_latents():
    assert _rank_analysis_role("exp_avg_sq") == "compression_candidate"
    assert _rank_analysis_role("sf_delta") == "compression_candidate"
    assert _rank_analysis_role("apollo_exp_avg_sq_latent") == "latent_utilization"
    assert _rank_analysis_role("apollo_R_update") == "projection_baseline"

def test_schedulefree_trajectory_snapshot_reconstructs_hidden_and_eval_views():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    model = torch.nn.Linear(2, 2, bias=False)
    model.weight = parameter
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    optimizer.param_groups[0]["betas"] = (0.9, 0.999)
    optimizer.state[parameter]["z"] = torch.full_like(parameter, 3.0)

    snapshots = _schedulefree_trajectory_snapshot(
        model,
        optimizer,
        max_elements=100,
        max_tensors=1,
        parameter_substring=None,
    )
    values = {source: value for _, source, value in snapshots}

    assert torch.equal(values["train_parameter"], torch.ones(4))
    assert torch.equal(values["hidden_state"], torch.full((4,), 3.0))
    assert torch.allclose(
        values["eval_parameter"], torch.full((4,), 1.0 - 2.0 / 9.0),
    )

def test_schedulefree_trajectory_snapshot_decodes_lrsf_delta():
    parameter = torch.nn.Parameter(torch.zeros(4, 2))
    model = torch.nn.Linear(2, 4, bias=False)
    model.weight = parameter
    optimizer = AdamWLRSF([parameter], rank=1, backend="torch")
    optimizer.param_groups[0]["train_mode"] = True
    optimizer.state[parameter].update({
        "backend": "lrsf",
        "lrsf_projection": torch.tensor([[1.0], [0.0]]),
        "lrsf_delta": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    })

    snapshots = _schedulefree_trajectory_snapshot(
        model,
        optimizer,
        max_elements=100,
        max_tensors=1,
        parameter_substring=None,
    )
    values = {source: value for _, source, value in snapshots}

    expected_hidden = torch.tensor([1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0])
    assert torch.equal(values["hidden_state"], expected_hidden)
    assert torch.allclose(
        values["eval_parameter"], expected_hidden * (1.0 - 1.0 / 0.9),
    )

def test_schedulefree_trajectory_gap_snapshot_reports_sf_scaling():
    snapshots = [
        ("weight", "train_parameter", torch.zeros(2)),
        ("weight", "hidden_state", torch.full((2,), 2.0)),
        ("weight", "eval_parameter", torch.full((2,), -2.0 / 9.0)),
    ]

    records = _schedulefree_trajectory_gap_snapshot(
        snapshots, step=4, training_loss=1.25,
    )

    assert len(records) == 1
    record = records[0]
    assert record["step"] == 4
    assert record["training_loss"] == 1.25
    assert torch.isclose(
        torch.tensor(record["hidden_train_gap_norm"]),
        torch.tensor(2.0 * 2.0**0.5),
    )
    assert torch.isclose(
        torch.tensor(record["eval_hidden_gap_ratio"]),
        torch.tensor(1.0 / 0.9),
    )

def test_state_trajectory_snapshot_excludes_apollo_latent_state():
    parameter = torch.nn.Parameter(torch.zeros(4, 4))
    model = torch.nn.Linear(4, 4, bias=False)
    model.weight = parameter
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    parameter.grad = torch.ones_like(parameter)
    optimizer.state[parameter].update({
        "backend": "apollo",
        "exp_avg": torch.zeros(4, 2),
        "exp_avg_sq": torch.zeros(4, 2),
    })

    snapshots = _state_trajectory_snapshot(
        model,
        optimizer,
        max_elements=100,
        max_tensors=1,
        parameter_substring=None,
    )

    assert snapshots == []

def test_state_rank_delta_ema_tracks_schedule_free_drift():
    parameter = torch.nn.Parameter(torch.zeros(4, 2))
    model = torch.nn.Linear(2, 4, bias=False)
    model.weight = parameter
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    optimizer.state[parameter]["z"] = torch.ones_like(parameter)

    ema = _update_sf_delta_ema(
        model,
        optimizer,
        {},
        max_elements=100,
        max_tensors=1,
        parameter_substring=None,
        decay=0.5,
    )
    assert torch.equal(ema["weight"], torch.ones(4, 2))

    optimizer.state[parameter]["z"].zero_()
    _update_sf_delta_ema(
        model,
        optimizer,
        ema,
        max_elements=100,
        max_tensors=1,
        parameter_substring=None,
        decay=0.5,
    )
    assert torch.equal(ema["weight"], torch.full((4, 2), 0.5))
