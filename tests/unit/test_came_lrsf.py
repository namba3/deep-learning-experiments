import copy

import pytest
import torch

import optimizers.came_lrsf as came_lrsf_module
from optimizers import CAME, CAMESF, CAMELRSF
from optimizers import OrthogonalRefreshPolicy, ProjectionRefreshPolicy
from optimizers.projection_refresh import (
    add_mixed_delta,
    advance_refresh_mix,
    initialize_refresh_mix,
    prepare_stochastic_refresh,
    refresh_mix_weight,
    update_mixed_delta,
)


def test_came_sf_oracle_keeps_full_delta_and_round_trips_eval():
    torch.manual_seed(2)
    parameter = torch.nn.Parameter(torch.randn(4, 3))
    optimizer = CAMESF([parameter], lr=1e-3, weight_decay=0.0)
    optimizer.train()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        parameter.square().mean().backward()
        optimizer.step()

    state = optimizer.state[parameter]
    assert state["backend"] == "sf_full"
    assert tuple(state["sf_delta"].shape) == tuple(parameter.shape)
    train_parameter = parameter.detach().clone()
    optimizer.eval()
    eval_parameter = parameter.detach().clone()
    assert not torch.equal(train_parameter, eval_parameter)
    optimizer.train()
    assert torch.equal(parameter, train_parameter)


@pytest.mark.parametrize("optimizer_type", [CAME, CAMESF, CAMELRSF])
def test_came_family_uses_parameter_dtype_for_full_size_state(optimizer_type):
    parameter = torch.nn.Parameter(torch.randn(16, 8, dtype=torch.bfloat16))
    if optimizer_type is CAME:
        optimizer = optimizer_type([parameter], lr=1e-3, weight_decay=0.0)
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
    elif optimizer_type is CAMESF:
        optimizer = optimizer_type([parameter], lr=1e-3, weight_decay=0.0)
        optimizer.train()
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
    else:
        optimizer = optimizer_type(
            [parameter], lr=1e-3, rank=4, weight_decay=0.0,
        )
        optimizer.train()
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

    state = optimizer.state[parameter]
    assert state["exp_avg"].dtype == parameter.dtype
    if "sf_delta" in state:
        assert state["sf_delta"].dtype == parameter.dtype
    if "lrsf_delta" in state:
        assert state["lrsf_delta"].dtype == torch.float32


def test_came_sf_checkpoint_round_trip_continues_identically():
    torch.manual_seed(19)
    first = torch.nn.Parameter(torch.randn(4, 3))
    original = CAMESF([first], lr=1e-3, weight_decay=0.0)
    original.train()
    for _ in range(2):
        original.zero_grad(set_to_none=True)
        first.square().mean().backward()
        original.step()

    restored_parameter = torch.nn.Parameter(first.detach().clone())
    restored = CAMESF([restored_parameter], lr=1e-3, weight_decay=0.0)
    restored.load_state_dict(copy.deepcopy(original.state_dict()))
    restored.train()
    original.zero_grad(set_to_none=True)
    restored.zero_grad(set_to_none=True)
    first.square().mean().backward()
    restored_parameter.square().mean().backward()
    original.step()
    restored.step()

    assert torch.equal(first, restored_parameter)


def test_came_lrsf_uses_low_rank_state_and_round_trips_train_eval():
    torch.manual_seed(3)
    parameter = torch.nn.Parameter(torch.randn(16, 8))
    optimizer = CAMELRSF(
        [parameter], lr=1e-2, rank=4, sf_beta1=0.9, weight_decay=0.0,
    )

    optimizer.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        parameter.square().mean().backward()
        optimizer.step()

    state = optimizer.state[parameter]
    assert state["backend"] == "lrsf"
    assert tuple(state["lrsf_projection"].shape) == (8, 4)
    assert tuple(state["lrsf_delta"].shape) == (16, 4)

    train_parameter = parameter.detach().clone()
    optimizer.eval()
    eval_parameter = parameter.detach().clone()
    assert not torch.equal(train_parameter, eval_parameter)
    optimizer.train()
    assert torch.allclose(parameter, train_parameter, atol=1e-6, rtol=1e-6)


def test_came_lrsf_falls_back_for_vectors_and_full_rank_matrices():
    for shape, rank in [((16,), 4), ((4, 3), 4), ((8, 8), 8)]:
        parameter = torch.nn.Parameter(torch.ones(shape))
        optimizer = CAMELRSF([parameter], lr=1e-3, rank=rank)
        optimizer.train()
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        assert optimizer.state[parameter]["backend"] == "came"


def test_came_lrsf_preserves_wide_and_flattened_matrix_layouts():
    for shape, projection_shape, delta_shape in [
        ((4, 16), (2, 4), (2, 16)),
        ((4, 3, 3), (2, 4), (2, 9)),
    ]:
        parameter = torch.nn.Parameter(torch.ones(shape))
        optimizer = CAMELRSF([parameter], lr=1e-3, rank=2)
        optimizer.train()
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        state = optimizer.state[parameter]
        assert state["backend"] == "lrsf"
        assert tuple(state["lrsf_projection"].shape) == projection_shape
        assert tuple(state["lrsf_delta"].shape) == delta_shape


def test_came_lrsf_came_fallback_matches_came():
    torch.manual_seed(11)
    came_parameter = torch.nn.Parameter(torch.randn(4, 3))
    lrsf_parameter = torch.nn.Parameter(came_parameter.detach().clone())
    came = CAME([came_parameter], lr=1e-3, weight_decay=0.0, backend="torch")
    lrsf = CAMELRSF([lrsf_parameter], lr=1e-3, rank=4, weight_decay=0.0)
    gradient = torch.randn_like(came_parameter)
    came_parameter.grad = gradient
    lrsf_parameter.grad = gradient.clone()

    came.step()
    lrsf.train()
    lrsf.step()

    assert torch.equal(came_parameter, lrsf_parameter)


def test_came_lrsf_checkpoint_round_trip_continues_identically():
    torch.manual_seed(17)
    first = torch.nn.Parameter(torch.randn(16, 16))
    original = CAMELRSF([first], lr=1e-3, rank=4, weight_decay=0.0)
    original.train()
    for _ in range(3):
        original.zero_grad(set_to_none=True)
        first.square().mean().backward()
        original.step()

    restored_parameter = torch.nn.Parameter(first.detach().clone())
    restored = CAMELRSF(
        [restored_parameter], lr=1e-3, rank=4, weight_decay=0.0,
    )
    restored.load_state_dict(copy.deepcopy(original.state_dict()))
    restored.train()
    original.zero_grad(set_to_none=True)
    restored.zero_grad(set_to_none=True)
    first.square().mean().backward()
    restored_parameter.square().mean().backward()
    original.step()
    restored.step()

    assert torch.equal(first, restored_parameter)


def test_came_lrsf_smooth_projection_refresh_uses_pa_pb_and_promotes():
    torch.manual_seed(21)
    parameter = torch.nn.Parameter(torch.randn(16, 16))
    optimizer = CAMELRSF(
        [parameter], lr=1e-3, rank=4, weight_decay=0.0,
        projection_refresh={"mode": "smooth", "interval": 2, "window": 2},
    )
    assert optimizer.param_groups[0]["projection_refresh"] == {
        "mode": "smooth", "interval": 2, "window": 2, "mix": "smoothstep",
    }
    optimizer.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        parameter.square().mean().backward()
        optimizer.step()

    state = optimizer.state[parameter]
    assert state["refresh_active"] is True
    assert "refresh_next_projection" in state
    assert "refresh_next_delta" in state
    assert state["refresh_progress"] == 1

    optimizer.zero_grad(set_to_none=True)
    parameter.square().mean().backward()
    optimizer.step()
    assert optimizer.state[parameter]["refresh_active"] is False
    assert "refresh_next_projection" not in optimizer.state[parameter]
    assert optimizer.state[parameter]["refresh_count"] == 1


@pytest.mark.parametrize("shape", [(16, 8), (8, 16)])
def test_lrsf_hard_refresh_records_optional_transport_diagnostics(shape):
    torch.manual_seed(23)
    parameter = torch.nn.Parameter(torch.randn(*shape))
    optimizer = CAMELRSF(
        [parameter],
        lr=1e-3,
        rank=4,
        projection_refresh={
            "mode": "hard",
            "interval": 2,
            "diagnostics": True,
        },
    )
    optimizer.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        parameter.square().mean().backward()
        optimizer.step()

    state = optimizer.state[parameter]
    assert optimizer.param_groups[0]["projection_refresh"]["diagnostics"] is True
    assert state["refresh_transport_diagnostic_count"] == 1
    assert state["refresh_transport_error_sum"] >= 0.0
    assert state["refresh_transport_error_max"] >= 0.0
    assert state["refresh_transport_norm_ratio_sum"] >= 0.0
    assert -1.0 <= state["refresh_transport_cosine_sum"] <= 1.0
    assert state["refresh_transport_last_step"] == 2


def test_lrsf_hard_refresh_transport_overlap_one_keeps_old_basis():
    torch.manual_seed(29)
    parameter = torch.nn.Parameter(torch.randn(16, 8))
    optimizer = CAMELRSF(
        [parameter],
        lr=1e-3,
        rank=4,
        projection_refresh={
            "mode": "hard",
            "interval": 2,
            "diagnostics": True,
            "transport_overlap": 1.0,
        },
    )
    optimizer.train()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    old_projection = optimizer.state[parameter]["lrsf_projection"].clone()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    torch.testing.assert_close(state["lrsf_projection"], old_projection)
    assert state["refresh_transport_error_max"] < 1e-6


def test_lrsf_shadow_refresh_promotes_trained_branch_and_starts_next_one():
    torch.manual_seed(37)
    parameter = torch.nn.Parameter(torch.randn(16, 8))
    optimizer = CAMELRSF(
        [parameter],
        lr=1e-3,
        rank=4,
        projection_refresh={
            "mode": "shadow",
            "interval": 2,
            "diagnostics": True,
            "transport_overlap": 0.9,
        },
    )
    optimizer.train()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    assert state["shadow_active"] is True
    assert torch.linalg.vector_norm(state["lrsf_shadow_delta"]) > 0
    shadow_projection = state["lrsf_shadow_projection"].clone()

    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    torch.testing.assert_close(state["lrsf_projection"], shadow_projection)
    assert state["refresh_count"] == 1
    assert state["shadow_generation"] == 1
    assert state["shadow_gap_count"] == 1
    assert state["shadow_gap_error_max"] >= 0.0
    assert not torch.equal(state["lrsf_shadow_projection"], shadow_projection)
    assert torch.linalg.vector_norm(state["lrsf_shadow_delta"]) > 0


def test_lrsf_shadow_refresh_checkpoint_round_trip_continues_identically():
    torch.manual_seed(41)
    first = torch.nn.Parameter(torch.randn(16, 16))
    original = CAMELRSF(
        [first],
        lr=1e-3,
        rank=4,
        projection_refresh={"mode": "shadow", "interval": 2},
    )
    original.train()
    for _ in range(3):
        original.zero_grad(set_to_none=True)
        first.square().mean().backward()
        original.step()

    restored_parameter = torch.nn.Parameter(first.detach().clone())
    restored = CAMELRSF(
        [restored_parameter],
        lr=1e-3,
        rank=4,
        projection_refresh={"mode": "shadow", "interval": 2},
    )
    restored.load_state_dict(copy.deepcopy(original.state_dict()))
    restored.train()
    original.zero_grad(set_to_none=True)
    restored.zero_grad(set_to_none=True)
    first.square().mean().backward()
    restored_parameter.square().mean().backward()
    original.step()
    restored.step()

    assert torch.equal(first, restored_parameter)


def test_lrsf_refresh_buffers_update_as_independent_ema_states():
    state = {
        "lrsf_delta": torch.ones(2, 2),
        "refresh_active": True,
        "refresh_next_delta": torch.zeros(2, 2),
    }
    projected_a = torch.full((2, 2), 2.0)
    projected_b = torch.full((2, 2), 3.0)

    update_mixed_delta(
        state, (projected_a, projected_b), decay=0.5, update_scale=0.25,
    )

    torch.testing.assert_close(
        state["lrsf_delta"], torch.full((2, 2), 1.0),
    )
    torch.testing.assert_close(
        state["refresh_next_delta"], torch.full((2, 2), 0.75),
    )


def test_stochastic_refresh_moves_selection_probability_from_pa_to_pb():
    policy = ProjectionRefreshPolicy(
        mode="smooth", interval=2, window=4, mix="stochastic",
    )
    state = {
        "refresh_active": True,
        "refresh_progress": 0,
        "lrsf_delta": torch.ones(2, 2),
        "lrsf_projection": torch.eye(2),
        "refresh_next_delta": torch.full((2, 2), 2.0),
        "refresh_next_projection": torch.eye(2),
    }
    assert prepare_stochastic_refresh(state, policy, seed=13) == 0
    state["refresh_progress"] = policy.window
    assert prepare_stochastic_refresh(state, policy, seed=13) == 1

    matrix = torch.zeros(2, 2)

    def add_low_rank(target, delta, _projection, alpha):
        target.add_(delta, alpha=alpha)

    add_mixed_delta(matrix, state, 1.0, add_low_rank, policy)
    torch.testing.assert_close(matrix, state["refresh_next_delta"])


def test_ema_refresh_weight_moves_exponentially_from_pa_to_pb():
    policy = ProjectionRefreshPolicy(
        mode="smooth", interval=2, window=4, mix="ema",
    )
    state = {"refresh_active": True, "refresh_progress": 0}
    initialize_refresh_mix(state, policy)
    assert refresh_mix_weight(state, policy) == 0.0

    advance_refresh_mix(state, policy)
    first = refresh_mix_weight(state, policy)
    advance_refresh_mix(state, policy)
    second = refresh_mix_weight(state, policy)

    assert 0.0 < first < second < 1.0
    expected_first = 1.0 - torch.exp(torch.tensor(-0.25)).item()
    assert first == pytest.approx(expected_first)


@pytest.mark.parametrize("shape", [(16, 8), (8, 16)])
def test_came_lrsf_orthogonal_refresh_rotates_every_step(shape):
    torch.manual_seed(27)
    parameter = torch.nn.Parameter(torch.randn(*shape))
    optimizer = CAMELRSF(
        [parameter], lr=1e-3, rank=2, weight_decay=0.0,
        orthogonal_refresh={"rate": 0.05, "seed": 31},
    )
    assert optimizer.param_groups[0]["orthogonal_refresh"] == {
        "rate": 0.05, "seed": 31,
    }
    assert isinstance(
        OrthogonalRefreshPolicy.from_value(
            optimizer.param_groups[0]["orthogonal_refresh"]
        ),
        OrthogonalRefreshPolicy,
    )
    optimizer.train()
    projections = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        parameter.square().mean().backward()
        optimizer.step()
        projections.append(optimizer.state[parameter]["lrsf_projection"].clone())

    assert optimizer.state[parameter]["orthogonal_refresh_count"] == 2
    assert not torch.equal(projections[0], projections[1])
    if shape[0] >= shape[1]:
        torch.testing.assert_close(
            projections[-1].transpose(0, 1).matmul(projections[-1]),
            torch.eye(2),
            atol=1e-5,
            rtol=1e-5,
        )
    else:
        torch.testing.assert_close(
            projections[-1].matmul(projections[-1].transpose(0, 1)),
            torch.eye(2),
            atol=1e-5,
            rtol=1e-5,
        )


@pytest.mark.parametrize("shape", [(16, 8), (8, 16)])
def test_came_lrsf_loss_directed_refresh_rotates_delta_projection(shape):
    torch.manual_seed(28)
    parameter = torch.nn.Parameter(torch.randn(*shape))
    optimizer = CAMELRSF(
        [parameter], lr=1e-3, rank=2, weight_decay=0.0,
        orthogonal_refresh={
            "rate": 0.01, "seed": 32, "direction": "loss_directed",
        },
    )
    optimizer.train()
    projections = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        parameter.square().mean().backward()
        optimizer.step()
        projections.append(
            optimizer.state[parameter]["lrsf_projection"].clone()
        )

    state = optimizer.state[parameter]
    assert state["orthogonal_refresh_count"] == 2
    assert not torch.equal(projections[0], projections[1])
    assert torch.isfinite(state["lrsf_delta"]).all()


def test_came_lrsf_loss_lowering_refresh_uses_hidden_delta():
    parameter = torch.nn.Parameter(torch.randn(8, 6))
    optimizer = CAMELRSF(
        [parameter], lr=1e-3, rank=2,
        orthogonal_refresh={
            "rate": 0.01, "seed": 12, "direction": "loss_lowering",
        },
    )
    optimizer.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        parameter.square().mean().backward()
        optimizer.step()

    state = optimizer.state[parameter]
    assert state["orthogonal_refresh_count"] == 3
    assert torch.isfinite(state["lrsf_projection"]).all()
    assert torch.isfinite(state["lrsf_delta"]).all()


def test_came_lrsf_effective_update_is_used_as_rotation_signal(monkeypatch):
    captured = {}
    original = came_lrsf_module.rotate_projection_state

    def wrapped(state, policy, *, gradient=None):
        captured["signal"] = gradient
        return original(state, policy, gradient=gradient)

    monkeypatch.setattr(came_lrsf_module, "rotate_projection_state", wrapped)
    parameter = torch.nn.Parameter(torch.randn(8, 8))
    optimizer = CAMELRSF(
        [parameter], lr=1e-3, rank=2, weight_decay=0.0,
        orthogonal_refresh={
            "rate": 0.01, "direction": "loss_directed",
            "signal": "effective_update",
        },
    )
    optimizer.train()
    parameter.square().mean().backward()
    optimizer.step()

    assert captured["signal"] is not parameter.grad
    assert captured["signal"].shape == parameter.shape


def test_came_lrsf_checkpoint_resume_preserves_active_refresh():
    torch.manual_seed(23)
    kwargs = {
        "lr": 1e-3,
        "rank": 4,
        "projection_refresh": {
            "mode": "smooth", "interval": 2, "window": 3,
        },
    }
    first = torch.nn.Parameter(torch.randn(16, 16))
    original = CAMELRSF([first], **kwargs)
    original.train()
    for _ in range(3):
        original.zero_grad(set_to_none=True)
        first.square().mean().backward()
        original.step()
    assert original.state[first]["refresh_active"] is True

    restored_parameter = torch.nn.Parameter(first.detach().clone())
    restored = CAMELRSF([restored_parameter], **kwargs)
    restored.load_state_dict(copy.deepcopy(original.state_dict()))
    restored.train()
    original.zero_grad(set_to_none=True)
    restored.zero_grad(set_to_none=True)
    first.square().mean().backward()
    restored_parameter.square().mean().backward()
    original.step()
    restored.step()

    assert torch.equal(first, restored_parameter)
    assert original.state[first]["refresh_progress"] == restored.state[restored_parameter]["refresh_progress"]


def test_projection_refresh_policy_round_trips_mapping_and_rejects_invalid_mode():
    policy = ProjectionRefreshPolicy.from_value({
        "mode": "hard", "interval": 3,
    })
    assert policy.as_dict() == {
        "mode": "hard", "interval": 3, "window": 0, "mix": "smoothstep",
    }
    try:
        ProjectionRefreshPolicy(mode="invalid")
    except ValueError as error:
        assert "refresh mode" in str(error)
    else:
        raise AssertionError("invalid refresh mode should fail")
