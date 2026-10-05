import copy

import pytest
import torch

from optimizers import APOLLO, APOLLOCAME, APOLLOCAMELRSF
from optimizers.projection_refresh import (
    OrthogonalRefreshPolicy,
    rotate_orthogonal_projection,
)


def test_loss_directed_policy_round_trips_explicit_direction():
    policy = OrthogonalRefreshPolicy.from_value(
        {"rate": 0.01, "seed": 7, "direction": "loss_directed"}
    )

    assert policy.direction == "loss_directed"
    assert policy.as_dict() == {
        "rate": 0.01, "seed": 7, "direction": "loss_directed",
    }


def test_loss_directed_policy_accepts_effective_update_signal():
    policy = OrthogonalRefreshPolicy.from_value({
        "rate": 0.01,
        "direction": "loss_directed",
        "signal": "effective_update",
    })

    assert policy.signal == "effective_update"
    assert policy.as_dict()["signal"] == "effective_update"


def test_loss_lowering_policy_round_trips_as_lrsf_only_direction():
    policy = OrthogonalRefreshPolicy.from_value({
        "rate": 0.005, "seed": 9, "direction": "loss_lowering",
    })

    assert policy.direction == "loss_lowering"
    assert policy.as_dict()["direction"] == "loss_lowering"


@pytest.mark.parametrize("rows_ge_cols", [True, False])
def test_apollo_transport_refresh_maps_first_and_second_moments(rows_ge_cols):
    if rows_ge_cols:
        old_projection = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        new_projection = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, -1.0]])
        state = {
            "exp_avg": torch.arange(6.0).reshape(3, 2),
            "exp_avg_sq": torch.arange(6.0).reshape(3, 2).square(),
        }
        overlap = old_projection.t().mm(new_projection)
        expected_avg = state["exp_avg"].mm(overlap)
        expected_sq = state["exp_avg_sq"].mm(overlap.square())
    else:
        old_projection = torch.tensor([[1.0, 0.0, 0.0]])
        new_projection = torch.tensor([[0.0, 1.0, 0.0]])
        state = {
            "exp_avg": torch.arange(8.0).reshape(1, 8),
            "exp_avg_sq": torch.arange(8.0).reshape(1, 8).square(),
        }
        overlap = new_projection.mm(old_projection.t())
        expected_avg = overlap.mm(state["exp_avg"])
        expected_sq = overlap.square().mm(state["exp_avg_sq"])

    APOLLO._transport_projection_state(
        state, old_projection, new_projection, rows_ge_cols=rows_ge_cols,
    )

    torch.testing.assert_close(state["exp_avg"], expected_avg)
    torch.testing.assert_close(state["exp_avg_sq"], expected_sq)


def test_apollo_came_transport_refresh_maps_projected_came_factors():
    old_projection = torch.eye(3, 2)
    new_projection = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    state = {
        "exp_avg": torch.arange(6.0).reshape(3, 2),
        "exp_avg_sq_row": torch.tensor([1.0, 2.0, 3.0]),
        "exp_avg_sq_col": torch.tensor([4.0, 5.0]),
        "exp_avg_res_row": torch.tensor([6.0, 7.0, 8.0]),
        "exp_avg_res_col": torch.tensor([9.0, 10.0]),
    }
    overlap = old_projection.t().mm(new_projection)
    expected_col = overlap.square().t().matmul(state["exp_avg_sq_col"])
    expected_res_col = overlap.square().t().matmul(state["exp_avg_res_col"])

    APOLLOCAME._transport_projection_state(
        state, old_projection, new_projection, rows_ge_cols=True,
    )

    torch.testing.assert_close(state["exp_avg_sq_col"], expected_col)
    torch.testing.assert_close(state["exp_avg_res_col"], expected_res_col)
    torch.testing.assert_close(state["exp_avg_sq_row"], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(state["exp_avg_res_row"], torch.tensor([6.0, 7.0, 8.0]))


@pytest.mark.parametrize("optimizer_cls", [APOLLO, APOLLOCAME])
@pytest.mark.parametrize("transport", [False, True])
def test_apollo_uses_shared_refresh_policy_for_smooth_mode(optimizer_cls, transport):
    parameter = torch.nn.Parameter(torch.randn(6, 4))
    optimizer = optimizer_cls(
        [parameter],
        update_proj_gap=3,
        fallback={"small_matrix": "apollo"},
        projection_refresh={"mode": "smooth", "interval": 2, "window": 2},
        projection_refresh_state="transport" if transport else "reset",
    )

    assert optimizer.param_groups[0]["projection_refresh"] == {
        "mode": "smooth", "interval": 2, "window": 2,
        "mix": "smoothstep",
    }
    assert optimizer.param_groups[0]["projection_refresh_state"] == (
        "transport" if transport else "reset"
    )

    for _ in range(3):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

    state = optimizer.state[parameter]
    assert torch.isfinite(parameter).all()
    assert not state.get("refresh_active", False)
    assert "refresh_next_projection" not in state


@pytest.mark.parametrize("optimizer_cls", [APOLLO, APOLLOCAME])
def test_apollo_hard_refresh_uses_next_projection_seed(optimizer_cls):
    parameter = torch.nn.Parameter(torch.randn(6, 4))
    kwargs = {
        "update_proj_gap": 2,
        "fallback": {"small_matrix": "apollo"},
        "projection_refresh": {"mode": "hard", "interval": 2},
    }
    if optimizer_cls is APOLLOCAME:
        kwargs["came_backend"] = "torch"
    optimizer = optimizer_cls([parameter], **kwargs)

    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    old_projection = optimizer.state[parameter]["projection"].detach().clone()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert state["projection_seed"] != 0
    assert not torch.equal(state["projection"], old_projection)


@pytest.mark.parametrize("optimizer_cls", [APOLLO, APOLLOCAME])
@pytest.mark.parametrize("shape", [(6, 4), (4, 6)])
def test_apollo_orthogonal_refresh_rotates_and_preserves_projection_layout(
    optimizer_cls, shape,
):
    parameter = torch.nn.Parameter(torch.randn(*shape))
    kwargs = {
        "fallback": {"small_matrix": "apollo"},
        "orthogonal_refresh": {"rate": 0.05, "seed": 17},
    }
    if optimizer_cls is APOLLOCAME:
        kwargs["came_backend"] = "torch"
    optimizer = optimizer_cls([parameter], **kwargs)

    for _ in range(2):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

    state = optimizer.state[parameter]
    projection = state["projection"]
    assert state["orthogonal_refresh_count"] == 2
    assert torch.isfinite(parameter).all()
    if shape[0] >= shape[1]:
        gram = projection.transpose(0, 1).matmul(projection)
    else:
        gram = projection.matmul(projection.transpose(0, 1))
    torch.testing.assert_close(gram, torch.eye(min(shape)), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("projection_shape", "gradient_shape"),
    [((6, 2), (8, 6)), ((2, 6), (6, 8))],
)
def test_loss_directed_rotation_increases_projected_gradient_energy(
    projection_shape, gradient_shape,
):
    torch.manual_seed(0)
    if projection_shape[0] >= projection_shape[1]:
        projection = torch.linalg.qr(
            torch.randn(*projection_shape), mode="reduced",
        ).Q
    else:
        projection = torch.linalg.qr(
            torch.randn(projection_shape[1], projection_shape[0]),
            mode="reduced",
        ).Q.transpose(0, 1)
    gradient = torch.randn(*gradient_shape)

    rotated = rotate_orthogonal_projection(
        projection,
        rate=0.01,
        seed=7,
        direction="loss_directed",
        gradient=gradient,
    )
    old_projected = (
        gradient.matmul(projection)
        if gradient.shape[0] >= gradient.shape[1]
        else projection.matmul(gradient)
    )
    new_projected = (
        gradient.matmul(rotated)
        if gradient.shape[0] >= gradient.shape[1]
        else rotated.matmul(gradient)
    )

    assert not torch.equal(rotated, projection)
    assert new_projected.square().sum() > old_projected.square().sum()


@pytest.mark.parametrize(
    ("projection_shape", "gradient_shape", "delta_shape"),
    [((6, 2), (8, 6), (8, 2)), ((2, 6), (6, 8), (2, 8))],
)
def test_loss_lowering_rotation_decreases_decoded_delta_linear_loss(
    projection_shape, gradient_shape, delta_shape,
):
    torch.manual_seed(4)
    if projection_shape[0] >= projection_shape[1]:
        projection = torch.linalg.qr(
            torch.randn(*projection_shape), mode="reduced",
        ).Q
    else:
        projection = torch.linalg.qr(
            torch.randn(projection_shape[1], projection_shape[0]),
            mode="reduced",
        ).Q.transpose(0, 1)
    gradient = torch.randn(*gradient_shape)
    delta = torch.randn(*delta_shape)

    rotated = rotate_orthogonal_projection(
        projection,
        rate=1e-3,
        seed=5,
        direction="loss_lowering",
        gradient=gradient,
        delta=delta,
    )
    old_decoded = (
        delta.matmul(projection.transpose(0, 1))
        if projection_shape[0] >= projection_shape[1]
        else projection.transpose(0, 1).matmul(delta)
    )
    new_decoded = (
        delta.matmul(rotated.transpose(0, 1))
        if projection_shape[0] >= projection_shape[1]
        else rotated.transpose(0, 1).matmul(delta)
    )

    old_linear_loss = (gradient * old_decoded).sum()
    new_linear_loss = (gradient * new_decoded).sum()
    assert new_linear_loss < old_linear_loss


@pytest.mark.parametrize("optimizer_cls", [APOLLO, APOLLOCAME])
def test_apollo_rejects_lrsf_only_loss_lowering_refresh(optimizer_cls):
    parameter = torch.nn.Parameter(torch.randn(6, 4))
    kwargs = {
        "fallback": {"small_matrix": "apollo"},
        "orthogonal_refresh": {
            "rate": 0.01, "seed": 13, "direction": "loss_lowering",
        },
    }
    if optimizer_cls is APOLLOCAME:
        kwargs["came_backend"] = "torch"
    optimizer = optimizer_cls([parameter], **kwargs)
    parameter.grad = torch.randn_like(parameter)

    with pytest.raises(ValueError, match="only supported for LRSF"):
        optimizer.step()


@pytest.mark.parametrize("optimizer_cls", [APOLLO, APOLLOCAME])
def test_apollo_loss_directed_refresh_uses_current_gradient(optimizer_cls):
    parameter = torch.nn.Parameter(torch.randn(6, 4))
    kwargs = {
        "fallback": {"small_matrix": "apollo"},
        "orthogonal_refresh": {
            "rate": 0.01, "seed": 11, "direction": "loss_directed",
        },
    }
    if optimizer_cls is APOLLOCAME:
        kwargs["came_backend"] = "torch"
    optimizer = optimizer_cls([parameter], **kwargs)

    for _ in range(2):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

    assert optimizer.state[parameter]["orthogonal_refresh_count"] == 2
    assert torch.isfinite(parameter).all()


@pytest.mark.parametrize("optimizer_cls", [APOLLO, APOLLOCAME])
def test_apollo_update_norm_variance_cap_is_checkpoint_state(optimizer_cls):
    parameter = torch.nn.Parameter(torch.zeros(6, 4))
    kwargs = {
        "fallback": {"small_matrix": "apollo"},
        "norm_growth_limiter": False,
        "update_norm_variance_cap": 0.0,
    }
    if optimizer_cls is APOLLOCAME:
        kwargs["came_backend"] = "torch"
    optimizer = optimizer_cls([parameter], **kwargs)

    first_parameter = None
    for scale in (1.0, 10.0, 100.0):
        parameter.grad = torch.ones_like(parameter) * scale
        optimizer.step()
        if first_parameter is None:
            first_parameter = parameter.detach().clone()

    state = optimizer.state[parameter]
    assert state["update_norm_variance_count"] == 3
    assert "update_norm_variance_mean" in state
    assert "update_norm_variance_m2" in state
    assert int(state.get("update_norm_variance_capped_count", 0)) == (
        2 if optimizer_cls is APOLLOCAME else 0
    )
    if optimizer_cls is APOLLOCAME:
        # With a zero cap, this scale-invariant gradient produces the first
        # update norm on every step. This also guards against applying both
        # the capped and original update in one optimizer step.
        torch.testing.assert_close(parameter, first_parameter * 3)
    assert torch.isfinite(parameter).all()


@pytest.mark.parametrize("optimizer_cls", [APOLLO, APOLLOCAME])
def test_apollo_stochastic_smooth_refresh_selects_checkpoint_safe_branch(
    optimizer_cls,
):
    parameter = torch.nn.Parameter(torch.randn(6, 4))
    kwargs = {
        "fallback": {"small_matrix": "apollo"},
        "projection_refresh": {
            "mode": "smooth", "interval": 2, "window": 2,
            "mix": "stochastic",
        },
        "projection_refresh_state": "transport",
    }
    if optimizer_cls is APOLLOCAME:
        kwargs["came_backend"] = "torch"
    optimizer = optimizer_cls([parameter], **kwargs)
    for _ in range(3):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

    state = optimizer.state[parameter]
    assert state["refresh_stochastic_counter"] == 2
    assert state["refresh_stochastic_choice"] in {0, 1}
    assert torch.isfinite(parameter).all()


@pytest.mark.parametrize("optimizer_cls", [APOLLO, APOLLOCAME])
def test_apollo_ema_smooth_refresh_advances_exponential_weight(optimizer_cls):
    parameter = torch.nn.Parameter(torch.randn(6, 4))
    kwargs = {
        "fallback": {"small_matrix": "apollo"},
        "projection_refresh": {
            "mode": "smooth", "interval": 2, "window": 4,
            "mix": "ema",
        },
    }
    if optimizer_cls is APOLLOCAME:
        kwargs["came_backend"] = "torch"
    optimizer = optimizer_cls([parameter], **kwargs)
    for _ in range(2):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

    state = optimizer.state[parameter]
    assert state["refresh_active"] is True
    assert 0.0 < state["refresh_ema_weight"] < 1.0


def test_apollo_old_checkpoint_defaults_to_legacy_refresh_behavior():
    parameter = torch.nn.Parameter(torch.randn(6, 4))
    optimizer = APOLLO(
        [parameter], update_proj_gap=2,
        fallback={"small_matrix": "apollo"},
    )
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    checkpoint = copy.deepcopy(optimizer.state_dict())
    checkpoint["param_groups"][0].pop("projection_refresh", None)
    checkpoint["param_groups"][0].pop("projection_refresh_state", None)

    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = APOLLO(
        [restored_parameter], update_proj_gap=2,
        fallback={"small_matrix": "apollo"},
    )
    restored.load_state_dict(checkpoint)
    restored_parameter.grad = torch.randn_like(restored_parameter)
    restored.step()

    assert torch.isfinite(restored_parameter).all()
    assert restored.state[restored_parameter]["projection_seed"] != 0


@pytest.mark.parametrize("optimizer_cls", [APOLLO, APOLLOCAME])
def test_smooth_refresh_checkpoint_resume_matches_active_window(optimizer_cls):
    parameter = torch.nn.Parameter(torch.randn(6, 4))
    optimizer = optimizer_cls(
        [parameter],
        fallback={"small_matrix": "apollo"},
        projection_refresh={"mode": "smooth", "interval": 2, "window": 3},
        projection_refresh_state="transport",
    )
    gradients = [
        torch.full_like(parameter, 0.25),
        torch.full_like(parameter, -0.5),
        torch.full_like(parameter, 0.75),
        torch.full_like(parameter, -0.125),
    ]
    for gradient in gradients[:2]:
        parameter.grad = gradient
        optimizer.step()

    checkpoint = copy.deepcopy(optimizer.state_dict())
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = optimizer_cls(
        [restored_parameter],
        fallback={"small_matrix": "apollo"},
        projection_refresh={"mode": "smooth", "interval": 2, "window": 3},
        projection_refresh_state="transport",
    )
    restored.load_state_dict(checkpoint)

    for gradient in gradients[2:]:
        parameter.grad = gradient
        restored_parameter.grad = gradient.clone()
        optimizer.step()
        restored.step()

    torch.testing.assert_close(parameter, restored_parameter)
    original_state = optimizer.state[parameter]
    restored_state = restored.state[restored_parameter]
    assert original_state.keys() == restored_state.keys()
    for key in original_state:
        original_value = original_state[key]
        restored_value = restored_state[key]
        if isinstance(original_value, torch.Tensor):
            torch.testing.assert_close(original_value, restored_value)
        else:
            assert original_value == restored_value


def test_apollo_came_lrsf_shadow_refresh_keeps_delta_branches_separate():
    parameter = torch.nn.Parameter(torch.randn(16, 8))
    optimizer = APOLLOCAMELRSF(
        [parameter],
        lr=1e-3,
        rank=4,
        lrsf_rank=4,
        delta_refresh={"mode": "shadow", "interval": 2},
        update_proj_gap=2,
    )
    optimizer.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        parameter.square().mean().backward()
        optimizer.step()

    state = optimizer.state[parameter]
    assert state["shadow_active"] is True
    assert state["refresh_count"] == 1
    assert state["shadow_generation"] == 1
    assert state["lrsf_projection"].shape == (8, 4)
    assert state["lrsf_shadow_projection"].shape == (8, 4)
    assert state["lrsf_delta"].shape == (16, 4)
    assert state["lrsf_shadow_delta"].shape == (16, 4)
