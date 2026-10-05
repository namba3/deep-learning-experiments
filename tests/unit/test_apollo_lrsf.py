import copy

import torch

from optimizers import APOLLOCAME, APOLLOCAMELRSF


def _fallback():
    return {
        "1d": "came",
        "small_matrix": "auto",
        "state_margin": 1.0,
        "min_savings_bytes": 0,
    }


def test_apollo_came_lrsf_keeps_update_and_delta_projections_separate():
    torch.manual_seed(4)
    parameter = torch.nn.Parameter(torch.randn(16, 16))
    optimizer = APOLLOCAMELRSF(
        [parameter], lr=1e-3, rank=4, lrsf_rank=2, weight_decay=0.0,
        fallback=_fallback(),
    )
    optimizer.train()
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert state["backend"] == "apollo"
    assert tuple(state["projection"].shape) == (16, 4)
    assert tuple(state["lrsf_projection"].shape) == (16, 2)
    assert tuple(state["lrsf_delta"].shape) == (16, 2)


def test_apollo_came_lrsf_fallback_is_ordinary_apollo_came():
    parameter = torch.nn.Parameter(torch.ones(4, 3))
    optimizer = APOLLOCAMELRSF(
        [parameter], lr=1e-3, rank=4, lrsf_rank=4, weight_decay=0.0,
        fallback=_fallback(),
    )
    optimizer.train()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    assert optimizer.state[parameter]["backend"] == "came"
    assert "lrsf_delta" not in optimizer.state[parameter]


def test_apollo_came_lrsf_fallback_matches_apollo_came():
    torch.manual_seed(12)
    came_parameter = torch.nn.Parameter(torch.randn(4, 3))
    lrsf_parameter = torch.nn.Parameter(came_parameter.detach().clone())
    came = APOLLOCAME(
        [came_parameter], lr=1e-3, weight_decay=0.0,
        fallback=_fallback(), came_backend="torch",
    )
    lrsf = APOLLOCAMELRSF(
        [lrsf_parameter], lr=1e-3, rank=4, lrsf_rank=4,
        weight_decay=0.0, fallback=_fallback(), came_backend="torch",
    )
    gradient = torch.randn_like(came_parameter)
    came_parameter.grad = gradient
    lrsf_parameter.grad = gradient.clone()

    came.step()
    lrsf.train()
    lrsf.step()

    assert torch.equal(came_parameter, lrsf_parameter)


def test_apollo_came_lrsf_checkpoint_round_trip_continues_identically():
    torch.manual_seed(18)
    kwargs = {
        "rank": 4, "lrsf_rank": 2, "weight_decay": 0.0,
        "fallback": _fallback(), "came_backend": "torch",
    }
    first = torch.nn.Parameter(torch.randn(16, 16))
    original = APOLLOCAMELRSF([first], lr=1e-3, **kwargs)
    original.train()
    for _ in range(3):
        original.zero_grad(set_to_none=True)
        first.square().mean().backward()
        original.step()

    restored_parameter = torch.nn.Parameter(first.detach().clone())
    restored = APOLLOCAMELRSF(
        [restored_parameter], lr=1e-3, **kwargs,
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


def test_apollo_came_lrsf_delta_refresh_is_configurable():
    torch.manual_seed(22)
    parameter = torch.nn.Parameter(torch.randn(16, 16))
    optimizer = APOLLOCAMELRSF(
        [parameter], lr=1e-3, rank=4, lrsf_rank=2, weight_decay=0.0,
        fallback=_fallback(), delta_refresh={
            "mode": "smooth", "interval": 2, "window": 2,
        },
    )
    optimizer.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        parameter.square().mean().backward()
        optimizer.step()

    state = optimizer.state[parameter]
    assert state["refresh_active"] is True
    assert "refresh_next_projection" in state
    optimizer.zero_grad(set_to_none=True)
    parameter.square().mean().backward()
    optimizer.step()
    assert optimizer.state[parameter]["refresh_active"] is False


def test_apollo_came_lrsf_stochastic_delta_refresh_is_configurable():
    torch.manual_seed(24)
    parameter = torch.nn.Parameter(torch.randn(16, 16))
    optimizer = APOLLOCAMELRSF(
        [parameter], lr=1e-3, rank=4, lrsf_rank=2, weight_decay=0.0,
        fallback=_fallback(), delta_refresh={
            "mode": "smooth", "interval": 2, "window": 2,
            "mix": "stochastic",
        },
    )
    optimizer.train()
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        parameter.square().mean().backward()
        optimizer.step()

    state = optimizer.state[parameter]
    assert state["refresh_stochastic_counter"] == 2
    assert state["refresh_stochastic_choice"] in {0, 1}


def test_apollo_came_lrsf_orthogonal_refresh_rotates_delta_projection():
    torch.manual_seed(29)
    parameter = torch.nn.Parameter(torch.randn(16, 16))
    optimizer = APOLLOCAMELRSF(
        [parameter], lr=1e-3, rank=4, lrsf_rank=2, weight_decay=0.0,
        fallback=_fallback(), orthogonal_refresh={"rate": 0.05, "seed": 37},
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
    torch.testing.assert_close(
        projections[-1].transpose(0, 1).matmul(projections[-1]),
        torch.eye(2),
        atol=1e-5,
        rtol=1e-5,
    )


def test_apollo_came_lrsf_loss_directed_refresh_uses_delta_projection():
    torch.manual_seed(30)
    parameter = torch.nn.Parameter(torch.randn(16, 16))
    optimizer = APOLLOCAMELRSF(
        [parameter], lr=1e-3, rank=4, lrsf_rank=2, weight_decay=0.0,
        fallback=_fallback(),
        orthogonal_refresh={
            "rate": 0.01, "seed": 38, "direction": "loss_directed",
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
