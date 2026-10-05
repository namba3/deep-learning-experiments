from copy import deepcopy

import pytest

import torch

from optimizers.apollo import APOLLO, APOLLOFallbackPolicy, APOLLOCAME

from optimizers.came import CAME

from optimizers.schedulefree import AdamWScheduleFree

def _assign_gradient(parameter, value=1.0):
    parameter.grad = torch.full_like(parameter, value)

def test_apollo_fallback_policy_accepts_mapping_and_preserves_legacy_string():
    policy = APOLLOFallbackPolicy.from_value(
        {"1d": "came", "small_matrix": "auto", "min_savings_bytes": 16}
    )
    assert policy.one_dimensional == "came"
    assert policy.small_matrix == "auto"
    assert policy.min_savings_bytes == 16

    legacy = APOLLOFallbackPolicy.from_value("came")
    assert legacy.one_dimensional == "came"
    assert legacy.small_matrix == "apollo"

    schedule_free = APOLLOFallbackPolicy.from_value(
        {"1d": "adamw-sf", "small_matrix": "auto-sf"}
    )
    assert schedule_free.one_dimensional == "adamw-sf"
    assert schedule_free.small_matrix == "auto-sf"

@pytest.mark.parametrize("optimizer_type", [APOLLO, APOLLOCAME])
def test_apollo_checkpoint_roundtrip_preserves_fallback_backend(optimizer_type):
    policy = {"1d": "came", "small_matrix": "auto"}
    initial_parameters = [torch.randn(4, 3), torch.randn(32, 32)]
    parameters_a = [
        torch.nn.Parameter(parameter.clone())
        for parameter in initial_parameters
    ]
    parameters_b = [
        torch.nn.Parameter(parameter.clone())
        for parameter in initial_parameters
    ]
    kwargs = {"lr": 0.01, "weight_decay": 0.0, "rank": 8, "fallback": policy}
    if optimizer_type is APOLLOCAME:
        kwargs["came_backend"] = "torch"
    optimizer_a = optimizer_type(parameters_a, **kwargs)
    optimizer_b = optimizer_type(parameters_b, **kwargs)

    for _ in range(2):
        for parameter in parameters_a:
            parameter.grad = torch.ones_like(parameter)
        optimizer_a.step()

    # A real resume loads model weights separately from optimizer state.
    for parameter_a, parameter_b in zip(parameters_a, parameters_b):
        parameter_b.data.copy_(parameter_a.data)
    optimizer_b.load_state_dict(deepcopy(optimizer_a.state_dict()))
    assert optimizer_b.state[parameters_b[0]]["backend"] == "came"
    assert optimizer_b.state[parameters_b[1]]["backend"] == "apollo"

    for parameter_a, parameter_b in zip(parameters_a, parameters_b):
        gradient = torch.full_like(parameter_a, 0.25)
        parameter_a.grad = gradient.clone()
        parameter_b.grad = gradient.clone()
    optimizer_a.step()
    optimizer_b.step()

    for parameter_a, parameter_b in zip(parameters_a, parameters_b):
        torch.testing.assert_close(parameter_a, parameter_b)

@pytest.mark.parametrize("optimizer_type", [APOLLO, APOLLOCAME])
def test_apollo_auto_matrix_fallback_uses_full_came_for_small_matrix(optimizer_type):
    parameter = torch.nn.Parameter(torch.randn(4, 3))
    optimizer = optimizer_type(
        [parameter],
        lr=0.01,
        weight_decay=0.0,
        rank=8,
        fallback={"1d": "came", "small_matrix": "auto"},
    )
    _assign_gradient(parameter)

    optimizer.step()

    state = optimizer.state[parameter]
    assert state["backend"] == "came"
    assert "projection" not in state
    assert "exp_avg_sq_row" in state
    assert "exp_avg_res_row" in state
    actual_state_bytes = sum(
        value.numel() * value.element_size()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    assert optimizer.estimate_parameter_state_bytes(parameter) == actual_state_bytes

def test_apollo_auto_matrix_fallback_keeps_apollo_for_large_matrix():
    parameter = torch.nn.Parameter(torch.randn(32, 32))
    optimizer = APOLLO(
        [parameter],
        lr=0.01,
        weight_decay=0.0,
        rank=8,
        fallback={"1d": "came", "small_matrix": "auto"},
    )
    _assign_gradient(parameter)

    optimizer.step()

    state = optimizer.state[parameter]
    assert state["backend"] == "apollo"
    assert "projection" in state

def test_apollo_auto_schedule_free_matrix_fallback_compares_state_bytes():
    small = torch.nn.Parameter(torch.randn(2, 2, dtype=torch.bfloat16))
    large = torch.nn.Parameter(torch.randn(32, 32, dtype=torch.bfloat16))
    optimizer = APOLLO(
        [small, large],
        rank=8,
        fallback={"1d": "came", "small_matrix": "auto-sf"},
    )

    assert optimizer.estimate_parameter_state_bytes(small) == 2 * small.numel() * 2
    assert optimizer._select_backend(small, optimizer.state[small], optimizer.param_groups[0]) == "adamw-sf"
    assert optimizer._select_backend(large, optimizer.state[large], optimizer.param_groups[0]) == "apollo"

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_apollo_adamw_sf_fallback_matches_standalone_and_resumes_exactly(dtype):
    torch.manual_seed(123)
    initial_sf = torch.randn(7, dtype=dtype)
    initial_small_matrix = torch.randn(4, 3, dtype=dtype)
    # Keep the APOLLO path in FP32, as AMP training normally keeps model
    # parameters in FP32 while exercising the Schedule-Free fallback dtype.
    initial_matrix = torch.randn(32, 32, dtype=torch.float32)
    sf_parameter = torch.nn.Parameter(initial_sf.clone())
    small_matrix_parameter = torch.nn.Parameter(initial_small_matrix.clone())
    matrix_parameter = torch.nn.Parameter(initial_matrix.clone())
    reference_parameter = torch.nn.Parameter(initial_sf.clone())
    reference_small_matrix = torch.nn.Parameter(initial_small_matrix.clone())
    hybrid = APOLLO(
        [sf_parameter, small_matrix_parameter, matrix_parameter],
        lr=0.01,
        weight_decay=0.03,
        rank=4,
        seed=9,
        fallback={"1d": "adamw-sf", "small_matrix": "auto-sf"},
    )
    reference = AdamWScheduleFree(
        [reference_parameter, reference_small_matrix],
        lr=0.01,
        weight_decay=0.03,
        backend="torch",
    )

    for step in range(3):
        sf_gradient = (torch.linspace(-0.5, 0.7, 7) * (step + 1)).to(dtype)
        small_matrix_gradient = torch.full_like(
            small_matrix_parameter, 0.15 - step * 0.02,
        )
        matrix_gradient = torch.full_like(matrix_parameter, 0.1 + step * 0.01)
        sf_parameter.grad = sf_gradient.clone()
        reference_parameter.grad = sf_gradient.clone()
        small_matrix_parameter.grad = small_matrix_gradient.clone()
        reference_small_matrix.grad = small_matrix_gradient.clone()
        matrix_parameter.grad = matrix_gradient
        hybrid.train()
        reference.train()
        hybrid.step()
        reference.step()
        torch.testing.assert_close(sf_parameter, reference_parameter, atol=0, rtol=0)
        torch.testing.assert_close(
            small_matrix_parameter, reference_small_matrix, atol=0, rtol=0,
        )

        matrix_before_eval = matrix_parameter.detach().clone()
        hybrid.eval()
        reference.eval()
        torch.testing.assert_close(sf_parameter, reference_parameter, atol=0, rtol=0)
        torch.testing.assert_close(
            small_matrix_parameter, reference_small_matrix, atol=0, rtol=0,
        )
        torch.testing.assert_close(matrix_parameter, matrix_before_eval, atol=0, rtol=0)
        hybrid.train()
        reference.train()
        torch.testing.assert_close(sf_parameter, reference_parameter, atol=0, rtol=0)
        torch.testing.assert_close(
            small_matrix_parameter, reference_small_matrix, atol=0, rtol=0,
        )

    # Checkpoint weights are saved in eval mode; resuming must restore the
    # averaged weights and then return to train weights before the next step.
    hybrid.eval()
    reference.eval()
    torch.testing.assert_close(sf_parameter, reference_parameter, atol=0, rtol=0)
    torch.testing.assert_close(
        small_matrix_parameter, reference_small_matrix, atol=0, rtol=0,
    )
    resumed_sf = torch.nn.Parameter(sf_parameter.detach().clone())
    resumed_small_matrix = torch.nn.Parameter(
        small_matrix_parameter.detach().clone(),
    )
    resumed_matrix = torch.nn.Parameter(matrix_parameter.detach().clone())
    resumed = APOLLO(
        [resumed_sf, resumed_small_matrix, resumed_matrix],
        lr=0.01,
        weight_decay=0.03,
        rank=4,
        seed=9,
        fallback={"1d": "adamw-sf", "small_matrix": "auto-sf"},
    )
    resumed.load_state_dict(deepcopy(hybrid.state_dict()))
    assert not resumed.param_groups[0]["sf_train_mode"]
    resumed_reference_parameter = torch.nn.Parameter(
        reference_parameter.detach().clone(),
    )
    resumed_reference_small_matrix = torch.nn.Parameter(
        reference_small_matrix.detach().clone(),
    )
    resumed_reference = AdamWScheduleFree(
        [resumed_reference_parameter, resumed_reference_small_matrix],
        lr=0.01,
        weight_decay=0.03,
        backend="torch",
    )
    resumed_reference.load_state_dict(deepcopy(reference.state_dict()))
    resumed.train()
    resumed_reference.train()
    hybrid.train()

    next_sf_gradient = torch.linspace(0.2, -0.4, 7)
    next_sf_gradient = next_sf_gradient.to(dtype)
    next_small_matrix_gradient = torch.full_like(resumed_small_matrix, -0.03)
    next_matrix_gradient = torch.full_like(resumed_matrix, -0.07)
    sf_parameter.grad = next_sf_gradient.clone()
    resumed_sf.grad = next_sf_gradient.clone()
    small_matrix_parameter.grad = next_small_matrix_gradient.clone()
    resumed_small_matrix.grad = next_small_matrix_gradient.clone()
    resumed_reference_parameter.grad = next_sf_gradient.clone()
    resumed_reference_small_matrix.grad = next_small_matrix_gradient.clone()
    matrix_parameter.grad = next_matrix_gradient.clone()
    resumed_matrix.grad = next_matrix_gradient.clone()
    hybrid.step()
    resumed.step()
    resumed_reference.step()

    # Resume transitions from checkpointed eval weights using the same
    # Schedule-Free contract as standalone; low-precision eval weights may
    # not reconstruct the uninterrupted training weights bit-for-bit.
    torch.testing.assert_close(
        resumed_sf, resumed_reference_parameter, atol=0, rtol=0,
    )
    torch.testing.assert_close(
        resumed_small_matrix, resumed_reference_small_matrix, atol=0, rtol=0,
    )
    torch.testing.assert_close(matrix_parameter, resumed_matrix, atol=0, rtol=0)
    assert hybrid.param_groups[0]["sf_k"] == resumed.param_groups[0]["sf_k"]

def test_apollo_matrix_fallback_matches_full_came_update():
    initial = torch.randn(4, 3)
    gradient = torch.randn_like(initial)
    fallback_parameter = torch.nn.Parameter(initial.clone())
    came_parameter = torch.nn.Parameter(initial.clone())
    fallback = APOLLO(
        [fallback_parameter],
        lr=0.01,
        weight_decay=0.0,
        norm_growth_limiter=False,
        fallback={"small_matrix": "came"},
    )
    reference = CAME(
        [came_parameter], lr=0.01, weight_decay=0.0, backend="torch",
    )
    fallback_parameter.grad = gradient.clone()
    came_parameter.grad = gradient.clone()

    fallback.step()
    reference.step()

    torch.testing.assert_close(
        fallback_parameter, came_parameter, atol=1e-6, rtol=1e-6,
    )

@pytest.mark.parametrize("optimizer_type", [APOLLO, APOLLOCAME])
def test_apollo_matrix_came_fallback_matches_came_with_limiter_default(
    optimizer_type,
):
    initial = torch.randn(4, 3)
    fallback_parameter = torch.nn.Parameter(initial.clone())
    came_parameter = torch.nn.Parameter(initial.clone())
    fallback = optimizer_type(
        [fallback_parameter],
        lr=0.01,
        weight_decay=0.0,
        fallback={"small_matrix": "came"},
    )
    reference = CAME(
        [came_parameter], lr=0.01, weight_decay=0.0, backend="torch",
    )

    for _ in range(2):
        gradient = torch.randn_like(initial)
        fallback_parameter.grad = gradient.clone()
        came_parameter.grad = gradient.clone()
        fallback.step()
        reference.step()

    torch.testing.assert_close(
        fallback_parameter, came_parameter, atol=1e-6, rtol=1e-6,
    )
    fallback_state = fallback.state[fallback_parameter]
    came_state = reference.state[came_parameter]
    assert fallback_state["backend"] == "came"
    assert "scaled_grad_norm" not in fallback_state
    assert fallback.estimate_parameter_state_bytes(fallback_parameter) == sum(
        value.numel() * value.element_size()
        for value in came_state.values()
        if torch.is_tensor(value)
    )
