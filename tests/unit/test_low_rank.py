import pytest
import torch
from torch import nn

from core.low_rank import (
    DoRALinear,
    GLULoRALinear,
    LoRALinear,
    LoHALinear,
    RGLULoRALinear,
    canonicalize_adapter_type,
    inject_adapter,
    inject_lora,
    materialize_adapter,
    mark_only_adapter_trainable,
    mark_only_lora_trainable,
    merge_lora,
    unmerge_lora,
)


def test_rglu_lora_uses_new_name_and_accepts_legacy_alias():
    assert canonicalize_adapter_type("rglu_lora") == "rglu_lora"
    assert canonicalize_adapter_type("residual_swiglu_loha") == "rglu_lora"
    model = nn.Sequential(nn.Linear(3, 2))
    inject_adapter(model, "residual_swiglu_loha", [r"^0$"], rank=1)
    assert isinstance(model[0], RGLULoRALinear)
    assert model[0].adapter_type == "rglu_lora"


def test_lora_is_identity_before_training_and_only_factors_are_trainable():
    torch.manual_seed(0)
    original = nn.Sequential(nn.Linear(5, 7), nn.ReLU(), nn.Linear(7, 3))
    model = nn.Sequential(nn.Linear(5, 7), nn.ReLU(), nn.Linear(7, 3))
    model.load_state_dict(original.state_dict())
    matched = inject_lora(
        model,
        [r"^0$", r"^2$"],
        rank=2,
        alpha=4,
    )
    trainable = mark_only_lora_trainable(model)

    assert matched == ["0", "2"]
    assert trainable == (2 * 5 + 7 * 2) + (2 * 7 + 3 * 2)
    input_value = torch.randn(4, 5)
    assert torch.allclose(model(input_value), original(input_value))
    assert all(
        parameter.requires_grad == ("lora_" in name)
        for name, parameter in model.named_parameters()
    )


def test_lora_merge_and_unmerge_preserve_forward_values():
    torch.manual_seed(1)
    layer = nn.Linear(6, 4)
    model = nn.Sequential(layer)
    inject_lora(model, [r"^0$"], rank=3, alpha=6)
    adapter = model[0]
    assert isinstance(adapter, LoRALinear)
    with torch.no_grad():
        adapter.lora_A.normal_()
        adapter.lora_B.normal_()
    inputs = torch.randn(3, 6)
    unmerged_output = model(inputs)

    merge_lora(model)
    merged_output = model(inputs)
    unmerge_lora(model)
    restored_output = model(inputs)

    assert torch.allclose(merged_output, unmerged_output, atol=1e-6, rtol=1e-6)
    assert torch.allclose(restored_output, unmerged_output, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("adapter_name", "adapter_class", "expected_trainable_multiplier"),
    [
        ("lora", LoRALinear, 1),
        ("dora", DoRALinear, 1),
        ("loha", LoHALinear, 2),
        ("glu_lora", GLULoRALinear, 2),
        ("rglu_lora", RGLULoRALinear, 2),
    ],
)
def test_all_adapters_start_as_identity_and_produce_finite_gradients(
    adapter_name, adapter_class, expected_trainable_multiplier,
):
    torch.manual_seed(2)
    reference = nn.Linear(5, 4)
    model = nn.Sequential(nn.Linear(5, 4))
    model[0].load_state_dict(reference.state_dict())
    inject_adapter(model, adapter_name, [r"^0$"], rank=2, alpha=4)
    trainable = mark_only_adapter_trainable(model)

    inputs = torch.randn(3, 5)
    assert isinstance(model[0], adapter_class)
    assert torch.allclose(model(inputs), reference(inputs))
    expected = (2 * 5 + 4 * 2) * expected_trainable_multiplier
    if adapter_name == "dora":
        expected += 4
    assert trainable == expected

    model(inputs).square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.parametrize(
    "adapter_name", ["dora", "loha", "glu_lora", "rglu_lora"],
)
def test_non_lora_adapters_merge_and_unmerge_preserve_forward_values(adapter_name):
    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(6, 4))
    inject_adapter(model, adapter_name, [r"^0$"], rank=2, alpha=4)
    mark_only_adapter_trainable(model)
    adapter = model[0]
    with torch.no_grad():
        for name, parameter in adapter.named_parameters(recurse=False):
            if name.startswith("lora_") or name == "magnitude":
                parameter.normal_()
    inputs = torch.randn(3, 6)
    before = model(inputs)
    from core.low_rank import merge_adapter, unmerge_adapter

    merge_adapter(model)
    merged = model(inputs)
    unmerge_adapter(model)
    restored = model(inputs)

    assert torch.allclose(merged, before, atol=1e-5, rtol=1e-5)
    assert torch.allclose(restored, before, atol=1e-5, rtol=1e-5)


def test_rglu_lora_uses_weight_space_hadamard_gate():
    layer = RGLULoRALinear(nn.Linear(2, 2, bias=False), rank=1)
    with torch.no_grad():
        layer.lora_A1.copy_(torch.tensor([[1.0, 2.0]]))
        layer.lora_B1.copy_(torch.tensor([[3.0], [4.0]]))
        layer.lora_A2.copy_(torch.tensor([[2.0, -1.0]]))
        layer.lora_B2.copy_(torch.tensor([[0.5], [-0.25]]))

    value = layer.lora_B1 @ layer.lora_A1
    gate = 1.0 + torch.nn.functional.silu(layer.lora_B2 @ layer.lora_A2)
    expected = value * gate

    assert torch.allclose(layer._delta_weight(), expected)


def test_glu_lora_uses_silu_without_residual_offset():
    layer = GLULoRALinear(nn.Linear(2, 2, bias=False), rank=1)
    with torch.no_grad():
        layer.lora_A1.copy_(torch.tensor([[1.0, 2.0]]))
        layer.lora_B1.copy_(torch.tensor([[3.0], [4.0]]))
        layer.lora_A2.copy_(torch.tensor([[2.0, -1.0]]))
        layer.lora_B2.copy_(torch.tensor([[0.5], [-0.25]]))

    value = layer.lora_B1 @ layer.lora_A1
    gate = torch.nn.functional.silu(layer.lora_B2 @ layer.lora_A2)
    expected = value * gate

    assert torch.allclose(layer._delta_weight(), expected)


def test_glu_lora_identity_initialization_keeps_gradient_path_alive():
    torch.manual_seed(5)
    layer = GLULoRALinear(
        nn.Linear(5, 4), rank=2, init_mode="identity",
    )

    assert torch.count_nonzero(layer.lora_B1) > 0
    assert torch.count_nonzero(layer.lora_B2) == 0
    assert torch.count_nonzero(layer._delta_weight()) == 0

    layer(torch.randn(3, 5)).square().mean().backward()
    assert layer.lora_B2.grad is not None
    assert torch.isfinite(layer.lora_B2.grad).all()
    assert torch.count_nonzero(layer.lora_B2.grad) > 0


def test_rglu_lora_initialization_modes_have_distinct_contracts():
    torch.manual_seed(4)
    identity = RGLULoRALinear(
        nn.Linear(5, 4), rank=2, init_mode="identity",
    )
    torch.manual_seed(4)
    warm = RGLULoRALinear(
        nn.Linear(5, 4), rank=2, init_mode="lora_warm",
    )

    assert torch.count_nonzero(identity.lora_B1) == 0
    assert torch.count_nonzero(warm.lora_B1) > 0
    assert torch.count_nonzero(identity.lora_B2) == 0
    assert torch.count_nonzero(warm.lora_B2) == 0
    assert torch.count_nonzero(identity._delta_weight()) == 0
    assert torch.count_nonzero(warm._delta_weight()) > 0


def test_non_identity_initialization_is_residual_only():
    with pytest.raises(ValueError, match="only supported"):
        inject_adapter(
            nn.Linear(2, 2), "lora", [r"^$"], rank=1,
            init_mode="lora_warm",
        )


def test_materialize_adapter_replaces_wrapper_and_preserves_forward():
    model = nn.Sequential(nn.Linear(3, 2))
    inject_lora(model, [r"^0$"], rank=1)
    with torch.no_grad():
        model[0].lora_A.fill_(0.5)
        model[0].lora_B.fill_(0.25)
    inputs = torch.randn(4, 3)
    expected = model(inputs)

    materialize_adapter(model)

    assert isinstance(model[0], nn.Linear)
    assert not hasattr(model[0], "lora_A")
    assert torch.allclose(model(inputs), expected, atol=1e-6, rtol=1e-6)


def test_lora_rejects_invalid_or_unmatched_targets():
    with pytest.raises(ValueError, match="positive"):
        LoRALinear(nn.Linear(2, 2), rank=0)
    with pytest.raises(ValueError, match="matched no"):
        inject_lora(nn.Linear(2, 2), [r"not_a_layer"], rank=1)
    with pytest.raises(ValueError, match="invalid"):
        inject_lora(nn.Linear(2, 2), ["["], rank=1)


def test_adapter_name_requires_positive_rank():
    with pytest.raises(ValueError, match="positive adapter rank"):
        inject_adapter(nn.Linear(2, 2), "dora", [r"^$"], rank=0)
