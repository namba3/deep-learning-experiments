import pytest
import torch

from core.kernels import (
    apply_rope,
    apply_rope_naive,
    gated_ffn,
    gated_ffn_naive,
    gated_silu,
    gated_silu_naive,
    rms_norm,
    rms_norm_naive,
)


def test_rms_norm_naive_matches_torch_reference_and_backward():
    x = torch.randn(2, 3, 8, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(8, dtype=torch.float32, requires_grad=True)
    x_reference = x.detach().clone().requires_grad_()
    weight_reference = weight.detach().clone().requires_grad_()

    actual = rms_norm_naive(x, weight, eps=1e-6)
    expected = torch.nn.functional.rms_norm(
        x_reference, (8,), weight=weight_reference, eps=1e-6,
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

    actual.square().mean().backward()
    expected.square().mean().backward()
    assert torch.allclose(x.grad, x_reference.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        weight.grad, weight_reference.grad, atol=1e-6, rtol=1e-6,
    )


def test_rms_norm_dispatcher_works_on_cpu():
    x = torch.randn(2, 4, 8)
    weight = torch.randn(8)
    actual = rms_norm(x, weight, backend="auto")
    expected = rms_norm_naive(x, weight)
    assert actual.shape == x.shape
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_rope_naive_matches_inverse_rotation_gradient():
    x = torch.randn(2, 3, 5, 8, requires_grad=True)
    cos = torch.randn(5, 4)
    sin = torch.randn(5, 4)
    output = apply_rope_naive(x, cos, sin)
    output.sum().backward()
    expected_gradient = apply_rope_naive(
        torch.ones_like(x), cos, -sin,
    )
    assert torch.allclose(x.grad, expected_gradient, atol=1e-6, rtol=1e-6)


def test_rope_dispatcher_works_on_cpu_and_supports_non_square_token_counts():
    x = torch.randn(2, 3, 6, 8)
    cos = torch.randn(6, 4)
    sin = torch.randn(6, 4)
    actual = apply_rope(x, cos, sin, backend="auto")
    expected = apply_rope_naive(x, cos, sin)
    assert actual.shape == x.shape
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("backend", ["naive", "auto"])
def test_rope_backward_accepts_inference_mode_tables(backend: str) -> None:
    x = torch.randn(2, 3, 5, 8, requires_grad=True)
    with torch.inference_mode():
        cos = torch.randn(5, 4)
        sin = torch.randn(5, 4)

    output = apply_rope(x, cos, sin, backend=backend)
    output.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_rope_rejects_incompatible_tables():
    with pytest.raises(ValueError, match="RoPE tables"):
        apply_rope_naive(torch.randn(1, 2, 4), torch.ones(3, 1), torch.ones(3, 1))


def test_gated_silu_matches_torch_reference_and_backward():
    gate = torch.randn(2, 3, 8, requires_grad=True)
    value = torch.randn(2, 3, 8, requires_grad=True)
    gate_reference = gate.detach().clone().requires_grad_()
    value_reference = value.detach().clone().requires_grad_()

    actual = gated_silu(gate, value, backend="torch")
    expected = gated_silu_naive(gate_reference, value_reference)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

    actual.square().mean().backward()
    expected.square().mean().backward()
    assert torch.allclose(gate.grad, gate_reference.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(value.grad, value_reference.grad, atol=1e-6, rtol=1e-6)


def test_gated_silu_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="same shape"):
        gated_silu(torch.randn(2, 4), torch.randn(2, 5), backend="torch")


def test_gated_ffn_matches_reference_and_backward():
    x = torch.randn(2, 3, 8, requires_grad=True)
    input_weight = torch.randn(12, 8, requires_grad=True)
    input_bias = torch.randn(12, requires_grad=True)
    output_weight = torch.randn(8, 6, requires_grad=True)
    output_bias = torch.randn(8, requires_grad=True)
    actual_inputs = [
        x, input_weight, input_bias, output_weight, output_bias,
    ]
    reference_inputs = [value.detach().clone().requires_grad_() for value in actual_inputs]

    actual = gated_ffn(*actual_inputs, backend="torch")
    expected = gated_ffn_naive(*reference_inputs)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

    actual.square().mean().backward()
    expected.square().mean().backward()
    for value, reference in zip(actual_inputs, reference_inputs):
        assert torch.allclose(value.grad, reference.grad, atol=1e-6, rtol=1e-6)


def test_core_kernel_explicit_triton_backend_rejects_cpu_inputs():
    if torch.cuda.is_available():
        pytest.skip("This test checks the CPU-side explicit-backend contract")

    x = torch.randn(2, 3, 7)
    weight = torch.randn(7)
    cos = torch.randn(5, 5)
    sin = torch.randn(5, 5)
    input_weight = torch.randn(10, 7)
    input_bias = torch.randn(10)
    output_weight = torch.randn(6, 5)
    output_bias = torch.randn(6)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        rms_norm(x, weight, backend="triton")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        apply_rope(x.unsqueeze(1), cos, sin, backend="triton")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        gated_silu(x, x, backend="triton")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        gated_ffn(
            x, input_weight, input_bias, output_weight, output_bias,
            backend="triton",
        )


def test_rms_norm_triton_validates_weight_contract_before_launch():
    x = torch.randn(2, 3, 8, dtype=torch.float32)

    with pytest.raises(ValueError, match="weight shape"):
        rms_norm(x, torch.randn(7), backend="triton")
    with pytest.raises(ValueError, match="same dtype"):
        rms_norm(x, torch.randn(8, dtype=torch.bfloat16), backend="triton")
    with pytest.raises(ValueError, match="same device"):
        rms_norm(x, torch.empty(8, device="meta"), backend="triton")

    non_contiguous_weight = torch.randn(8, 2)[:, 0]
    assert not non_contiguous_weight.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        rms_norm(x, non_contiguous_weight, backend="triton")


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [(torch.float32, 1e-6, 1e-6), (torch.bfloat16, 1e-2, 1e-2)],
)
def test_rms_norm_dispatcher_matches_reference_for_dtypes_and_rectangular_tokens(
    dtype, atol, rtol,
):
    # CPU native BF16 RMSNorm and the FP32-accumulating reference round at
    # different points; the BF16 tolerance records that expected difference.
    x = torch.randn(2, 3, 7, dtype=dtype, requires_grad=True)
    weight = torch.randn(7, dtype=dtype, requires_grad=True)
    x_reference = x.detach().clone().requires_grad_()
    weight_reference = weight.detach().clone().requires_grad_()

    actual = rms_norm(x, weight, eps=1e-6, backend="auto")
    expected = rms_norm_naive(x_reference, weight_reference, eps=1e-6)

    assert torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    assert torch.allclose(x.grad.float(), x_reference.grad.float(), atol=atol, rtol=rtol)
    assert torch.allclose(
        weight.grad.float(), weight_reference.grad.float(), atol=atol, rtol=rtol,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rope_dispatcher_matches_reference_with_odd_token_count(dtype):
    x = torch.randn(2, 3, 5, 10, dtype=dtype, requires_grad=True)
    cos = torch.randn(5, 5, dtype=torch.float32)
    sin = torch.randn(5, 5, dtype=torch.float32)
    x_reference = x.detach().clone().requires_grad_()

    actual = apply_rope(x, cos, sin, backend="auto")
    expected = apply_rope_naive(x_reference, cos, sin)

    assert torch.equal(actual, expected)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    assert torch.equal(x.grad, x_reference.grad)


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [(torch.float32, 1e-6, 1e-6), (torch.bfloat16, 3e-3, 3e-3)],
)
def test_gated_silu_dispatcher_matches_reference_for_bfloat16(dtype, atol, rtol):
    gate = torch.randn(2, 3, 7, dtype=dtype, requires_grad=True)
    value = torch.randn(2, 3, 7, dtype=dtype, requires_grad=True)
    gate_reference = gate.detach().clone().requires_grad_()
    value_reference = value.detach().clone().requires_grad_()

    actual = gated_silu(gate, value, backend="auto")
    expected = gated_silu_naive(gate_reference, value_reference)

    assert torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    assert torch.allclose(gate.grad.float(), gate_reference.grad.float(), atol=atol, rtol=rtol)
    assert torch.allclose(value.grad.float(), value_reference.grad.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [(torch.float32, 1e-6, 1e-6), (torch.bfloat16, 5e-3, 5e-3)],
)
def test_gated_ffn_dispatcher_matches_reference_for_rectangular_tokens(
    dtype, atol, rtol,
):
    x = torch.randn(2, 3, 7, dtype=dtype, requires_grad=True)
    input_weight = torch.randn(10, 7, dtype=dtype, requires_grad=True)
    input_bias = torch.randn(10, dtype=dtype, requires_grad=True)
    output_weight = torch.randn(6, 5, dtype=dtype, requires_grad=True)
    output_bias = torch.randn(6, dtype=dtype, requires_grad=True)
    actual_inputs = [x, input_weight, input_bias, output_weight, output_bias]
    reference_inputs = [value.detach().clone().requires_grad_() for value in actual_inputs]

    actual = gated_ffn(*actual_inputs, backend="auto")
    expected = gated_ffn_naive(*reference_inputs)

    assert actual.shape == (2, 3, 6)
    assert torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    for value, reference in zip(actual_inputs, reference_inputs):
        assert torch.allclose(value.grad.float(), reference.grad.float(), atol=atol, rtol=rtol)
