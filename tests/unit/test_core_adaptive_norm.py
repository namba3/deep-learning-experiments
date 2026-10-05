import pytest
import torch

from core.adaptive_norm import AdaRMSScaleProjection, ScaleOnlyAdaRMSNorm


def _reference_rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)).to(x.dtype)


def test_zero_initialized_projection_starts_at_base_rmsnorm() -> None:
    torch.manual_seed(3)
    x = torch.randn(2, 5, 8)
    condition = torch.randn(2, 4)
    projection = AdaRMSScaleProjection(4, 8)
    norm = ScaleOnlyAdaRMSNorm(8, elementwise_affine=False)

    scale = projection(condition)
    actual = norm(x, scale)

    assert scale.shape == (2, 8)
    assert torch.count_nonzero(scale) == 0
    torch.testing.assert_close(actual, _reference_rms(x))


@pytest.mark.parametrize("shape", [(1, 8), (2, 3, 8), (2, 2, 3, 8)])
def test_scale_broadcasts_over_all_non_batch_axes(shape: tuple[int, ...]) -> None:
    x = torch.randn(shape)
    scale = torch.randn(shape[0], shape[-1])
    norm = ScaleOnlyAdaRMSNorm(shape[-1], elementwise_affine=False)

    actual = norm(x, scale)
    expected = _reference_rms(x) * (1 + scale.reshape((shape[0],) + (1,) * (len(shape) - 2) + (shape[-1],)))

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_adaptive_norm_low_precision_dtype_and_gradients(dtype: torch.dtype) -> None:
    torch.manual_seed(31)
    x = torch.randn(2, 3, 5, 8, dtype=dtype, requires_grad=True)
    scale = torch.randn(2, 8, dtype=dtype, requires_grad=True)
    norm = ScaleOnlyAdaRMSNorm(8).to(dtype=dtype)

    actual = norm(x, scale)
    expected = _reference_rms(x) * (1 + scale.reshape(2, 1, 1, 8))
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()

    actual.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert scale.grad is not None and torch.isfinite(scale.grad).all()
    assert norm.weight is not None and norm.weight.grad is not None
    assert torch.isfinite(norm.weight.grad).all()


def test_adaptive_norm_propagates_gradients() -> None:
    x = torch.randn(2, 3, 8, requires_grad=True)
    condition = torch.randn(2, 4, requires_grad=True)
    projection = AdaRMSScaleProjection(4, 8)
    norm = ScaleOnlyAdaRMSNorm(8)

    norm(x, projection(condition)).square().mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert condition.grad is not None and torch.isfinite(condition.grad).all()
    assert projection.proj.weight.grad is not None
    assert torch.isfinite(projection.proj.weight.grad).all()


@pytest.mark.parametrize(
    ("x", "scale"),
    [
        (torch.randn(2, 8), torch.randn(8)),
        (torch.randn(2, 7), torch.randn(2, 8)),
        (torch.randn(2, 3, 8), torch.randn(1, 8)),
    ],
)
def test_adaptive_norm_rejects_invalid_shapes(x: torch.Tensor, scale: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        ScaleOnlyAdaRMSNorm(8)(x, scale)
