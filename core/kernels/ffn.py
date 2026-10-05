"""Reusable gated-FFN activation kernels.

The matrix multiplications around a gated FFN are intentionally left to
PyTorch/cuBLAS.  This module fuses only the elementwise
``silu(gate) * value`` operation, which avoids materializing the intermediate
activation between the two elementwise operations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # Triton is optional.
    triton = None
    tl = None


def _validate_inputs(gate: torch.Tensor, value: torch.Tensor) -> None:
    if gate.shape != value.shape:
        raise ValueError(
            "gated FFN inputs must have the same shape; "
            f"got gate={tuple(gate.shape)} value={tuple(value.shape)}"
        )
    if gate.device != value.device:
        raise ValueError("gated FFN inputs must be on the same device")
    if gate.dtype != value.dtype:
        raise ValueError(
            "gated FFN inputs must have the same dtype; "
            f"got gate={gate.dtype} value={value.dtype}"
        )


def gated_silu_naive(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Reference implementation of the gated FFN elementwise operation."""
    _validate_inputs(gate, value)
    return F.silu(gate) * value


def _triton_is_available(x: torch.Tensor) -> bool:
    return (
        triton is not None
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.numel() > 0
    )


def triton_available(x: torch.Tensor) -> bool:
    """Return whether the Triton gated activation can run for ``x``."""
    return _triton_is_available(x)


if triton is not None:

    @triton.jit
    def _gated_silu_forward_kernel(
        gate_ptr,
        value_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        program = tl.program_id(0)
        offsets = program * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        value = tl.load(value_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        output = sigmoid * gate * value
        tl.store(output_ptr + offsets, output, mask=mask)


class _TritonGatedSiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(gate)
        block_size = 256
        grid = (triton.cdiv(gate.numel(), block_size),)
        _gated_silu_forward_kernel[grid](  # pyright: ignore[reportIndexIssue]
            gate,
            value,
            output,
            gate.numel(),
            BLOCK_SIZE=block_size,
        )
        ctx.save_for_backward(gate, value)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate, value = ctx.saved_tensors
        # Keep the backward portable initially.  The two elementwise gradient
        # expressions are small compared with the surrounding Linear GEMMs,
        # while this keeps the custom kernel easy to validate.
        gate_float = gate.float()
        sigmoid = torch.sigmoid(gate_float)
        silu = gate_float * sigmoid
        grad_output_float = grad_output.float()
        grad_gate = (
            grad_output_float
            * value.float()
            * sigmoid
            * (1.0 + gate_float * (1.0 - sigmoid))
        ).to(dtype=gate.dtype)
        grad_value = (grad_output_float * silu).to(dtype=value.dtype)
        return grad_gate, grad_value


def gated_silu_triton(
    gate: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Run the fused Triton gated activation with a PyTorch backward."""
    _validate_inputs(gate, value)
    if not _triton_is_available(gate):
        raise RuntimeError(
            "Gated FFN Triton backend requires CUDA, Triton, and float16, "
            "bfloat16, or float32 inputs"
        )
    # The kernel uses flat addressing.  These copies remain differentiable and
    # allow callers to pass views produced by chunk/transpose operations.
    gate = gate.contiguous()
    value = value.contiguous()
    return _TritonGatedSiLUFunction.apply(gate, value)


def gated_silu(
    gate: torch.Tensor,
    value: torch.Tensor,
    *,
    backend: str = "auto",
) -> torch.Tensor:
    """Dispatch the gated FFN activation to PyTorch or Triton."""
    if backend not in {"auto", "torch", "naive", "triton"}:
        raise ValueError(f"unknown gated FFN backend: {backend}")
    if backend in {"torch", "naive"}:
        return gated_silu_naive(gate, value)
    if backend == "triton":
        return gated_silu_triton(gate, value)
    if _triton_is_available(gate):
        return gated_silu_triton(gate, value)
    return gated_silu_naive(gate, value)


def gated_ffn_naive(
    x: torch.Tensor,
    input_weight: torch.Tensor,
    input_bias: torch.Tensor | None,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor | None,
) -> torch.Tensor:
    """Reference GatedFFN with ordinary PyTorch Linear operations."""
    if x.shape[-1] != input_weight.shape[1]:
        raise ValueError("GatedFFN input width does not match input weight")
    if input_weight.shape[0] % 2:
        raise ValueError("GatedFFN input projection must have an even output width")
    hidden = input_weight.shape[0] // 2
    if output_weight.shape[1] != hidden:
        raise ValueError("GatedFFN output weight width does not match hidden width")
    projected = F.linear(x, input_weight, input_bias)
    gate, value = projected.chunk(2, dim=-1)
    return F.linear(
        gated_silu_naive(gate, value), output_weight, output_bias,
    )


if triton is not None:

    @triton.jit
    def _gated_ffn_forward_kernel(
        x_ptr,
        input_weight_ptr,
        input_bias_ptr,
        output_ptr,
        rows,
        input_features,
        hidden_features,
        HAS_INPUT_BIAS: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
        BLOCK_M: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
        BLOCK_N: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
        BLOCK_K: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        block_m = tl.program_id(0)
        block_n = tl.program_id(1)
        row = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        col = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = row < rows
        col_mask = col < hidden_features
        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_value = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in tl.range(0, input_features, BLOCK_K):
            k = k_start + tl.arange(0, BLOCK_K)
            x_mask = row_mask[:, None] & (k[None, :] < input_features)
            x_value = tl.load(
                x_ptr + row[:, None] * input_features + k[None, :],
                mask=x_mask,
                other=0.0,
            )
            weight_mask = col_mask[None, :] & (k[:, None] < input_features)
            gate_weight = tl.load(
                input_weight_ptr + col[None, :] * input_features + k[:, None],
                mask=weight_mask,
                other=0.0,
            )
            value_weight = tl.load(
                input_weight_ptr
                + (col[None, :] + hidden_features) * input_features
                + k[:, None],
                mask=weight_mask,
                other=0.0,
            )
            acc_gate += tl.dot(x_value, gate_weight)
            acc_value += tl.dot(x_value, value_weight)
        if HAS_INPUT_BIAS:
            gate_bias = tl.load(input_bias_ptr + col, mask=col_mask, other=0.0)
            value_bias = tl.load(
                input_bias_ptr + col + hidden_features,
                mask=col_mask,
                other=0.0,
            )
            acc_gate += gate_bias[None, :]
            acc_value += value_bias[None, :]
        sigmoid = 1.0 / (1.0 + tl.exp(-acc_gate))
        output = sigmoid * acc_gate * acc_value
        output_mask = row_mask[:, None] & col_mask[None, :]
        tl.store(
            output_ptr + row[:, None] * hidden_features + col[None, :],
            output,
            mask=output_mask,
        )


class _TritonGatedFFNFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
    ):
        rows, input_features = x.shape
        hidden_features = input_weight.shape[0] // 2
        hidden = torch.empty(
            (rows, hidden_features), device=x.device, dtype=x.dtype,
        )
        block_m = 128
        block_n = 64
        block_k = 32
        grid = (
            triton.cdiv(rows, block_m),
            triton.cdiv(hidden_features, block_n),
        )
        _gated_ffn_forward_kernel[grid](  # pyright: ignore[reportIndexIssue]
            x,
            input_weight,
            input_bias,
            hidden,
            rows,
            input_features,
            hidden_features,
            HAS_INPUT_BIAS=input_bias.numel() > 0,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
        )
        output = F.linear(
            hidden,
            output_weight,
            output_bias if output_bias.numel() > 0 else None,
        )
        ctx.save_for_backward(
            x, input_weight, output_weight, hidden, input_bias, output_bias,
        )
        ctx.input_bias_enabled = input_bias.numel() > 0
        ctx.output_bias_enabled = output_bias.numel() > 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            x, input_weight, output_weight, hidden, input_bias, output_bias,
        ) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        # ``output_weight`` is stored as [out_features, hidden_features],
        # while the incoming gradient is [rows, out_features].  The reverse
        # projection therefore uses its transpose.
        grad_hidden = F.linear(grad_output, output_weight.transpose(-2, -1))

        # Recompute the two pre-activation halves.  This avoids saving a
        # second hidden-width tensor from the fused forward, trading compute
        # for a smaller activation footprint.
        projected = F.linear(
            x,
            input_weight,
            input_bias if ctx.input_bias_enabled else None,
        )
        gate, value = projected.chunk(2, dim=-1)
        gate_float = gate.float()
        sigmoid = torch.sigmoid(gate_float)
        silu = gate_float * sigmoid
        grad_hidden_float = grad_hidden.float()
        grad_gate = (
            grad_hidden_float
            * value.float()
            * sigmoid
            * (1.0 + gate_float * (1.0 - sigmoid))
        ).to(dtype=grad_hidden.dtype)
        grad_value = (grad_hidden_float * silu).to(dtype=grad_hidden.dtype)
        grad_projected = torch.cat((grad_gate, grad_value), dim=-1)

        # ``input_weight`` is [2 * hidden_features, input_features].
        # Projecting the activation gradient back to x therefore also uses
        # the transposed weight.
        grad_x = F.linear(grad_projected, input_weight.transpose(-2, -1))
        grad_input_weight = torch.matmul(
            grad_projected.transpose(-2, -1), x,
        )
        grad_input_weight = grad_input_weight.reshape_as(input_weight)
        grad_input_bias = (
            grad_projected.sum(dim=tuple(range(grad_projected.ndim - 1)))
            if ctx.input_bias_enabled else None
        )
        grad_output_weight = torch.matmul(
            grad_output.transpose(-2, -1), hidden,
        ).reshape_as(output_weight)
        grad_output_bias = (
            grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))
            if ctx.output_bias_enabled else None
        )
        return (
            grad_x,
            grad_input_weight,
            grad_input_bias,
            grad_output_weight,
            grad_output_bias,
        )


def gated_ffn_triton(
    x: torch.Tensor,
    input_weight: torch.Tensor,
    input_bias: torch.Tensor | None,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor | None,
) -> torch.Tensor:
    """Fuse the first Linear, gate split, and gated activation with Triton."""
    if not _triton_is_available(x):
        raise RuntimeError(
            "GatedFFN Triton backend requires CUDA, Triton, and float16, "
            "bfloat16, or float32 inputs"
        )
    if x.ndim < 2:
        raise ValueError("GatedFFN input must have at least two dimensions")
    if input_weight.ndim != 2 or output_weight.ndim != 2:
        raise ValueError("GatedFFN weights must be rank-2 tensors")
    if input_weight.shape[0] % 2 or input_weight.shape[1] != x.shape[-1]:
        raise ValueError("GatedFFN input weight has incompatible shape")
    if output_weight.shape[1] != input_weight.shape[0] // 2:
        raise ValueError("GatedFFN output weight has incompatible shape")
    for parameter in (input_weight, output_weight):
        if parameter.device != x.device or parameter.dtype != x.dtype:
            raise ValueError("GatedFFN tensors must share device and dtype")
    if input_bias is None:
        input_bias = x.new_empty(0)
    if output_bias is None:
        output_bias = x.new_empty(0)
    for bias, expected in (
        (input_bias, input_weight.shape[0]),
        (output_bias, output_weight.shape[0]),
    ):
        if bias.numel() not in {0, expected}:
            raise ValueError("GatedFFN bias has incompatible shape")
        if bias.numel() and (bias.device != x.device or bias.dtype != x.dtype):
            raise ValueError("GatedFFN biases must share device and dtype")
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    input_weight = input_weight.contiguous()
    output_weight = output_weight.contiguous()
    input_bias = input_bias.contiguous()
    output_bias = output_bias.contiguous()
    output = _TritonGatedFFNFunction.apply(
        x_2d, input_weight, input_bias, output_weight, output_bias,
    )
    return output.reshape(*x.shape[:-1], output_weight.shape[0])


def gated_ffn(
    x: torch.Tensor,
    input_weight: torch.Tensor,
    input_bias: torch.Tensor | None,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor | None,
    *,
    backend: str = "auto",
) -> torch.Tensor:
    """Dispatch a complete GatedFFN implementation."""
    if backend not in {"auto", "torch", "naive", "triton"}:
        raise ValueError(f"unknown gated FFN backend: {backend}")
    if backend in {"torch", "naive"}:
        return gated_ffn_naive(
            x, input_weight, input_bias, output_weight, output_bias,
        )
    if backend == "triton":
        return gated_ffn_triton(
            x, input_weight, input_bias, output_weight, output_bias,
        )
    if _triton_is_available(x):
        return gated_ffn_triton(
            x, input_weight, input_bias, output_weight, output_bias,
        )
    return gated_ffn_naive(
        x, input_weight, input_bias, output_weight, output_bias,
    )


__all__ = [
    "gated_ffn",
    "gated_ffn_naive",
    "gated_ffn_triton",
    "gated_silu",
    "gated_silu_naive",
    "gated_silu_triton",
    "triton_available",
]
