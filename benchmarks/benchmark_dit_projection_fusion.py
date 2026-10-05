"""Benchmark separate versus packed same-input Linear projections.

This is a projection microbenchmark, not a full DiT training benchmark. It
compares forward+backward for Q/K/V and SwiGLU input projections with identical
weights. Run on an otherwise idle target GPU before considering a fused default.

Example:
    python3 -m benchmarks.benchmark_dit_projection_fusion --device cuda --dtype bfloat16
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F


class SeparateQKV(nn.Module):
    def __init__(self, width: int, kv_width: int, bias: bool) -> None:
        super().__init__()
        self.q = nn.Linear(width, width, bias=bias)
        self.k = nn.Linear(width, kv_width, bias=bias)
        self.v = nn.Linear(width, kv_width, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.q(x), self.k(x), self.v(x)), dim=-1)


class PackedQKV(nn.Module):
    def __init__(self, width: int, kv_width: int, bias: bool) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width + 2 * kv_width, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SeparateSwiGLUInput(nn.Module):
    def __init__(self, width: int, hidden: int, bias: bool) -> None:
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=bias)
        self.value = nn.Linear(width, hidden, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.gate(x)) * self.value(x)


class PackedSwiGLUInput(nn.Module):
    def __init__(self, width: int, hidden: int, bias: bool) -> None:
        super().__init__()
        self.proj = nn.Linear(width, 2 * hidden, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.proj(x).chunk(2, dim=-1)
        return F.silu(gate) * value


def _copy_linear(source: nn.Linear, target_weight: torch.Tensor, target_bias: torch.Tensor | None) -> None:
    with torch.no_grad():
        target_weight.copy_(source.weight)
        if source.bias is not None:
            if target_bias is None:
                raise ValueError("fused projection is missing a source bias")
            target_bias.copy_(source.bias)


def _copy_separate_qkv_to_packed(source: SeparateQKV, target: PackedQKV) -> None:
    _copy_linear(
        source.q,
        target.proj.weight[: source.q.out_features],
        None if target.proj.bias is None else target.proj.bias[: source.q.out_features],
    )
    q_end = source.q.out_features
    k_end = q_end + source.k.out_features
    _copy_linear(
        source.k,
        target.proj.weight[q_end:k_end],
        None if target.proj.bias is None else target.proj.bias[q_end:k_end],
    )
    _copy_linear(
        source.v,
        target.proj.weight[k_end:],
        None if target.proj.bias is None else target.proj.bias[k_end:],
    )


def _copy_separate_swiglu_to_packed(
    source: SeparateSwiGLUInput,
    target: PackedSwiGLUInput,
) -> None:
    hidden = source.gate.out_features
    _copy_linear(source.gate, target.proj.weight[:hidden],
                 None if target.proj.bias is None else target.proj.bias[:hidden])
    _copy_linear(source.value, target.proj.weight[hidden:],
                 None if target.proj.bias is None else target.proj.bias[hidden:])


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parameter_gradient_pairs(
    reference: nn.Module, packed: nn.Module,
) -> list[tuple[torch.Tensor | None, torch.Tensor | None]]:
    if isinstance(reference, SeparateQKV) and isinstance(packed, PackedQKV):
        modules = (reference.q, reference.k, reference.v)
        pairs = []
        offset = 0
        for module in modules:
            end = offset + module.out_features
            pairs.append((module.weight.grad, packed.proj.weight.grad[offset:end]))
            if module.bias is not None:
                pairs.append((module.bias.grad, packed.proj.bias.grad[offset:end]))
            offset = end
        return pairs
    if isinstance(reference, SeparateSwiGLUInput) and isinstance(packed, PackedSwiGLUInput):
        hidden = reference.gate.out_features
        modules = ((reference.gate, slice(0, hidden)), (reference.value, slice(hidden, None)))
        pairs = []
        for module, part in modules:
            pairs.append((module.weight.grad, packed.proj.weight.grad[part]))
            if module.bias is not None:
                pairs.append((module.bias.grad, packed.proj.bias.grad[part]))
        return pairs
    raise TypeError(f"unsupported projection pair: {type(reference).__name__}, {type(packed).__name__}")


def _max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return (left.float() - right.float()).abs().max().item()


def _measure(
    module: nn.Module,
    x: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> tuple[float, int | None, int | None]:
    module.train()
    for _ in range(warmup):
        module.zero_grad(set_to_none=True)
        x.grad = None
        module(x).float().square().mean().backward()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    durations = []
    if device.type == "cuda":
        for _ in range(iterations):
            module.zero_grad(set_to_none=True)
            x.grad = None
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            module(x).float().square().mean().backward()
            end.record()
            durations.append((start, end))
        _sync(device)
        times = [start.elapsed_time(end) for start, end in durations]
    else:
        times = []
        for _ in range(iterations):
            module.zero_grad(set_to_none=True)
            x.grad = None
            start = time.perf_counter()
            module(x).float().square().mean().backward()
            times.append((time.perf_counter() - start) * 1000)

    allocated = reserved = None
    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
    return statistics.median(times), allocated, reserved


def _report_pair(
    label: str,
    reference: nn.Module,
    packed: nn.Module,
    x_shape: tuple[int, int, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> None:
    reference = reference.to(device=device, dtype=dtype)
    packed = packed.to(device=device, dtype=dtype)
    x_ref = torch.randn(x_shape, device=device, dtype=dtype, requires_grad=True)
    x_packed = x_ref.detach().clone().requires_grad_()
    reference.train()
    packed.train()

    y_ref = reference(x_ref)
    y_packed = packed(x_packed)
    torch.testing.assert_close(
        y_ref.float(), y_packed.float(),
        rtol=2e-2 if dtype != torch.float32 else 1e-5,
        atol=2e-2 if dtype != torch.float32 else 1e-6,
    )
    y_ref.float().square().mean().backward()
    y_packed.float().square().mean().backward()
    rtol = 2e-2 if dtype != torch.float32 else 1e-5
    atol = 2e-2 if dtype != torch.float32 else 1e-6
    torch.testing.assert_close(x_ref.grad, x_packed.grad, rtol=rtol, atol=atol)
    parameter_pairs = _parameter_gradient_pairs(reference, packed)
    for separate_grad, packed_grad in parameter_pairs:
        if separate_grad is None or packed_grad is None:
            raise RuntimeError("a projection parameter did not receive a gradient")
        torch.testing.assert_close(separate_grad, packed_grad, rtol=rtol, atol=atol)
    input_grad_error = _max_error(x_ref.grad, x_packed.grad)
    parameter_grad_error = max(
        _max_error(left, right)
        for left, right in parameter_pairs
        if left is not None and right is not None
    )
    output_error = _max_error(y_ref, y_packed)

    separate_ms, separate_alloc, separate_reserved = _measure(
        reference, x_ref, warmup=warmup, iterations=iterations, device=device,
    )
    packed_ms, packed_alloc, packed_reserved = _measure(
        packed, x_packed, warmup=warmup, iterations=iterations, device=device,
    )
    ratio = separate_ms / packed_ms if packed_ms else float("inf")
    print(
        f"{label}: output_max_abs={output_error:.6g} "
        f"input_grad_max_abs={input_grad_error:.6g} "
        f"parameter_grad_max_abs={parameter_grad_error:.6g}"
    )
    print(
        f"  separate median_ms={separate_ms:.4f} "
        f"peak_allocated={separate_alloc} peak_reserved={separate_reserved}"
    )
    print(
        f"  packed   median_ms={packed_ms:.4f} "
        f"peak_allocated={packed_alloc} peak_reserved={packed_reserved} "
        f"speedup={ratio:.3f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--ff-hidden", type=int, default=4096)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    if min(args.batch, args.tokens, args.width, args.heads, args.kv_heads, args.ff_hidden) <= 0:
        parser.error("batch, token count, widths, and head counts must be positive")
    if args.width % args.heads or args.heads % args.kv_heads:
        parser.error("width must divide heads and heads must be divisible by kv-heads")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be non-negative and iterations must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    dtype = getattr(torch, args.dtype)
    head_dim = args.width // args.heads
    kv_width = args.kv_heads * head_dim
    shape = (args.batch, args.tokens, args.width)

    print(f"device={device} dtype={dtype} tokens={shape} heads={args.heads}/{args.kv_heads}")
    torch.manual_seed(0)
    qkv_ref = SeparateQKV(args.width, kv_width, args.bias)
    qkv_packed = PackedQKV(args.width, kv_width, args.bias)
    _copy_separate_qkv_to_packed(qkv_ref, qkv_packed)
    _report_pair(
        "QKV projections", qkv_ref, qkv_packed, shape,
        dtype=dtype, device=device, warmup=args.warmup, iterations=args.iterations,
    )

    torch.manual_seed(1)
    mlp_ref = SeparateSwiGLUInput(args.width, args.ff_hidden, args.bias)
    mlp_packed = PackedSwiGLUInput(args.width, args.ff_hidden, args.bias)
    _copy_separate_swiglu_to_packed(mlp_ref, mlp_packed)
    _report_pair(
        "SwiGLU input projections", mlp_ref, mlp_packed, shape,
        dtype=dtype, device=device, warmup=args.warmup, iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
