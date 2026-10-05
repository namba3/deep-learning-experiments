"""Compare PyTorch SDPA with an explicit reference attention implementation.

Example:
    python3 -m benchmarks.benchmark_sdpa_attention --device cuda --dtype bfloat16

The explicit path is for comparison only. SDPA's selected backend is reported
from profiler event names; backend availability depends on PyTorch and hardware.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F


def explicit_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Reference attention for Q/K/V shaped (B, H, sequence, head_dim)."""
    scale = q.shape[-1] ** -0.5
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    probs = logits.softmax(dim=-1).to(dtype=v.dtype)
    return torch.matmul(probs, v)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> tuple[float, int | None, int | None]:
    samples = []
    for _ in range(warmup):
        fn(q, k, v).float().mean().backward()
        q.grad = k.grad = v.grad = None
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(iterations):
        start = time.perf_counter()
        fn(q, k, v).float().mean().backward()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1000)
        q.grad = k.grad = v.grad = None
    allocated = reserved = None
    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
    return statistics.median(samples), allocated, reserved


def _profile_backend(fn, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> list[str]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if q.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as profile:
        fn(q, k, v)
    names = sorted(
        {
            event.key
            for event in profile.key_averages()
            if "scaled_dot_product" in event.key.lower() or "flash_attention" in event.key.lower()
        }
    )
    return names or ["backend event not identified"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--query-tokens", type=int, default=256)
    parser.add_argument("--key-tokens", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if min(args.batch, args.heads, args.query_tokens, args.key_tokens, args.head_dim) <= 0:
        parser.error("batch, heads, token counts, and head dimension must be positive")
    dtype = getattr(torch, args.dtype)
    shape_q = (args.batch, args.heads, args.query_tokens, args.head_dim)
    shape_kv = (args.batch, args.heads, args.key_tokens, args.head_dim)
    q, k, v = [torch.randn(shape, device=device, dtype=dtype, requires_grad=True) for shape in (shape_q, shape_kv, shape_kv)]
    cases = {
        "sdpa": lambda q_, k_, v_: F.scaled_dot_product_attention(q_, k_, v_),
        "explicit_fp32_logits": explicit_attention,
    }

    print(f"device={device} dtype={dtype} Q={shape_q} K/V={shape_kv}")
    for name, fn in cases.items():
        latency, allocated, reserved = _measure(
            fn,
            q,
            k,
            v,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )
        print(
            f"{name}: median_step_ms={latency:.3f} "
            f"peak_allocated={allocated} peak_reserved={reserved}"
        )
        if name == "sdpa":
            print(f"  profiler_events={_profile_backend(fn, q, k, v)}")


if __name__ == "__main__":
    main()
