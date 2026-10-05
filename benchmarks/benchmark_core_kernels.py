"""Benchmark reusable core kernels.

Examples::

    python3 -m benchmarks.benchmark_core_kernels --kernels both --backends naive,torch
    python3 -m benchmarks.benchmark_core_kernels --kernels both \
        --backends naive,triton --device cuda --dtype bf16

The benchmark reports separate forward/backward timings after warmup.  CUDA
measurements synchronize around each timed span and also report peak allocator
usage.  Triton compilation is included only in warmup, never in the reported
repetitions.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from core.kernels import apply_rope, rms_norm


KERNELS = ("rms_norm", "rope")
BACKENDS = ("naive", "torch", "triton", "auto")


def _parse_csv(value: str, choices: tuple[str, ...], label: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values or any(value not in choices for value in values):
        raise argparse.ArgumentTypeError(
            f"{label} must be a comma-separated subset of: {','.join(choices)}"
        )
    return values


def parse_kernels(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "both":
        return KERNELS
    return _parse_csv(value, KERNELS, "kernels")


def parse_backends(value: str) -> tuple[str, ...]:
    return _parse_csv(value, BACKENDS, "backends")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "fp32":
        return torch.float32
    if value == "fp16":
        if device.type == "cpu":
            raise ValueError("fp16 benchmark requires CUDA")
        return torch.float16
    if value == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 is not supported by the CUDA device")
        return torch.bfloat16
    raise ValueError(f"unknown dtype: {value}")


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    return torch.device(value)


def _make_inputs(args, device: torch.device, dtype: torch.dtype, kernel: str):
    if kernel == "rms_norm":
        x = torch.randn(
            args.batch_size,
            args.tokens,
            args.dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        weight = torch.randn(
            args.dim, device=device, dtype=dtype, requires_grad=True,
        )
        return x, weight

    x = torch.randn(
        args.batch_size,
        args.heads,
        args.tokens,
        args.head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    angles = torch.randn(
        args.tokens,
        args.head_dim // 2,
        device=device,
        dtype=dtype,
    )
    return x, angles.cos(), angles.sin()


def _run(kernel: str, backend: str, inputs, args, device: torch.device):
    if kernel == "rms_norm":
        x, weight = inputs

        def invoke():
            return rms_norm(x, weight, eps=args.eps, backend=backend)

        parameters = (x, weight)
    else:
        x, cos, sin = inputs

        def invoke():
            return apply_rope(x, cos, sin, backend=backend)

        parameters = (x,)

    def clear_grads():
        for parameter in parameters:
            parameter.grad = None

    for _ in range(args.warmup):
        clear_grads()
        output = invoke()
        if not args.forward_only:
            output.float().square().mean().backward()
        del output
    _synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    forward_times = []
    backward_times = []
    for _ in range(args.repeats):
        clear_grads()
        _synchronize(device)
        started = time.perf_counter()
        output = invoke()
        _synchronize(device)
        forward_times.append((time.perf_counter() - started) * 1000.0)

        if not args.forward_only:
            loss = output.float().square().mean()
            _synchronize(device)
            started = time.perf_counter()
            loss.backward()
            _synchronize(device)
            backward_times.append((time.perf_counter() - started) * 1000.0)
            del loss
        del output

    result = {
        "kernel": kernel,
        "backend": backend,
        "forward_median_ms": statistics.median(forward_times),
        "forward_p90_ms": percentile(forward_times, 0.90),
        "forward_mean_ms": statistics.mean(forward_times),
        "backward_median_ms": (
            statistics.median(backward_times) if backward_times else None
        ),
        "backward_p90_ms": (
            percentile(backward_times, 0.90) if backward_times else None
        ),
        "backward_mean_ms": (
            statistics.mean(backward_times) if backward_times else None
        ),
        "peak_allocated_mib": None,
        "peak_reserved_mib": None,
    }
    if device.type == "cuda":
        result["peak_allocated_mib"] = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )
        result["peak_reserved_mib"] = (
            torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        )
    return result


def _verify(kernel: str, backends: tuple[str, ...], inputs, args, device, dtype):
    reference = _run_inference(kernel, "naive", inputs, args, device)
    errors = {}
    for backend in backends:
        if backend == "naive":
            errors[backend] = 0.0
            continue
        try:
            actual = _run_inference(kernel, backend, inputs, args, device)
        except (RuntimeError, ValueError) as error:
            errors[backend] = f"unavailable: {error}"
            continue
        errors[backend] = float((actual - reference).abs().max().item())
    return errors


def _run_inference(kernel: str, backend: str, inputs, args, device):
    with torch.no_grad():
        if kernel == "rms_norm":
            x, weight = inputs
            return rms_norm(x, weight, eps=args.eps, backend=backend)
        x, cos, sin = inputs
        return apply_rope(x, cos, sin, backend=backend)


def _print_result(result, errors=None):
    backward = "forward-only"
    if result["backward_median_ms"] is not None:
        backward = (
            f"backward median={result['backward_median_ms']:.3f}ms "
            f"p90={result['backward_p90_ms']:.3f}ms"
        )
    memory = ""
    if result["peak_allocated_mib"] is not None:
        memory = (
            f" peak_allocated={result['peak_allocated_mib']:.1f}MiB"
            f" peak_reserved={result['peak_reserved_mib']:.1f}MiB"
        )
    error = ""
    if errors is not None:
        error = f" max_error={errors.get(result['backend'])}"
    print(
        f"kernel={result['kernel']:8s} backend={result['backend']:7s} "
        f"forward median={result['forward_median_ms']:.3f}ms "
        f"p90={result['forward_p90_ms']:.3f}ms {backward}"
        f"{memory}{error}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", type=parse_kernels, default=KERNELS)
    parser.add_argument(
        "--backends", type=parse_backends,
        default=("naive", "torch", "triton"),
        help="Comma-separated backends. Unsupported backends are reported and skipped.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats must be positive")
    for name in ("batch_size", "heads", "tokens", "dim", "head_dim"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.dim <= 0 or args.head_dim <= 0:
        parser.error("dimensions must be positive")
    if args.head_dim % 2:
        parser.error("--head-dim must be even for RoPE")

    device = _device(args.device)
    dtype = _dtype(args.dtype or ("bf16" if device.type == "cuda" else "fp32"), device)
    print(
        f"benchmark: kernels={','.join(args.kernels)} "
        f"backends={','.join(args.backends)} device={device} dtype={dtype} "
        f"batch={args.batch_size} tokens={args.tokens} dim={args.dim} "
        f"heads={args.heads} head_dim={args.head_dim} "
        f"warmup={args.warmup} repeats={args.repeats} "
        f"forward_only={args.forward_only}",
        flush=True,
    )

    all_results = []
    for kernel in args.kernels:
        if kernel == "rms_norm" and args.dim <= 0:
            continue
        inputs = _make_inputs(args, device, dtype, kernel)
        errors = None
        if args.verify:
            errors = _verify(kernel, args.backends, inputs, args, device, dtype)
        for backend in args.backends:
            try:
                result = _run(kernel, backend, inputs, args, device)
            except (RuntimeError, ValueError) as error:
                print(
                    f"kernel={kernel:8s} backend={backend:7s} skipped: {error}",
                    flush=True,
                )
                continue
            _print_result(result, errors)
            all_results.append(result | {"max_error": None if errors is None else errors.get(backend)})
        del inputs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "device": str(device),
            "dtype": str(dtype),
            "config": vars(args) | {"json": str(args.json)},
            "results": all_results,
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"saved benchmark JSON: {args.json}")


if __name__ == "__main__":
    main()
