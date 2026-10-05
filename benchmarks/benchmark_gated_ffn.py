"""Benchmark the reusable gated FFN implementation.

The benchmark compares the regular PyTorch GatedFFN with a Triton path that
fuses the first Linear, gate split, and ``SiLU(gate) * value`` operation.  The
output Linear remains a cuBLAS/PyTorch Linear.  It measures the complete
GatedFFN, including forward and backward, so the result reflects the actual
training use case.

Examples::

    python3 -m benchmarks.benchmark_gated_ffn \
        --device cuda --dtype bf16 --dim 1024 --hidden-dim 3072
    python3 -m benchmarks.benchmark_gated_ffn \
        --device cuda --dtype bf16 --batch-size 2 --tokens 4096 \
        --warmup 3 --repeats 20
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from core.layers import GatedFFN


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    return torch.device(value)


def resolve_dtype(value: str | None, device: torch.device) -> torch.dtype:
    value = value or ("bf16" if device.type == "cuda" else "fp32")
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


def build_model(
    backend: str,
    dim: int,
    hidden_dim: int,
    output_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> GatedFFN:
    model = GatedFFN(
        dim, hidden_dim, output_dim, bias=False, backend=backend,
    ).to(device=device, dtype=dtype)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model


def clear_grads(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def run_backend(
    backend: str,
    model: GatedFFN,
    base_input: torch.Tensor,
    warmup: int,
    repeats: int,
    forward_only: bool,
    device: torch.device,
) -> dict[str, object]:
    def invoke() -> tuple[torch.Tensor, torch.Tensor]:
        x = base_input.detach().clone().requires_grad_(not forward_only)
        return model(x), x

    for _ in range(warmup):
        clear_grads(model)
        output, x = invoke()
        if not forward_only:
            output.float().square().mean().backward()
        del output, x
    synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    forward_times = []
    backward_times = []
    for _ in range(repeats):
        clear_grads(model)
        synchronize(device)
        started = time.perf_counter()
        output, x = invoke()
        synchronize(device)
        forward_times.append((time.perf_counter() - started) * 1000.0)

        if not forward_only:
            loss = output.float().square().mean()
            synchronize(device)
            started = time.perf_counter()
            loss.backward()
            synchronize(device)
            backward_times.append((time.perf_counter() - started) * 1000.0)
            del loss
        del output, x

    result: dict[str, object] = {
        "backend": backend,
        "forward_median_ms": statistics.median(forward_times),
        "forward_p90_ms": percentile(forward_times, 0.90),
        "backward_median_ms": (
            statistics.median(backward_times) if backward_times else None
        ),
        "backward_p90_ms": (
            percentile(backward_times, 0.90) if backward_times else None
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


def output_error(
    torch_model: GatedFFN,
    model: GatedFFN,
    base_input: torch.Tensor,
) -> float:
    with torch.no_grad():
        expected = torch_model(base_input)
        actual = model(base_input)
    return float((actual - expected).abs().max().item())


def print_result(result: dict[str, object], max_error: float | str) -> None:
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
    print(
        f"kernel=gated_ffn backend={result['backend']:7s} "
        f"forward median={result['forward_median_ms']:.3f}ms "
        f"p90={result['forward_p90_ms']:.3f}ms {backward}"
        f"{memory} max_error={max_error}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends", default="torch,triton",
        help="Comma-separated backends: torch, triton, or auto.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--output-dim", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    backends = tuple(part.strip() for part in args.backends.split(",") if part.strip())
    if not backends or any(backend not in {"torch", "triton", "auto"} for backend in backends):
        parser.error("--backends must contain only torch, triton, or auto")
    for name in ("batch_size", "tokens", "dim"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    hidden_dim = args.hidden_dim or args.dim * 3
    output_dim = args.output_dim or args.dim
    if hidden_dim <= 0 or output_dim <= 0:
        parser.error("--hidden-dim and --output-dim must be positive")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats must be positive")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    base_input = torch.randn(
        args.batch_size, args.tokens, args.dim,
        device=device, dtype=dtype,
    )
    template = build_model(
        "torch", args.dim, hidden_dim, output_dim, device, dtype,
    )
    state_dict = {
        name: value.detach().clone()
        for name, value in template.state_dict().items()
    }
    models = {}
    for backend in backends:
        models[backend] = build_model(
            backend, args.dim, hidden_dim, output_dim, device, dtype, state_dict,
        )

    print(
        f"benchmark: kernel=gated_ffn backends={','.join(backends)} "
        f"device={device} dtype={dtype} batch={args.batch_size} "
        f"tokens={args.tokens} dim={args.dim} hidden_dim={hidden_dim} "
        f"output_dim={output_dim} warmup={args.warmup} repeats={args.repeats} "
        f"forward_only={args.forward_only}",
        flush=True,
    )

    results = []
    for backend, model in models.items():
        try:
            result = run_backend(
                backend, model, base_input, args.warmup, args.repeats,
                args.forward_only, device,
            )
        except (RuntimeError, ValueError) as error:
            print(f"kernel=gated_ffn backend={backend:7s} skipped: {error}")
            continue
        error: float | str = "not-checked"
        if args.verify:
            try:
                error = output_error(template, model, base_input)
            except (RuntimeError, ValueError) as verification_error:
                error = f"unavailable: {verification_error}"
        print_result(result, error)
        results.append(result | {"max_error": error})
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "device": str(device),
            "dtype": str(dtype),
            "config": vars(args) | {
                "hidden_dim": hidden_dim,
                "output_dim": output_dim,
                "backends": backends,
                "json": str(args.json),
            },
            "results": results,
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"saved benchmark JSON: {args.json}")


if __name__ == "__main__":
    main()
