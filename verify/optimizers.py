"""Runtime benchmark for the local optimizer implementations.

The benchmark keeps gradient generation outside the timed region and reports
persistent state bytes separately from allocator peaks.  It is intentionally
independent of a model or dataset so APOLLO/CAME choices can be compared on
the same parameter shapes before running a long training job.

Examples::

    python3 -m verify.optimizers --device auto --dtype fp32
    python3 -m verify.optimizers --device cuda --dtype bf16 --steps 20
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from time import perf_counter

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from optimizers import APOLLO, APOLLOCAME, APOLLOMini, CAME  # noqa: E402
from optimizers.projection_refresh import ProjectionRefreshPolicy  # noqa: E402


OPTIMIZERS = ("CAME", "APOLLO", "APOLLO-CAME", "APOLLOMini")
DEFAULT_SHAPES = "4x3,16x16,32x32,64x32x3x3,4096"


def _parse_csv(value: str, choices: tuple[str, ...], label: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values or any(item not in choices for item in values):
        raise argparse.ArgumentTypeError(
            f"{label} must be a comma-separated subset of: {','.join(choices)}"
        )
    return values


def parse_shape(value: str) -> tuple[int, ...]:
    parts = tuple(part.strip() for part in value.lower().split("x"))
    if not parts or any(not part.isdigit() or int(part) <= 0 for part in parts):
        raise argparse.ArgumentTypeError(
            "shape must contain positive dimensions separated by 'x'"
        )
    return tuple(int(part) for part in parts)


def parse_shapes(value: str) -> tuple[tuple[int, ...], ...]:
    shapes = tuple(parse_shape(part) for part in value.split(",") if part.strip())
    if not shapes:
        raise argparse.ArgumentTypeError("shapes must not be empty")
    return shapes


def parse_ranks(value: str) -> tuple[int, ...]:
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts or any(not part.isdigit() or int(part) <= 0 for part in parts):
        raise argparse.ArgumentTypeError(
            "ranks must be a comma-separated list of positive integers"
        )
    return tuple(int(part) for part in parts)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="Execution device. Default: auto.",
    )
    parser.add_argument(
        "--dtype", choices=("fp32", "bf16"), default="fp32",
        help="Parameter and gradient dtype. Default: fp32.",
    )
    parser.add_argument(
        "--warmup", type=int, default=2,
        help="Untimed state-initialization steps per case. Default: 2.",
    )
    parser.add_argument(
        "--steps", type=int, default=10,
        help="Timed optimizer steps per case. Default: 10.",
    )
    parser.add_argument(
        "--update-proj-gap", type=int, default=200,
        help=(
            "APOLLO projection refresh interval. Use a small value to "
            "measure refresh steps. Default: 200."
        ),
    )
    parser.add_argument(
        "--projection-refresh-mode", choices=("none", "hard", "smooth"),
        default="hard",
        help="APOLLO projection refresh mode. Default: hard.",
    )
    parser.add_argument(
        "--projection-refresh-window", type=int, default=200,
        help="Smooth refresh window in steps. Default: 200.",
    )
    parser.add_argument(
        "--projection-refresh-mix",
        choices=("linear", "smoothstep", "stochastic", "ema"),
        default="smoothstep",
        help="Smooth refresh interpolation. Default: smoothstep.",
    )
    parser.add_argument(
        "--projection-refresh-state", choices=("reset", "transport"),
        default="reset",
        help="State handling at refresh. Default: reset.",
    )
    parser.add_argument(
        "--orthogonal-refresh-rate", type=float, default=0.0,
        help="Per-step APOLLO projection rotation rate. Default: 0.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for common initial parameters and gradients. Default: 0.",
    )
    parser.add_argument(
        "--shapes", type=parse_shapes, default=parse_shapes(DEFAULT_SHAPES),
        help="Comma-separated shapes, for example 4x3,32x32,4096.",
    )
    parser.add_argument(
        "--optimizers", type=lambda value: _parse_csv(value, OPTIMIZERS, "optimizers"),
        default=OPTIMIZERS,
        help="Comma-separated optimizer names. Default: all.",
    )
    parser.add_argument(
        "--ranks", type=parse_ranks, default=(1, 8),
        help="Ranks for APOLLO variants. Default: 1,8.",
    )
    parser.add_argument(
        "--matrix-fallback", choices=("apollo", "came", "auto"),
        default="auto",
        help="Matrix backend policy for APOLLO variants. Default: auto.",
    )
    parser.add_argument(
        "--fallback-state-margin", type=float, default=1.0,
        help="CAME/APOLLO state-size ratio for auto fallback. Default: 1.0.",
    )
    parser.add_argument(
        "--fallback-min-savings-bytes", type=int, default=0,
        help="Minimum state savings required by auto fallback. Default: 0.",
    )
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(requested)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if device.type == "cuda" and not torch.cuda.is_bf16_supported(device):
        raise ValueError("BF16 is not supported by the selected CUDA device")
    return torch.bfloat16


def state_metrics(
    optimizer: torch.optim.Optimizer,
    parameter: torch.Tensor,
) -> tuple[int, int]:
    """Return persistent state bytes and tensor elements separately.

    The state may mix BF16/FP32 tensors, so element count cannot be derived
    from bytes using one fixed element size.
    """
    persistent_bytes = 0
    persistent_elements = 0
    for value in optimizer.state[parameter].values():
        if not isinstance(value, torch.Tensor):
            continue
        persistent_elements += value.numel()
        persistent_bytes += value.numel() * value.element_size()
    return persistent_bytes, persistent_elements


def build_optimizer(
    name: str,
    parameter: torch.Tensor,
    rank: int,
    fallback: dict[str, object],
    update_proj_gap: int,
    *,
    projection_refresh_mode: str = "hard",
    projection_refresh_window: int = 200,
    projection_refresh_mix: str = "smoothstep",
    projection_refresh_state: str = "reset",
    orthogonal_refresh_rate: float = 0.0,
    orthogonal_refresh_seed: int = 0,
):
    refresh_policy = ProjectionRefreshPolicy(
        mode=projection_refresh_mode,
        interval=update_proj_gap,
        window=(projection_refresh_window
                if projection_refresh_mode == "smooth" else 0),
        mix=projection_refresh_mix,
    ).as_dict()
    if name == "CAME":
        return CAME([parameter], lr=0.01, weight_decay=0.0, backend="torch")
    if name == "APOLLO":
        return APOLLO(
            [parameter], lr=0.01, weight_decay=0.0, rank=rank,
            fallback=fallback, update_proj_gap=update_proj_gap,
            projection_refresh=refresh_policy,
            projection_refresh_state=projection_refresh_state,
            orthogonal_refresh={
                "rate": orthogonal_refresh_rate,
                "seed": orthogonal_refresh_seed,
            },
        )
    if name == "APOLLO-CAME":
        return APOLLOCAME(
            [parameter], lr=0.01, weight_decay=0.0, rank=rank,
            fallback=fallback, update_proj_gap=update_proj_gap,
            projection_refresh=refresh_policy,
            projection_refresh_state=projection_refresh_state,
            orthogonal_refresh={
                "rate": orthogonal_refresh_rate,
                "seed": orthogonal_refresh_seed,
            },
            came_backend="torch",
        )
    if name == "APOLLOMini":
        return APOLLOMini(
            [parameter], lr=0.01, weight_decay=0.0,
            fallback=fallback, update_proj_gap=update_proj_gap,
            projection_refresh=refresh_policy,
            projection_refresh_state=projection_refresh_state,
            orthogonal_refresh={
                "rate": orthogonal_refresh_rate,
                "seed": orthogonal_refresh_seed,
            },
        )
    raise ValueError(f"unknown optimizer: {name}")


def _configs(names: tuple[str, ...], ranks: tuple[int, ...]):
    for name in names:
        if name in {"APOLLO", "APOLLO-CAME"}:
            for rank in ranks:
                yield f"{name}@rank={rank}", name, rank
        elif name == "APOLLOMini":
            yield f"{name}@rank=1", name, 1
        else:
            yield name, name, 0


def _common_tensors(
    shape: tuple[int, ...],
    steps: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randn(shape, generator=generator, dtype=torch.float32)
    gradients = [
        torch.randn(shape, generator=generator, dtype=torch.float32)
        for _ in range(steps)
    ]
    return initial.to(device=device, dtype=dtype), [
        gradient.to(device=device, dtype=dtype) for gradient in gradients
    ]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_case(
    name: str,
    optimizer_name: str,
    rank: int,
    shape: tuple[int, ...],
    args,
    device: torch.device,
    dtype: torch.dtype,
    case_seed: int,
    fallback: dict[str, object],
) -> dict[str, object]:
    initial, gradients = _common_tensors(
        shape, args.warmup + args.steps, dtype, device, case_seed,
    )
    update_proj_gap = int(getattr(args, "update_proj_gap", 200))
    projection_refresh_mode = getattr(
        args, "projection_refresh_mode", "hard"
    )
    projection_refresh_window = int(
        getattr(args, "projection_refresh_window", 200)
    )
    projection_refresh_mix = getattr(
        args, "projection_refresh_mix", "smoothstep"
    )
    projection_refresh_state = getattr(
        args, "projection_refresh_state", "reset"
    )
    orthogonal_refresh_rate = float(
        getattr(args, "orthogonal_refresh_rate", 0.0)
    )
    orthogonal_refresh_seed = int(getattr(args, "seed", 0))
    parameter = torch.nn.Parameter(initial)
    optimizer = build_optimizer(
        optimizer_name, parameter, rank, fallback, update_proj_gap,
        projection_refresh_mode=projection_refresh_mode,
        projection_refresh_window=projection_refresh_window,
        projection_refresh_mix=projection_refresh_mix,
        projection_refresh_state=projection_refresh_state,
        orthogonal_refresh_rate=orthogonal_refresh_rate,
        orthogonal_refresh_seed=orthogonal_refresh_seed,
    )

    for gradient in gradients[:args.warmup]:
        parameter.grad = gradient
        optimizer.step()
    _synchronize(device)
    peak_persistent_bytes, peak_persistent_elements = state_metrics(
        optimizer, parameter
    )

    baseline_allocated = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)

    host_started = perf_counter()
    events = []
    step_host_seconds = []
    refresh_flags = []
    refresh_active_flags = []
    for gradient in gradients[args.warmup:]:
        parameter.grad = gradient
        was_refresh_active = bool(
            optimizer.state[parameter].get("refresh_active", False)
        )
        previous_projection_seed = optimizer.state[parameter].get(
            "projection_seed"
        )
        step_started = perf_counter()
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            events.append((start, end))
        optimizer.step()
        step_host_seconds.append(perf_counter() - step_started)
        current_state_bytes, current_state_elements = state_metrics(
            optimizer, parameter
        )
        peak_persistent_bytes = max(
            peak_persistent_bytes, current_state_bytes
        )
        peak_persistent_elements = max(
            peak_persistent_elements, current_state_elements
        )
        current_projection_seed = optimizer.state[parameter].get(
            "projection_seed"
        )
        refresh_flags.append(
            previous_projection_seed is not None
            and current_projection_seed != previous_projection_seed
        )
        refresh_active_flags.append(
            was_refresh_active or refresh_flags[-1]
        )
        if device.type == "cuda":
            end.record()
    host_seconds = perf_counter() - host_started
    _synchronize(device)
    persistent_bytes, persistent_elements = state_metrics(optimizer, parameter)
    estimate_state_bytes = getattr(
        optimizer, "estimate_parameter_state_bytes", None
    )
    estimated_persistent_bytes = None
    state_estimate_matches_actual = None
    if callable(estimate_state_bytes):
        estimated_persistent_bytes = int(estimate_state_bytes(parameter))
        state_estimate_matches_actual = (
            estimated_persistent_bytes == persistent_bytes
        )
    regular_host_seconds = [
        elapsed
        for elapsed, refreshed in zip(step_host_seconds, refresh_flags)
        if not refreshed
    ]
    refresh_host_seconds = [
        elapsed
        for elapsed, refreshed in zip(step_host_seconds, refresh_flags)
        if refreshed
    ]
    active_refresh_host_seconds = [
        elapsed
        for elapsed, active in zip(step_host_seconds, refresh_active_flags)
        if active
    ]

    def average(values):
        return sum(values) / len(values) if values else None

    result: dict[str, object] = {
        "status": "passed"
        if (
            torch.isfinite(parameter).all()
            and state_estimate_matches_actual is not False
        )
        else "failed",
        "optimizer": name,
        "shape": list(shape),
        "numel": parameter.numel(),
        "rank": rank if optimizer_name != "CAME" else None,
        "projection_refresh_mode": (
            projection_refresh_mode if optimizer_name != "CAME" else None
        ),
        "projection_refresh_window": (
            projection_refresh_window
            if optimizer_name != "CAME" and projection_refresh_mode == "smooth"
            else (0 if optimizer_name != "CAME" else None)
        ),
        "projection_refresh_mix": (
            projection_refresh_mix if optimizer_name != "CAME" else None
        ),
        "projection_refresh_state": (
            projection_refresh_state if optimizer_name != "CAME" else None
        ),
        "orthogonal_refresh_rate": (
            orthogonal_refresh_rate if optimizer_name != "CAME" else None
        ),
        "orthogonal_refresh_steps": int(
            optimizer.state[parameter].get("orthogonal_refresh_count", 0)
        ),
        "backend": optimizer.state[parameter].get("backend", "native"),
        "persistent_state_bytes": persistent_bytes,
        "persistent_state_elements": persistent_elements,
        "peak_persistent_state_bytes": peak_persistent_bytes,
        "peak_persistent_state_elements": peak_persistent_elements,
        "estimated_persistent_state_bytes": estimated_persistent_bytes,
        "state_estimate_matches_actual": state_estimate_matches_actual,
        "host_seconds_total": host_seconds,
        "host_seconds_per_step": host_seconds / args.steps,
        "projection_refresh_steps": sum(refresh_flags),
        "projection_refresh_active_steps": sum(refresh_active_flags),
        "host_seconds_per_regular_step": average(regular_host_seconds),
        "host_seconds_per_projection_refresh_step": average(
            refresh_host_seconds
        ),
        "host_seconds_per_projection_refresh_active_step": average(
            active_refresh_host_seconds
        ),
        "final_parameter_norm": float(parameter.detach().float().norm()),
    }
    if device.type == "cuda":
        gpu_milliseconds = [start.elapsed_time(end) for start, end in events]
        regular_gpu_seconds = [
            milliseconds / 1000.0
            for milliseconds, refreshed in zip(gpu_milliseconds, refresh_flags)
            if not refreshed
        ]
        refresh_gpu_seconds = [
            milliseconds / 1000.0
            for milliseconds, refreshed in zip(gpu_milliseconds, refresh_flags)
            if refreshed
        ]
        active_refresh_gpu_seconds = [
            milliseconds / 1000.0
            for milliseconds, active in zip(
                gpu_milliseconds, refresh_active_flags
            )
            if active
        ]
        result.update(
            {
                "cuda_seconds_per_step": sum(gpu_milliseconds)
                / (1000.0 * len(gpu_milliseconds)),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "peak_delta_allocated_bytes": (
                    torch.cuda.max_memory_allocated(device) - baseline_allocated
                ),
                "cuda_seconds_per_regular_step": average(regular_gpu_seconds),
                "cuda_seconds_per_projection_refresh_step": average(
                    refresh_gpu_seconds
                ),
                "cuda_seconds_per_projection_refresh_active_step": average(
                    active_refresh_gpu_seconds
                ),
            }
        )
    return result


def run(args) -> dict[str, object]:
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    update_proj_gap = int(getattr(args, "update_proj_gap", 200))
    if update_proj_gap <= 0:
        raise ValueError("--update-proj-gap must be positive")
    projection_refresh_mode = getattr(
        args, "projection_refresh_mode", "hard"
    )
    projection_refresh_window = int(
        getattr(args, "projection_refresh_window", 200)
    )
    projection_refresh_mix = getattr(
        args, "projection_refresh_mix", "smoothstep"
    )
    projection_refresh_state = getattr(
        args, "projection_refresh_state", "reset"
    )
    orthogonal_refresh_rate = float(
        getattr(args, "orthogonal_refresh_rate", 0.0)
    )
    if projection_refresh_window < 0:
        raise ValueError("--projection-refresh-window must be non-negative")
    if orthogonal_refresh_rate < 0.0:
        raise ValueError("--orthogonal-refresh-rate must be non-negative")
    # Validate once even when the selected optimizer is CAME, so a benchmark
    # invocation has one unambiguous policy contract in its top-level JSON.
    refresh_policy = ProjectionRefreshPolicy(
        mode=projection_refresh_mode,
        interval=update_proj_gap,
        window=(projection_refresh_window
                if projection_refresh_mode == "smooth" else 0),
        mix=projection_refresh_mix,
    )
    if projection_refresh_state not in {"reset", "transport"}:
        raise ValueError(
            "--projection-refresh-state must be 'reset' or 'transport'"
        )
    matrix_fallback = getattr(args, "matrix_fallback", "auto")
    fallback_state_margin = float(
        getattr(args, "fallback_state_margin", 1.0)
    )
    fallback_min_savings_bytes = int(
        getattr(args, "fallback_min_savings_bytes", 0)
    )
    if fallback_state_margin <= 0.0:
        raise ValueError("--fallback-state-margin must be positive")
    if fallback_min_savings_bytes < 0:
        raise ValueError("--fallback-min-savings-bytes must be non-negative")
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    fallback = {
        "1d": "came",
        "small_matrix": matrix_fallback,
        "state_margin": fallback_state_margin,
        "min_savings_bytes": fallback_min_savings_bytes,
    }
    cases: dict[str, object] = {}
    for shape_index, shape in enumerate(args.shapes):
        for name, optimizer_name, rank in _configs(args.optimizers, args.ranks):
            key = f"{name}:{'x'.join(str(dimension) for dimension in shape)}"
            try:
                cases[key] = run_case(
                    name, optimizer_name, rank, shape, args, device, dtype,
                    args.seed + shape_index * 1000, fallback,
                )
            except Exception as error:
                cases[key] = {
                    "status": "failed",
                    "optimizer": name,
                    "shape": list(shape),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    passed = all(case["status"] == "passed" for case in cases.values())
    return {
        "status": "passed" if passed else "failed",
        "device": str(device),
        "dtype": args.dtype,
        "warmup": args.warmup,
        "steps": args.steps,
        "update_proj_gap": update_proj_gap,
        "projection_refresh": refresh_policy.as_dict(),
        "projection_refresh_state": projection_refresh_state,
        "orthogonal_refresh_rate": orthogonal_refresh_rate,
        "fallback": fallback,
        "cases": cases,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except Exception as error:
        print(json.dumps({
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
