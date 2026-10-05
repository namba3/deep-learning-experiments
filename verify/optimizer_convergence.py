"""Deterministic convergence probe for the local optimizers.

This is a small model-shaped regression task, not a substitute for an
image-generation run.  It keeps the initial model, inputs, and targets fixed
across optimizer choices while measuring actual forward/backward and update
steps.  The probe is intended to catch gross convergence regressions before a
long external-dataset experiment.

Example::

    python3 -m verify.optimizer_convergence --device cuda --dtype bf16
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from time import perf_counter

import torch
from torch import nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from optimizers import APOLLO, APOLLOCAME, APOLLOMini, CAME  # noqa: E402


OPTIMIZERS = ("CAME", "APOLLO", "APOLLO-CAME", "APOLLOMini")


class ProbeModel(nn.Module):
    """Small MLP containing matrix weights and one-dimensional parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(32, 64)
        self.norm = nn.LayerNorm(64)
        self.output = nn.Linear(64, 16)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.norm(self.input(inputs))))


def _parse_csv(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values or any(value not in OPTIMIZERS for value in values):
        raise argparse.ArgumentTypeError(
            "optimizers must be a comma-separated subset of: "
            + ",".join(OPTIMIZERS)
        )
    return values


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("values must be comma-separated integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def _parse_positive_floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("values must be comma-separated numbers") from error
    if not values or any(item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive numbers")
    return values


def _parse_norm_growth_rates(value: str) -> tuple[float, ...]:
    values = _parse_positive_floats(value)
    if any(item <= 1.0 for item in values):
        raise argparse.ArgumentTypeError(
            "norm growth rates must be greater than 1.0"
        )
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--optimizers", type=_parse_csv, default=OPTIMIZERS,
        help="Comma-separated optimizer names. Default: all.",
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument(
        "--ranks", type=_parse_positive_ints, default=None,
        help="Optional comma-separated APOLLO ranks to sweep.",
    )
    parser.add_argument(
        "--matrix-fallback", choices=("apollo", "came", "auto"), default="auto",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--learning-rates", type=_parse_positive_floats, default=None,
        help="Optional comma-separated learning rates to sweep.",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--scales", type=_parse_positive_floats, default=None,
        help="Optional comma-separated APOLLO scales to sweep.",
    )
    parser.add_argument(
        "--disable-norm-growth-limiter", action="store_true",
        help="Disable APOLLO's norm-growth limiter for this probe.",
    )
    parser.add_argument(
        "--norm-growth-rate", type=float, default=1.01,
        help="APOLLO norm-growth limit multiplier. Default: 1.01.",
    )
    parser.add_argument(
        "--norm-growth-rates", type=_parse_norm_growth_rates, default=None,
        help="Optional comma-separated norm-growth rates to sweep.",
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


def state_metrics(optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    bytes_total = 0
    elements_total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                elements_total += value.numel()
                bytes_total += value.numel() * value.element_size()
    return bytes_total, elements_total


def build_optimizer(name: str, model: nn.Module, args):
    parameters = list(model.parameters())
    fallback = {
        "1d": "came",
        "small_matrix": args.matrix_fallback,
        "state_margin": 1.0,
        "min_savings_bytes": 0,
    }
    common = {"lr": args.learning_rate, "weight_decay": 0.0}
    if name == "CAME":
        return CAME(parameters, **common, backend="torch")
    if name == "APOLLO":
        return APOLLO(
            parameters, **common, rank=args.rank, scale=args.scale,
            norm_growth_limiter=args.norm_growth_limiter,
            norm_growth_rate=args.norm_growth_rate, fallback=fallback,
        )
    if name == "APOLLO-CAME":
        return APOLLOCAME(
            parameters, **common, rank=args.rank, scale=args.scale,
            norm_growth_limiter=args.norm_growth_limiter,
            norm_growth_rate=args.norm_growth_rate,
            fallback=fallback, came_backend="torch",
        )
    if name == "APOLLOMini":
        return APOLLOMini(
            parameters, **common, scale=args.scale,
            norm_growth_limiter=args.norm_growth_limiter,
            norm_growth_rate=args.norm_growth_rate, fallback=fallback,
        )
    raise ValueError(f"unknown optimizer: {name}")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_probe_data(args, device: torch.device, dtype: torch.dtype):
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    inputs = torch.randn(args.batch_size, 32, generator=generator)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 2)
        teacher = nn.Sequential(nn.Linear(32, 64), nn.Tanh(), nn.Linear(64, 16))
    with torch.no_grad():
        targets = teacher(inputs)
    return (
        inputs.to(device=device, dtype=dtype),
        targets.to(device=device, dtype=dtype),
    )


def run_case(
    name: str,
    args,
    device: torch.device,
    dtype: torch.dtype,
    initial_state: dict[str, torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, object]:
    model = ProbeModel().to(device=device, dtype=dtype)
    model.load_state_dict(
        {key: value.to(device=device, dtype=dtype) for key, value in initial_state.items()}
    )
    optimizer = build_optimizer(name, model, args)

    with torch.no_grad():
        initial_loss = (
            model(inputs).float() - targets.float()
        ).square().mean()

    for _ in range(args.warmup):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs).float() - targets.float()).square().mean()
        loss.backward()
        optimizer.step()
    _synchronize(device)

    baseline_allocated = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)

    losses: list[float] = []
    total_host = 0.0
    optimizer_host = 0.0
    cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        started = perf_counter()
        loss = (model(inputs).float() - targets.float()).square().mean()
        loss.backward()
        optimizer_started = perf_counter()
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        optimizer.step()
        if device.type == "cuda":
            end_event.record()
            cuda_events.append((start_event, end_event))
        optimizer_host += perf_counter() - optimizer_started
        total_host += perf_counter() - started
        # Evaluate after the update, outside the timed forward/backward/step
        # interval, so the loss trajectory is comparable across optimizers.
        with torch.no_grad():
            observed_loss = (
                model(inputs).float() - targets.float()
            ).square().mean()
        losses.append(float(observed_loss))
    _synchronize(device)

    state_bytes, state_elements = state_metrics(optimizer)
    result: dict[str, object] = {
        "status": "passed"
        if torch.isfinite(torch.tensor([float(initial_loss), *losses])).all()
        else "failed",
        "optimizer": name,
        "rank": args.rank if name in {"APOLLO", "APOLLO-CAME"} else (1 if name == "APOLLOMini" else None),
        "learning_rate": args.learning_rate,
        "scale": args.scale if name != "CAME" else None,
        "norm_growth_limiter": (
            args.norm_growth_limiter if name != "CAME" else None
        ),
        "norm_growth_rate": (
            args.norm_growth_rate if name != "CAME" else None
        ),
        "backend_counts": {
            backend: sum(
                state.get("backend") == backend
                for state in optimizer.state.values()
            )
            for backend in ("apollo", "came", "sgd")
        },
        "parameter_numel": sum(parameter.numel() for parameter in model.parameters()),
        "persistent_state_bytes": state_bytes,
        "persistent_state_elements": state_elements,
        "initial_loss": float(initial_loss),
        "final_loss": losses[-1] if losses else None,
        "best_loss": min([float(initial_loss), *losses]),
        "loss_history": losses,
        "host_seconds_per_step": total_host / args.steps,
        "host_seconds_per_optimizer_step": optimizer_host / args.steps,
    }
    if device.type == "cuda":
        result.update(
            {
                "cuda_seconds_per_optimizer_step": sum(
                    start.elapsed_time(end) for start, end in cuda_events
                ) / (1000.0 * len(cuda_events)),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "peak_delta_allocated_bytes": (
                    torch.cuda.max_memory_allocated(device) - baseline_allocated
                ),
            }
        )
    return result


def run(args) -> dict[str, object]:
    # Keep ``run`` usable by small external smoke tests that construct an old
    # argparse.Namespace manually instead of going through parse_args().
    args.ranks = getattr(args, "ranks", None)
    args.learning_rates = getattr(args, "learning_rates", None)
    args.scale = getattr(args, "scale", 1.0)
    args.scales = getattr(args, "scales", None)
    args.norm_growth_rate = getattr(args, "norm_growth_rate", 1.01)
    args.norm_growth_rates = getattr(args, "norm_growth_rates", None)
    args.disable_norm_growth_limiter = getattr(
        args, "disable_norm_growth_limiter", False
    )
    args.norm_growth_limiter = not args.disable_norm_growth_limiter
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.rank <= 0:
        raise ValueError("--rank must be positive")
    if args.learning_rate < 0.0:
        raise ValueError("--learning-rate must be non-negative")
    if args.scale <= 0.0:
        raise ValueError("--scale must be positive")
    if args.norm_growth_rate <= 1.0:
        raise ValueError("--norm-growth-rate must be greater than 1.0")
    rank_values = args.ranks or (args.rank,)
    learning_rate_values = args.learning_rates or (args.learning_rate,)
    scale_values = args.scales or (args.scale,)
    if any(rank <= 0 for rank in rank_values):
        raise ValueError("all ranks must be positive")
    if any(rate <= 0.0 for rate in learning_rate_values):
        raise ValueError("all learning rates must be positive")
    if any(scale <= 0.0 for scale in scale_values):
        raise ValueError("all scales must be positive")
    norm_growth_rate_values = args.norm_growth_rates or (args.norm_growth_rate,)
    if any(rate <= 1.0 for rate in norm_growth_rate_values):
        raise ValueError("all norm growth rates must be greater than 1.0")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    inputs, targets = _make_probe_data(args, device, dtype)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    initial_model = ProbeModel()
    with torch.no_grad():
        for parameter in initial_model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator))
    initial_state = {
        key: value.detach().clone() for key, value in initial_model.state_dict().items()
    }

    cases: dict[str, object] = {}
    for name in args.optimizers:
        if name == "CAME":
            configurations = [
                {"rank": args.rank, "learning_rate": rate, "scale": args.scale}
                for rate in learning_rate_values
            ]
        elif name == "APOLLOMini":
            configurations = [
                {
                    "rank": 1,
                    "learning_rate": rate,
                    "scale": scale,
                    "norm_growth_rate": growth_rate,
                }
                for rate in learning_rate_values
                for scale in scale_values
                for growth_rate in norm_growth_rate_values
            ]
        else:
            configurations = [
                {
                    "rank": rank,
                    "learning_rate": rate,
                    "scale": scale,
                    "norm_growth_rate": growth_rate,
                }
                for rank in rank_values
                for rate in learning_rate_values
                for scale in scale_values
                for growth_rate in norm_growth_rate_values
            ]

        for configuration in configurations:
            case_args = argparse.Namespace(**vars(args))
            case_args.rank = configuration["rank"]
            case_args.learning_rate = configuration["learning_rate"]
            case_args.scale = configuration["scale"]
            if name != "CAME":
                case_args.norm_growth_rate = configuration["norm_growth_rate"]
            is_sweep = any(
                value is not None
                for value in (
                    args.ranks, args.learning_rates, args.scales,
                    args.norm_growth_rates,
                )
            )
            if not is_sweep:
                case_name = name
            elif name == "CAME":
                case_name = f"CAME@lr={configuration['learning_rate']:g}"
            elif name == "APOLLOMini":
                case_name = (
                    f"APOLLOMini@lr={configuration['learning_rate']:g},"
                    f"scale={configuration['scale']:g}"
                )
            else:
                case_name = (
                    f"{name}@rank={configuration['rank']},"
                    f"lr={configuration['learning_rate']:g},"
                    f"scale={configuration['scale']:g}"
                )
            if name != "CAME" and args.norm_growth_rates is not None:
                case_name += f",growth={configuration['norm_growth_rate']:g}"
            try:
                cases[case_name] = run_case(
                    name, case_args, device, dtype, initial_state, inputs, targets,
                )
            except Exception as error:
                cases[case_name] = {
                    "status": "failed",
                    "optimizer": name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return {
        "status": "passed" if all(case["status"] == "passed" for case in cases.values()) else "failed",
        "device": str(device),
        "dtype": args.dtype,
        "warmup": args.warmup,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "rank": args.rank,
        "ranks": list(rank_values),
        "matrix_fallback": args.matrix_fallback,
        "learning_rate": args.learning_rate,
        "learning_rates": list(learning_rate_values),
        "scale": args.scale,
        "scales": list(scale_values),
        "norm_growth_limiter": args.norm_growth_limiter,
        "norm_growth_rate": args.norm_growth_rate,
        "norm_growth_rates": list(norm_growth_rate_values),
        "cases": cases,
    }


def main(argv=None) -> int:
    try:
        result = run(parse_args(argv))
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
