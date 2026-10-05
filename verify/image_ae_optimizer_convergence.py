"""Deterministic optimizer probe using the actual local ImageAE model.

The probe uses ImageAE forward/backward and reconstruction loss on fixed
synthetic images.  It is intentionally independent of dataset and VAE
loading, so optimizer candidates can be compared much more cheaply than a
full image_gen run while retaining convolutional autoencoder parameter
shapes.

Example::

    python3 -m verify.image_ae_optimizer_convergence \
        --device cuda --dtype bf16 --warmup 5 --steps 200
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

from image_ae.train import ImageAE  # noqa: E402
from optimizers import (  # noqa: E402
    APOLLO, APOLLOCAME, APOLLOCAMELRSF, APOLLOMini, CAME, CAMESF, CAMELRSF,
)


OPTIMIZERS = (
    "CAME", "CAME-SF", "CAME-LRSF", "APOLLO", "APOLLO-CAME",
    "APOLLO-CAME-LRSF", "APOLLOMini",
)


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


def _parse_nonnegative_floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "values must be comma-separated numbers"
        ) from error
    if not values or any(item < 0.0 for item in values):
        raise argparse.ArgumentTypeError(
            "values must be non-negative numbers"
        )
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
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
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--learning-rates", type=_parse_positive_floats, default=None,
        help="Optional comma-separated learning rates to sweep.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--weight-decays", type=_parse_nonnegative_floats, default=None,
        help="Optional comma-separated decay values to sweep.",
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
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--bottleneck-channels", type=int, default=64)
    parser.add_argument("--downsample-stages", type=int, default=2)
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


def build_model(args, device: torch.device, dtype: torch.dtype) -> ImageAE:
    return ImageAE(
        latent_channels=args.latent_channels,
        bottleneck_channels=args.bottleneck_channels,
        encoder_type="residual_conv_ffn",
        decoder_type="residual_conv_ffn",
        encoder_blocks=1,
        decoder_blocks=1,
        downsample_stages=args.downsample_stages,
        vae=False,
    ).to(device=device, dtype=dtype)


def _lrsf_refresh_policy(args) -> dict[str, object]:
    """Return the refresh contract shared by the LRSF verification probes."""
    return {
        "mode": getattr(args, "came_lrsf_refresh_mode", "none"),
        "interval": int(getattr(args, "came_lrsf_refresh_interval", 200)),
        "window": int(getattr(args, "came_lrsf_refresh_window", 200)),
        "mix": getattr(args, "came_lrsf_refresh_mix", "smoothstep"),
    }


def _lrsf_orthogonal_refresh_policy(args) -> dict[str, object]:
    """Return the per-step orthogonal rotation contract for LRSF probes."""
    policy: dict[str, object] = {
        "rate": float(getattr(args, "came_lrsf_orthogonal_refresh_rate", 0.0)),
        "seed": int(getattr(args, "seed", 0)),
    }
    direction = getattr(
        args, "came_lrsf_orthogonal_refresh_direction", "random",
    )
    if direction != "random":
        policy["direction"] = direction
    signal = getattr(
        args, "came_lrsf_orthogonal_refresh_signal", "gradient",
    )
    if signal != "gradient":
        policy["signal"] = signal
    return policy


def build_optimizer(name: str, model: nn.Module, args):
    parameters = list(model.parameters())
    fallback = {
        "1d": "came",
        "small_matrix": "auto",
        "state_margin": 1.0,
        "min_savings_bytes": 0,
    }
    common = {
        "lr": args.learning_rate,
        "weight_decay": getattr(args, "weight_decay", 0.0),
    }
    if name == "CAME":
        return CAME(parameters, **common, backend="torch")
    if name == "CAME-SF":
        return CAMESF(
            parameters, **common, sf_beta1=0.9, warmup_steps=0,
        )
    if name == "CAME-LRSF":
        return CAMELRSF(
            parameters, **common, rank=args.rank, sf_beta1=0.9,
            warmup_steps=0, seed=args.seed,
            projection_refresh=_lrsf_refresh_policy(args),
            orthogonal_refresh=_lrsf_orthogonal_refresh_policy(args),
        )
    if name == "APOLLO":
        return APOLLO(
            parameters, **common, rank=args.rank, scale=args.scale,
            norm_growth_limiter=not args.disable_norm_growth_limiter,
            norm_growth_rate=args.norm_growth_rate,
            fallback=fallback,
        )
    if name == "APOLLO-CAME":
        return APOLLOCAME(
            parameters, **common, rank=args.rank, scale=args.scale,
            norm_growth_limiter=not args.disable_norm_growth_limiter,
            norm_growth_rate=args.norm_growth_rate,
            fallback=fallback, came_backend="torch",
        )
    if name == "APOLLO-CAME-LRSF":
        return APOLLOCAMELRSF(
            parameters, **common, rank=args.rank, lrsf_rank=args.rank,
            sf_beta1=0.9, warmup_steps=0, seed=args.seed,
            scale=args.scale,
            norm_growth_limiter=not args.disable_norm_growth_limiter,
            norm_growth_rate=args.norm_growth_rate,
            fallback=fallback, came_backend="torch",
            delta_refresh=_lrsf_refresh_policy(args),
            orthogonal_refresh=_lrsf_orthogonal_refresh_policy(args),
        )
    if name == "APOLLOMini":
        return APOLLOMini(
            parameters, **common, scale=args.scale,
            norm_growth_limiter=not args.disable_norm_growth_limiter,
            norm_growth_rate=args.norm_growth_rate,
            fallback=fallback,
        )
    raise ValueError(f"unknown optimizer: {name}")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _loss(model: ImageAE, images: torch.Tensor) -> torch.Tensor:
    reconstruction, _ = model(images)
    # Compute the scalar in FP32 while retaining the gradient path through
    # the BF16 reconstruction.  This matches the image_ae loss boundary.
    return (reconstruction.float() - images.float()).square().mean()


def run_case(
    name: str,
    args,
    device: torch.device,
    dtype: torch.dtype,
    initial_state: dict[str, torch.Tensor],
    images: torch.Tensor,
) -> dict[str, object]:
    model = build_model(args, device, dtype)
    model.load_state_dict(
        {key: value.to(device=device, dtype=dtype) for key, value in initial_state.items()}
    )
    optimizer = build_optimizer(name, model, args)
    if hasattr(optimizer, "train"):
        optimizer.train()

    with torch.no_grad():
        initial_loss = _loss(model, images)

    for _ in range(args.warmup):
        optimizer.zero_grad(set_to_none=True)
        _loss(model, images).backward()
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
        loss = _loss(model, images)
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
        with torch.no_grad():
            losses.append(float(_loss(model, images)))
    _synchronize(device)

    state_bytes, state_elements = state_metrics(optimizer)
    return {
        "status": "passed"
        if torch.isfinite(torch.tensor([float(initial_loss), *losses])).all()
        else "failed",
        "optimizer": name,
        "rank": (
            args.rank if name in {
                "CAME-LRSF", "APOLLO", "APOLLO-CAME", "APOLLO-CAME-LRSF",
            }
            else (1 if name == "APOLLOMini" else None)
        ),
        "learning_rate": args.learning_rate,
        "scale": args.scale if name not in {"CAME", "CAME-SF"} else None,
        "norm_growth_limiter": (
            not args.disable_norm_growth_limiter
            if name not in {"CAME", "CAME-SF"} else None
        ),
        "norm_growth_rate": (
            args.norm_growth_rate
            if name not in {"CAME", "CAME-SF"} else None
        ),
        "backend_counts": {
            "apollo": sum(
                state.get("backend") == "apollo"
                for state in optimizer.state.values()
            ),
            "came": sum(
                state.get("backend") == "came"
                for state in optimizer.state.values()
            ),
            "sgd": sum(
                state.get("backend") == "sgd"
                for state in optimizer.state.values()
            ),
            # APOLLO-CAME-LRSF keeps backend="apollo" for its update path;
            # the delta key is the authoritative marker for its second path.
            "lrsf": sum(
                "lrsf_delta" in state
                for state in optimizer.state.values()
            ),
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
        **(
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
            if device.type == "cuda"
            else {}
        ),
    }


def run(args) -> dict[str, object]:
    args.ranks = getattr(args, "ranks", None)
    args.learning_rates = getattr(args, "learning_rates", None)
    args.scales = getattr(args, "scales", None)
    args.norm_growth_rate = getattr(args, "norm_growth_rate", 1.01)
    args.norm_growth_rates = getattr(args, "norm_growth_rates", None)
    args.weight_decay = getattr(args, "weight_decay", 0.0)
    args.weight_decays = getattr(args, "weight_decays", None)
    args.disable_norm_growth_limiter = getattr(
        args, "disable_norm_growth_limiter", False
    )
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.batch_size <= 0 or args.image_size <= 0:
        raise ValueError("--batch-size and --image-size must be positive")
    if args.rank <= 0 or args.latent_channels <= 0:
        raise ValueError("--rank and --latent-channels must be positive")
    if args.bottleneck_channels <= 0 or args.downsample_stages <= 0:
        raise ValueError("--bottleneck-channels and --downsample-stages must be positive")
    divisor = 2 ** args.downsample_stages
    if args.image_size < divisor or args.image_size % divisor:
        raise ValueError(
            "--image-size must be divisible by 2**--downsample-stages"
        )
    if args.learning_rate < 0.0 or args.scale <= 0.0:
        raise ValueError("--learning-rate must be non-negative and --scale positive")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative")
    if args.norm_growth_rate <= 1.0:
        raise ValueError("--norm-growth-rate must be greater than 1.0")
    rank_values = args.ranks or (args.rank,)
    learning_rate_values = args.learning_rates or (args.learning_rate,)
    weight_decay_values = args.weight_decays or (args.weight_decay,)
    scale_values = args.scales or (args.scale,)
    norm_growth_rate_values = args.norm_growth_rates or (args.norm_growth_rate,)
    if any(rank <= 0 for rank in rank_values):
        raise ValueError("all ranks must be positive")
    if any(rate <= 0.0 for rate in learning_rate_values):
        raise ValueError("all learning rates must be positive")
    if any(decay < 0.0 for decay in weight_decay_values):
        raise ValueError("all weight decays must be non-negative")
    if any(scale <= 0.0 for scale in scale_values):
        raise ValueError("all scales must be positive")
    if any(rate <= 1.0 for rate in norm_growth_rate_values):
        raise ValueError("all norm growth rates must be greater than 1.0")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    images = torch.rand(
        args.batch_size, 3, args.image_size, args.image_size,
        generator=generator,
    ).to(device=device, dtype=dtype)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 1)
        initial_model = build_model(args, torch.device("cpu"), torch.float32)
    initial_state = {
        key: value.detach().clone() for key, value in initial_model.state_dict().items()
    }

    cases: dict[str, object] = {}
    for name in args.optimizers:
        if name in {"CAME", "CAME-SF"}:
            configurations = [
                {
                    "rank": args.rank,
                    "learning_rate": rate,
                    "scale": args.scale,
                    "weight_decay": weight_decay,
                }
                for rate in learning_rate_values
                for weight_decay in weight_decay_values
            ]
        elif name == "APOLLOMini":
            configurations = [
                {
                    "rank": 1,
                    "learning_rate": rate,
                    "scale": scale,
                    "norm_growth_rate": growth_rate,
                    "weight_decay": weight_decay,
                }
                for rate in learning_rate_values
                for scale in scale_values
                for growth_rate in norm_growth_rate_values
                for weight_decay in weight_decay_values
            ]
        else:
            configurations = [
                {
                    "rank": rank,
                    "learning_rate": rate,
                    "scale": scale,
                    "norm_growth_rate": growth_rate,
                    "weight_decay": weight_decay,
                }
                for rank in rank_values
                for rate in learning_rate_values
                for scale in scale_values
                for growth_rate in norm_growth_rate_values
                for weight_decay in weight_decay_values
            ]

        for configuration in configurations:
            case_args = argparse.Namespace(**vars(args))
            case_args.rank = configuration["rank"]
            case_args.learning_rate = configuration["learning_rate"]
            case_args.scale = configuration["scale"]
            case_args.weight_decay = configuration["weight_decay"]
            if name not in {"CAME", "CAME-SF"}:
                case_args.norm_growth_rate = configuration["norm_growth_rate"]
            is_sweep = any(
                value is not None
                for value in (
                    args.ranks, args.learning_rates, args.scales,
                    args.norm_growth_rates, args.weight_decays,
                )
            )
            if not is_sweep:
                case_name = name
            elif name in {"CAME", "CAME-SF"}:
                case_name = f"{name}@lr={configuration['learning_rate']:g}"
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
            if name not in {"CAME", "CAME-SF"} and args.norm_growth_rates is not None:
                case_name += f",growth={configuration['norm_growth_rate']:g}"
            if args.weight_decays is not None:
                case_name += f",wd={configuration['weight_decay']:g}"
            try:
                cases[case_name] = run_case(
                    name, case_args, device, dtype, initial_state, images,
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
        "image_size": args.image_size,
        "seed": args.seed,
        "rank": args.rank,
        "ranks": list(rank_values),
        "learning_rate": args.learning_rate,
        "learning_rates": list(learning_rate_values),
        "weight_decay": args.weight_decay,
        "weight_decays": list(weight_decay_values),
        "scale": args.scale,
        "scales": list(scale_values),
        "norm_growth_limiter": not args.disable_norm_growth_limiter,
        "norm_growth_rate": args.norm_growth_rate,
        "norm_growth_rates": list(norm_growth_rate_values),
        "model": {
            "latent_channels": args.latent_channels,
            "bottleneck_channels": args.bottleneck_channels,
            "downsample_stages": args.downsample_stages,
            "encoder": "residual_conv_ffn",
            "decoder": "residual_conv_ffn",
        },
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
