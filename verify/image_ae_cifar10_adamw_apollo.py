"""Compare AdamW and APOLLO variants on a deterministic CIFAR-10 subset.

The probe keeps the dataset subset, initial ImageAE weights, epoch
permutations, precision, and loss boundary identical across optimizers.  It
is intended to establish a fair baseline before measuring projection refresh
as a possible source of structured exploration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from time import perf_counter

import torch
from torchvision import datasets, transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from optimizers.factory import build_optimizer as build_registered_optimizer  # noqa: E402
from verify.image_ae_optimizer_convergence import (  # noqa: E402
    _loss,
    build_model,
    resolve_device,
    resolve_dtype,
    state_metrics,
)


OPTIMIZERS = ("AdamW", "CAME", "APOLLO", "APOLLO-CAME", "APOLLO-Mini")


def _parse_optimizers(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values or any(value not in OPTIMIZERS for value in values):
        raise argparse.ArgumentTypeError(
            "optimizers must be a comma-separated subset of: "
            + ",".join(OPTIMIZERS)
        )
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="cifar10/data")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-validation-samples", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--update-proj-gap", type=int, default=200)
    parser.add_argument(
        "--projection-refresh-mode", choices=("none", "hard", "smooth"), default="hard",
        help="APOLLO projection refresh policy.",
    )
    parser.add_argument(
        "--projection-refresh-window", type=int, default=200,
        help="APOLLO smooth refresh window in optimizer steps. Default: 200.",
    )
    parser.add_argument(
        "--projection-refresh-mix",
        choices=("linear", "smoothstep", "stochastic", "ema"),
        default="smoothstep",
        help="APOLLO smooth refresh mixing curve. Default: smoothstep.",
    )
    parser.add_argument(
        "--projection-refresh-state", choices=("reset", "transport"), default="reset",
        help="Reset or overlap-transport APOLLO low-rank moments at refresh.",
    )
    parser.add_argument(
        "--orthogonal-refresh-rate", type=float, default=0.0,
        help="Per-step APOLLO projection rotation rate. Default: 0.",
    )
    parser.add_argument(
        "--orthogonal-refresh-direction",
        choices=("random", "loss_directed"),
        default="random",
        help=(
            "Orthogonal direction: random or gradient-energy loss proxy. "
            "Default: random."
        ),
    )
    parser.add_argument(
        "--update-norm-variance-cap",
        type=float,
        default=None,
        help=(
            "Experimental cumulative upper-tail update-norm variance cap "
            "for APOLLO. Default: disabled."
        ),
    )
    parser.add_argument(
        "--freeze-projection",
        action="store_true",
        help="Use a practically unreachable refresh interval for APOLLO variants.",
    )
    parser.add_argument(
        "--disable-norm-growth-limiter", action="store_true",
        help="Disable APOLLO's norm-growth limiter for this probe.",
    )
    parser.add_argument(
        "--record-step-metrics", action="store_true",
        help="Record per-step train losses and detected projection refresh events.",
    )
    parser.add_argument(
        "--record-update-norms", action="store_true",
        help=(
            "Record parameter update-norm statistics. This adds host copies and "
            "is excluded from optimizer step timing."
        ),
    )
    parser.add_argument("--norm-growth-rate", type=float, default=1.01)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--bottleneck-channels", type=int, default=256)
    parser.add_argument("--downsample-stages", type=int, default=3)
    parser.add_argument("--optimizers", type=_parse_optimizers, default=OPTIMIZERS)
    return parser.parse_args(argv)


def _load_images(data_dir: str, max_samples: int, *, train: bool) -> torch.Tensor:
    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=False,
        transform=transforms.ToTensor(),
    )
    count = min(int(max_samples), len(dataset))
    if count <= 0:
        raise ValueError("sample count must be positive")
    return torch.stack([dataset[index][0] for index in range(count)])


def _validate_args(args) -> None:
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.max_train_samples <= 0 or args.max_validation_samples <= 0:
        raise ValueError("sample limits must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if args.rank <= 0 or args.scale <= 0.0:
        raise ValueError("rank and scale must be positive")
    if args.update_proj_gap <= 0:
        raise ValueError("--update-proj-gap must be positive")
    if args.projection_refresh_window < 0:
        raise ValueError("--projection-refresh-window must be non-negative")
    if args.projection_refresh_mode == "smooth" and args.projection_refresh_window <= 0:
        raise ValueError(
            "--projection-refresh-window must be positive for smooth mode"
        )
    if args.norm_growth_rate <= 1.0:
        raise ValueError("--norm-growth-rate must be greater than 1.0")
    if args.orthogonal_refresh_rate < 0.0:
        raise ValueError("--orthogonal-refresh-rate must be non-negative")
    if args.update_norm_variance_cap is not None and (
        not math.isfinite(args.update_norm_variance_cap)
        or args.update_norm_variance_cap < 0.0
    ):
        raise ValueError(
            "--update-norm-variance-cap must be finite and non-negative"
        )
    if args.latent_channels <= 0 or args.bottleneck_channels <= 0:
        raise ValueError("channel counts must be positive")
    if args.downsample_stages <= 0:
        raise ValueError("downsample stages must be positive")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_optimizer_args(args):
    """Adapt this probe's compact CLI to the shared optimizer factory."""
    return argparse.Namespace(
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        apollo_rank=args.rank,
        apollo_scale=args.scale,
        apollo_update_proj_gap=(2**31 - 1 if args.freeze_projection else args.update_proj_gap),
        apollo_projection_refresh={
            "mode": args.projection_refresh_mode,
            "interval": 2**31 - 1 if args.freeze_projection else args.update_proj_gap,
            "window": args.projection_refresh_window,
            "mix": args.projection_refresh_mix,
        },
        apollo_projection_refresh_state=args.projection_refresh_state,
        apollo_orthogonal_refresh_rate=args.orthogonal_refresh_rate,
        apollo_orthogonal_refresh_direction=args.orthogonal_refresh_direction,
        apollo_update_norm_variance_cap=args.update_norm_variance_cap,
        seed=args.seed,
        apollo_disable_norm_growth_limiter=args.disable_norm_growth_limiter,
        apollo_norm_growth_rate=args.norm_growth_rate,
        apollo_fallback="came",
        apollo_matrix_fallback="auto",
        apollo_came_backend="torch",
    )


def _backend_counts(optimizer: torch.optim.Optimizer) -> dict[str, int]:
    return {
        "apollo": sum(
            state.get("backend") == "apollo" for state in optimizer.state.values()
        ),
        "came": sum(
            state.get("backend") == "came" for state in optimizer.state.values()
        ),
        "sgd": sum(
            state.get("backend") == "sgd" for state in optimizer.state.values()
        ),
    }


def _projection_seed_snapshot(optimizer: torch.optim.Optimizer) -> dict[int, int]:
    """Return projection seeds for initialized APOLLO states."""
    return {
        id(parameter): int(state["projection_seed"])
        for parameter, state in optimizer.state.items()
        if "projection_seed" in state
    }


def _orthogonal_refresh_count(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        int(state.get("orthogonal_refresh_count", 0))
        for state in optimizer.state.values()
    )


def _update_norm_variance_capped_count(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        value = state.get("update_norm_variance_capped_count", 0)
        total += int(value.item()) if isinstance(value, torch.Tensor) else int(value)
    return total


def _refresh_is_active(optimizer: torch.optim.Optimizer) -> bool:
    return any(
        bool(state.get("refresh_active", False))
        for state in optimizer.state.values()
    )


def _projection_snapshot(optimizer: torch.optim.Optimizer) -> dict[int, dict[str, torch.Tensor]]:
    snapshot = {}
    for parameter, state in optimizer.state.items():
        if "projection" not in state:
            continue
        values = {
            "projection": state["projection"].detach().float().cpu().clone(),
        }
        if "refresh_next_projection" in state:
            values["refresh_next_projection"] = (
                state["refresh_next_projection"].detach().float().cpu().clone()
            )
        snapshot[id(parameter)] = values
    return snapshot


def run_case(
    name: str,
    args,
    device: torch.device,
    dtype: torch.dtype,
    initial_state: dict[str, torch.Tensor],
    train_images: torch.Tensor,
    validation_images: torch.Tensor,
) -> dict[str, object]:
    model = build_model(args, device, dtype)
    model.load_state_dict({
        key: value.to(device=device, dtype=dtype)
        for key, value in initial_state.items()
    })
    optimizer = build_registered_optimizer(
        name, model.parameters(), _build_optimizer_args(args)
    )

    if hasattr(optimizer, "train"):
        optimizer.train()
    permutation_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    epoch_permutations = tuple(
        torch.randperm(len(train_images), generator=permutation_generator)
        for _ in range(args.epochs)
    )
    total_steps = 0
    step_seconds = 0.0
    train_history: list[float] = []
    validation_history: list[float] = []
    step_loss_history: list[float] = []
    refresh_events: list[dict[str, object]] = []
    refresh_active_steps = 0
    orthogonal_refresh_steps = 0
    previous_orthogonal_count = 0
    update_norm_history: list[float] = []
    previous_projection_seeds: dict[int, int] = {}

    _synchronize(device)
    baseline_allocated = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)

    model.train()
    for epoch, permutation in enumerate(epoch_permutations):
        epoch_loss = 0.0
        batches = 0
        for offset in range(0, len(train_images), args.batch_size):
            batch_indices = permutation[offset:offset + args.batch_size]
            images = train_images[batch_indices].to(device=device, dtype=dtype)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, images)
            loss.backward()
            if args.record_update_norms:
                parameters_before_step = [
                    parameter.detach().float().cpu().clone()
                    for parameter in model.parameters()
                ]
            refresh_candidate_step = total_steps + 1
            should_capture_projection = (
                args.record_step_metrics
                and args.projection_refresh_mode in {"hard", "smooth"}
                and refresh_candidate_step % args.update_proj_gap == 0
            )
            previous_projections = (
                _projection_snapshot(optimizer)
                if should_capture_projection
                else {}
            )
            started = perf_counter()
            was_refresh_active = _refresh_is_active(optimizer)
            optimizer.step()
            _synchronize(device)
            step_seconds += perf_counter() - started
            loss_value = float(loss.detach())
            step_loss_history.append(loss_value)
            if args.record_update_norms:
                update_squared = sum(
                    (parameter.detach().float().cpu() - before).square().sum().item()
                    for parameter, before in zip(model.parameters(), parameters_before_step)
                )
                update_norm_history.append(update_squared**0.5)
            current_projection_seeds = _projection_seed_snapshot(optimizer)
            refreshed_parameters = [
                parameter_id
                for parameter_id, seed in current_projection_seeds.items()
                if (
                    parameter_id in previous_projection_seeds
                    and previous_projection_seeds[parameter_id] != seed
                )
            ]
            if refreshed_parameters:
                current_projections = _projection_snapshot(optimizer)
                projection_change_max = 0.0
                for parameter_id in refreshed_parameters:
                    previous = previous_projections.get(parameter_id)
                    current = current_projections.get(parameter_id)
                    if previous is None or current is None:
                        continue
                    old_projection = previous["projection"]
                    candidates = [current["projection"]]
                    if "refresh_next_projection" in current:
                        candidates.append(current["refresh_next_projection"])
                    projection_change_max = max(
                        projection_change_max,
                        *(
                            float((candidate - old_projection).abs().max())
                            for candidate in candidates
                        ),
                    )
                refresh_events.append({
                    "step": total_steps + 1,
                    "parameter_count": len(refreshed_parameters),
                    "projection_change_max_abs": projection_change_max,
                })
            if (
                args.projection_refresh_mode == "smooth"
                and (was_refresh_active or refreshed_parameters
                     or _refresh_is_active(optimizer))
            ):
                refresh_active_steps += 1
            current_orthogonal_count = _orthogonal_refresh_count(optimizer)
            if current_orthogonal_count > previous_orthogonal_count:
                orthogonal_refresh_steps += 1
            previous_orthogonal_count = current_orthogonal_count
            previous_projection_seeds = current_projection_seeds
            epoch_loss += loss_value
            batches += 1
            total_steps += 1
        train_history.append(epoch_loss / batches)

        if hasattr(optimizer, "eval"):
            optimizer.eval()
        model.eval()
        with torch.no_grad():
            validation_loss = 0.0
            batches = 0
            for offset in range(0, len(validation_images), args.batch_size):
                images = validation_images[offset:offset + args.batch_size].to(
                    device=device, dtype=dtype,
                )
                validation_loss += float(_loss(model, images))
                batches += 1
        validation_history.append(validation_loss / batches)
        if epoch < args.epochs - 1:
            if hasattr(optimizer, "train"):
                optimizer.train()
            model.train()

    _synchronize(device)
    state_bytes, state_elements = state_metrics(optimizer)
    losses = train_history + validation_history
    result: dict[str, object] = {
        "status": "passed" if all(torch.isfinite(torch.tensor(loss)) for loss in losses) else "failed",
        "optimizer": name,
        "rank": args.rank if name.startswith("APOLLO") and name != "APOLLO-Mini" else (
            1 if name == "APOLLO-Mini" else None
        ),
        "projection_mode": (
            "frozen" if args.freeze_projection else "refresh"
        ) if name.startswith("APOLLO") else None,
        "update_proj_gap": (
            2**31 - 1 if args.freeze_projection else args.update_proj_gap
        ) if name.startswith("APOLLO") else None,
        "projection_refresh_state": (
            args.projection_refresh_state
            if name.startswith("APOLLO") else None
        ),
        "projection_refresh_mode": (
            args.projection_refresh_mode
            if name.startswith("APOLLO") else None
        ),
        "projection_refresh_window": (
            args.projection_refresh_window
            if name.startswith("APOLLO") else None
        ),
        "projection_refresh_mix": (
            args.projection_refresh_mix
            if name.startswith("APOLLO") else None
        ),
        "orthogonal_refresh_rate": (
            args.orthogonal_refresh_rate
            if name.startswith("APOLLO") else None
        ),
        "orthogonal_refresh_direction": (
            args.orthogonal_refresh_direction
            if name.startswith("APOLLO") else None
        ),
        "update_norm_variance_cap": (
            args.update_norm_variance_cap
            if name.startswith("APOLLO") else None
        ),
        "update_norm_variance_capped_steps": (
            _update_norm_variance_capped_count(optimizer)
            if name.startswith("APOLLO") else 0
        ),
        "epochs": args.epochs,
        "total_steps": total_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_loss_history": train_history,
        "validation_loss_history": validation_history,
        "final_train_loss": train_history[-1],
        "final_validation_loss": validation_history[-1],
        "persistent_state_bytes": state_bytes,
        "persistent_state_elements": state_elements,
        "backend_counts": _backend_counts(optimizer),
        "host_seconds_per_optimizer_step": step_seconds / total_steps,
        "parameter_numel": sum(parameter.numel() for parameter in model.parameters()),
        "projection_refresh_steps": len(refresh_events),
        "projection_refresh_active_steps": refresh_active_steps,
        "orthogonal_refresh_steps": orthogonal_refresh_steps,
    }
    if args.record_step_metrics:
        for event in refresh_events:
            refresh_step = int(event["step"])
            refresh_index = refresh_step - 1
            event["loss_before_refresh"] = step_loss_history[refresh_index]
            if refresh_index + 1 < len(step_loss_history):
                loss_after = step_loss_history[refresh_index + 1]
                event["next_observed_loss"] = loss_after
                event["next_observed_loss_delta"] = (
                    loss_after - step_loss_history[refresh_index]
                )
                recovery_steps = None
                for offset, observed_loss in enumerate(
                    step_loss_history[refresh_index + 1:], start=1
                ):
                    if observed_loss <= step_loss_history[refresh_index]:
                        recovery_steps = offset
                        break
                event["recovery_steps_to_pre_refresh_loss"] = recovery_steps
            else:
                event["next_observed_loss"] = None
                event["next_observed_loss_delta"] = None
                event["recovery_steps_to_pre_refresh_loss"] = None
        result["step_loss_history"] = step_loss_history
        result["projection_refresh_events"] = refresh_events
    if args.record_update_norms:
        result["update_norm_mean"] = statistics.fmean(update_norm_history)
        result["update_norm_variance"] = statistics.pvariance(update_norm_history)
        result["update_norm_history"] = update_norm_history
    if device.type == "cuda":
        result.update({
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "peak_delta_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) - baseline_allocated
            ),
        })
    return result


def run(args) -> dict[str, object]:
    _validate_args(args)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    train_images = _load_images(args.data_dir, args.max_train_samples, train=True)
    validation_images = _load_images(
        args.data_dir, args.max_validation_samples, train=False,
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        initial_model = build_model(args, torch.device("cpu"), torch.float32)
    initial_state = {
        key: value.detach().clone() for key, value in initial_model.state_dict().items()
    }

    cases = {
        name: run_case(
            name, args, device, dtype, initial_state, train_images, validation_images,
        )
        for name in args.optimizers
    }
    return {
        "status": "passed" if all(
            case["status"] == "passed" for case in cases.values()
        ) else "failed",
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "data_dir": args.data_dir,
        "train_samples": len(train_images),
        "validation_samples": len(validation_images),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
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
