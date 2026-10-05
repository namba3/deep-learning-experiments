"""Compare ImageAE optimizers on a deterministic CIFAR-10 subset.

Unlike ``image_ae_optimizer_convergence``, this probe reads real CIFAR-10
images.  It intentionally uses a fixed, small subset so CPU runs remain
practical while keeping the model, optimizer, loss, and validation boundary
the same as the ImageAE training path.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from time import perf_counter

import torch
from torchvision import datasets, transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from verify.image_ae_optimizer_convergence import (  # noqa: E402
    _loss,
    build_model,
    build_optimizer,
    resolve_device,
    resolve_dtype,
    state_metrics,
)


OPTIMIZERS = ("CAME", "CAME-SF", "CAME-LRSF", "APOLLO-CAME-LRSF")


def _parse_optimizers(value: str) -> tuple[str, ...]:
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
        raise argparse.ArgumentTypeError(
            "ranks must be comma-separated integers"
        ) from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("ranks must be positive")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="cifar10/data")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-validation-samples", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument(
        "--ranks", type=_parse_positive_ints, default=None,
        help="Optional comma-separated LRSF ranks to sweep.",
    )
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--bottleneck-channels", type=int, default=16)
    parser.add_argument("--downsample-stages", type=int, default=2)
    parser.add_argument(
        "--refresh-mode", choices=("none", "hard", "smooth"), default="none",
        help="LRSF delta projection refresh mode. Default: none.",
    )
    parser.add_argument("--refresh-interval", type=int, default=200)
    parser.add_argument("--refresh-window", type=int, default=200)
    parser.add_argument(
        "--refresh-mix",
        choices=("linear", "smoothstep", "stochastic", "ema"),
        default="smoothstep",
    )
    parser.add_argument("--orthogonal-refresh-rate", type=float, default=0.0)
    parser.add_argument(
        "--orthogonal-refresh-direction",
        choices=("random", "loss_directed", "loss_lowering"),
        default="random",
        help="LRSF orthogonal direction. Default: random.",
    )
    parser.add_argument(
        "--orthogonal-refresh-signal",
        choices=("gradient", "effective_update"),
        default="gradient",
        help="Signal for LRSF loss-directed rotation. Default: gradient.",
    )
    parser.add_argument(
        "--record-step-metrics", action="store_true",
        help="Record refresh-event loss and recovery diagnostics.",
    )
    parser.add_argument(
        "--record-update-norms", action="store_true",
        help=(
            "Record parameter update-norm statistics. This adds host copies "
            "and is excluded from optimizer step timing."
        ),
    )
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
    if args.rank <= 0 or args.latent_channels <= 0 or args.bottleneck_channels <= 0:
        raise ValueError("rank and channel counts must be positive")
    if args.downsample_stages <= 0:
        raise ValueError("downsample stages must be positive")
    if args.refresh_interval < 0 or args.refresh_window < 0:
        raise ValueError("refresh interval and window must be non-negative")
    if args.refresh_mode != "none":
        if args.refresh_interval <= 0:
            raise ValueError("refresh interval must be positive when enabled")
        if args.refresh_mode == "smooth" and args.refresh_window <= 0:
            raise ValueError("smooth refresh window must be positive")
    if args.orthogonal_refresh_rate < 0.0:
        raise ValueError("orthogonal refresh rate must be non-negative")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _backend_counts(optimizer: torch.optim.Optimizer) -> dict[str, int]:
    return {
        "apollo": sum(
            state.get("backend") == "apollo" for state in optimizer.state.values()
        ),
        "came": sum(
            state.get("backend") == "came" for state in optimizer.state.values()
        ),
        "lrsf": sum(
            "lrsf_delta" in state for state in optimizer.state.values()
        ),
        "sf_full": sum(
            state.get("backend") == "sf_full" for state in optimizer.state.values()
        ),
    }


def _refresh_status(optimizer: torch.optim.Optimizer) -> tuple[int, bool, int]:
    counts = [
        int(state.get("refresh_count", 0))
        for state in optimizer.state.values()
    ]
    progresses = [
        int(state.get("refresh_progress", 0))
        for state in optimizer.state.values()
    ]
    return (
        max(counts, default=0),
        any(state.get("refresh_active", False) for state in optimizer.state.values()),
        max(progresses, default=0),
    )


def _orthogonal_refresh_count(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        int(state.get("orthogonal_refresh_count", 0))
        for state in optimizer.state.values()
    )


def _capture_loss_lowering_states(
    optimizer: torch.optim.Optimizer,
) -> list[tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor]]:
    """Snapshot LRSF basis/delta for the optional rotation diagnostic."""
    snapshots = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if (
                not state
                or "lrsf_projection" not in state
                or "lrsf_delta" not in state
            ):
                continue
            snapshots.append(
                (
                    parameter,
                    state["lrsf_projection"].detach().clone(),
                    state["lrsf_delta"].detach().clone(),
                )
            )
    return snapshots


def _loss_lowering_linear_proxy(
    snapshots: list[tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    use_new_projection: bool,
    use_new_delta: bool,
) -> float | None:
    """Evaluate ``<G, D R^T>`` or ``<G, R^T D>`` for diagnostic snapshots."""
    values = []
    for parameter, old_projection, old_delta in snapshots:
        if parameter.grad is None:
            continue
        state = optimizer.state[parameter]
        projection = (
            state["lrsf_projection"] if use_new_projection else old_projection
        ).detach().float()
        delta = (
            state["lrsf_delta"] if use_new_delta else old_delta
        ).detach().float()
        gradient = parameter.grad.detach().float().reshape(
            parameter.grad.shape[0], -1,
        )
        if projection.shape[0] >= projection.shape[1]:
            decoded_delta = delta.matmul(projection.transpose(0, 1))
        else:
            decoded_delta = projection.transpose(0, 1).matmul(delta)
        values.append((gradient * decoded_delta).sum())
    if not values:
        return None
    return float(torch.stack(values).sum().item())


def _complete_refresh_events(
    events: list[dict[str, object]], step_loss_history: list[float],
) -> None:
    """Add next-observation and recovery metrics to refresh events."""
    for event in events:
        refresh_index = int(event["step"]) - 1
        if refresh_index >= len(step_loss_history):
            continue
        pre_refresh_loss = step_loss_history[refresh_index]
        event["loss_before_refresh"] = pre_refresh_loss
        if refresh_index + 1 >= len(step_loss_history):
            event["next_observed_loss"] = None
            event["next_observed_loss_delta"] = None
            event["recovery_steps_to_pre_refresh_loss"] = None
            continue
        next_loss = step_loss_history[refresh_index + 1]
        event["next_observed_loss"] = next_loss
        event["next_observed_loss_delta"] = next_loss - pre_refresh_loss
        recovery_steps = None
        for offset, observed_loss in enumerate(
            step_loss_history[refresh_index + 1:], start=1
        ):
            if observed_loss <= pre_refresh_loss:
                recovery_steps = offset
                break
        event["recovery_steps_to_pre_refresh_loss"] = recovery_steps


def _build_args(args):
    """Provide the common optimizer probe contract to the shared factory."""
    return argparse.Namespace(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        rank=args.rank,
        scale=1.0,
        disable_norm_growth_limiter=True,
        norm_growth_rate=1.01,
        seed=args.seed,
        came_lrsf_refresh_mode=args.refresh_mode,
        came_lrsf_refresh_interval=args.refresh_interval,
        came_lrsf_refresh_window=args.refresh_window,
        came_lrsf_refresh_mix=args.refresh_mix,
        came_lrsf_orthogonal_refresh_rate=args.orthogonal_refresh_rate,
        came_lrsf_orthogonal_refresh_direction=getattr(
            args, "orthogonal_refresh_direction", "random",
        ),
        came_lrsf_orthogonal_refresh_signal=getattr(
            args, "orthogonal_refresh_signal", "gradient",
        ),
    )


def run_case(
    name: str,
    args,
    device: torch.device,
    dtype: torch.dtype,
    initial_state: dict[str, torch.Tensor],
    train_images: torch.Tensor,
    validation_images: torch.Tensor,
) -> dict[str, object]:
    model_args = argparse.Namespace(
        latent_channels=args.latent_channels,
        bottleneck_channels=args.bottleneck_channels,
        downsample_stages=args.downsample_stages,
    )
    model = build_model(model_args, device, dtype)
    model.load_state_dict({
        key: value.to(device=device, dtype=dtype)
        for key, value in initial_state.items()
    })
    optimizer = build_optimizer(name, model, _build_args(args))
    if hasattr(optimizer, "train"):
        optimizer.train()

    total_steps = 0
    step_seconds = 0.0
    train_history: list[float] = []
    step_loss_history: list[float] = []
    validation_history: list[float] = []
    refresh_events: list[dict[str, object]] = []
    orthogonal_refresh_events: list[dict[str, object]] = []
    orthogonal_refresh_steps = 0
    pending_refresh_event: dict[str, object] | None = None
    update_norm_history: list[float] = []
    loss_lowering_snapshots = []
    peak_state_bytes = 0
    peak_state_elements = 0
    permutation_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    epoch_permutations = tuple(
        torch.randperm(len(train_images), generator=permutation_generator)
        for _ in range(args.epochs)
    )
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
            images = train_images[batch_indices].to(
                device=device, dtype=dtype,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, images)
            loss_value = float(loss.detach())
            if pending_refresh_event is not None:
                pending_refresh_event["next_step_loss_before"] = loss_value
                pending_refresh_event = None
            loss.backward()
            parameters_before_step = None
            if getattr(args, "record_update_norms", False):
                parameters_before_step = [
                    parameter.detach().float().cpu().clone()
                    for parameter in model.parameters()
                ]
            refresh_count_before, _, _ = _refresh_status(optimizer)
            orthogonal_count_before = _orthogonal_refresh_count(optimizer)
            if (
                getattr(args, "record_step_metrics", False)
                and getattr(args, "orthogonal_refresh_direction", "random")
                == "loss_lowering"
            ):
                loss_lowering_snapshots = _capture_loss_lowering_states(optimizer)
            else:
                loss_lowering_snapshots = []
            started = perf_counter()
            optimizer.step()
            _synchronize(device)
            step_seconds += perf_counter() - started
            if parameters_before_step is not None:
                update_squared = sum(
                    (
                        parameter.detach().float().cpu() - before
                    ).square().sum().item()
                    for parameter, before in zip(
                        model.parameters(), parameters_before_step,
                    )
                )
                update_norm_history.append(update_squared ** 0.5)
            current_state_bytes, current_state_elements = state_metrics(optimizer)
            peak_state_bytes = max(peak_state_bytes, current_state_bytes)
            peak_state_elements = max(peak_state_elements, current_state_elements)
            epoch_loss += loss_value
            step_loss_history.append(loss_value)
            batches += 1
            total_steps += 1
            refresh_count_after, refresh_active, refresh_progress = _refresh_status(optimizer)
            orthogonal_count_after = _orthogonal_refresh_count(optimizer)
            if orthogonal_count_after > orthogonal_count_before:
                orthogonal_refresh_steps += 1
            if refresh_count_after > refresh_count_before:
                with torch.no_grad():
                    post_update_loss = float(_loss(model, images))
                pending_refresh_event = {
                    "step": total_steps,
                    "refresh_count": refresh_count_after,
                    "loss_before": loss_value,
                    "loss_after": post_update_loss,
                    "same_batch_delta": post_update_loss - loss_value,
                    "refresh_active": refresh_active,
                    "refresh_progress": refresh_progress,
                }
                refresh_events.append(pending_refresh_event)
            if (
                getattr(args, "record_step_metrics", False)
                and orthogonal_count_after > orthogonal_count_before
            ):
                with torch.no_grad():
                    post_update_loss = float(_loss(model, images))
                orthogonal_refresh_events.append({
                    "step": total_steps,
                    "refresh_count": orthogonal_count_after,
                    "loss_before": loss_value,
                    "loss_after": post_update_loss,
                    "same_batch_delta": post_update_loss - loss_value,
                })
                if (
                    getattr(args, "orthogonal_refresh_direction", "random")
                    == "loss_lowering"
                ):
                    if (
                        loss_lowering_snapshots
                        and refresh_count_after == refresh_count_before
                    ):
                        proxy_before = _loss_lowering_linear_proxy(
                            loss_lowering_snapshots,
                            optimizer,
                            use_new_projection=False,
                            use_new_delta=False,
                        )
                        proxy_after = _loss_lowering_linear_proxy(
                            loss_lowering_snapshots,
                            optimizer,
                            use_new_projection=True,
                            use_new_delta=False,
                        )
                        proxy_after_step = _loss_lowering_linear_proxy(
                            loss_lowering_snapshots,
                            optimizer,
                            use_new_projection=True,
                            use_new_delta=True,
                        )
                        event = orthogonal_refresh_events[-1]
                        event.update({
                            "loss_lowering_proxy_before": proxy_before,
                            "loss_lowering_proxy_after_frozen_delta": proxy_after,
                            "loss_lowering_proxy_decrease": (
                                None
                                if proxy_before is None or proxy_after is None
                                else proxy_before - proxy_after
                            ),
                            "loss_lowering_proxy_after_step": (
                                proxy_after_step
                            ),
                            "loss_lowering_proxy_decrease_after_step": (
                                None
                                if proxy_before is None
                                or proxy_after_step is None
                                else proxy_before - proxy_after_step
                            ),
                        })
                    else:
                        orthogonal_refresh_events[-1].update({
                            "loss_lowering_proxy_before": None,
                            "loss_lowering_proxy_after_frozen_delta": None,
                            "loss_lowering_proxy_decrease": None,
                            "loss_lowering_proxy_after_step": None,
                            "loss_lowering_proxy_decrease_after_step": None,
                        })
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
        if hasattr(optimizer, "train") and epoch < args.epochs - 1:
            optimizer.train()
            model.train()

    state_bytes, state_elements = state_metrics(optimizer)
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    peak_delta_allocated_bytes: int | None = None
    if device.type == "cuda":
        peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
        peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
        peak_delta_allocated_bytes = peak_allocated_bytes - baseline_allocated
    result: dict[str, object] = {
        "status": "passed",
        "optimizer": name,
        "rank": args.rank if "LRSF" in name else None,
        "epochs": args.epochs,
        "total_steps": total_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "orthogonal_refresh_rate": args.orthogonal_refresh_rate,
        "orthogonal_refresh_direction": getattr(
            args, "orthogonal_refresh_direction", "random",
        ),
        "orthogonal_refresh_signal": getattr(
            args, "orthogonal_refresh_signal", "gradient",
        ),
        "train_loss_history": train_history,
        "step_loss_history": step_loss_history,
        "validation_loss_history": validation_history,
        "final_train_loss": train_history[-1],
        "final_validation_loss": validation_history[-1],
        "persistent_state_bytes": state_bytes,
        "persistent_state_elements": state_elements,
        "peak_persistent_state_bytes": peak_state_bytes,
        "peak_persistent_state_elements": peak_state_elements,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "peak_delta_allocated_bytes": peak_delta_allocated_bytes,
        "backend_counts": _backend_counts(optimizer),
        "refresh_mode": args.refresh_mode,
        "refresh_events": refresh_events,
        "orthogonal_refresh_steps": orthogonal_refresh_steps,
        "host_seconds_per_optimizer_step": step_seconds / total_steps,
        "parameter_numel": sum(parameter.numel() for parameter in model.parameters()),
    }
    if getattr(args, "record_step_metrics", False):
        _complete_refresh_events(refresh_events, step_loss_history)
        _complete_refresh_events(orthogonal_refresh_events, step_loss_history)
        result["step_loss_history"] = step_loss_history
        result["refresh_events"] = refresh_events
        result["orthogonal_refresh_events"] = orthogonal_refresh_events
    if getattr(args, "record_update_norms", False):
        result["update_norm_mean"] = statistics.fmean(update_norm_history)
        result["update_norm_variance"] = statistics.pvariance(update_norm_history)
        result["update_norm_history"] = update_norm_history
    return result


def run(args) -> dict[str, object]:
    args.refresh_mode = getattr(args, "refresh_mode", "none")
    args.refresh_interval = getattr(args, "refresh_interval", 200)
    args.refresh_window = getattr(args, "refresh_window", 200)
    args.refresh_mix = getattr(args, "refresh_mix", "smoothstep")
    _validate_args(args)
    args.ranks = getattr(args, "ranks", None)
    rank_values = args.ranks or (args.rank,)
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

    cases = {}
    for name in args.optimizers:
        selected_ranks = rank_values if "LRSF" in name else (args.rank,)
        for rank in selected_ranks:
            case_args = argparse.Namespace(**vars(args))
            case_args.rank = rank
            case_name = name if "LRSF" not in name or args.ranks is None else (
                f"{name}@rank={rank}"
            )
            cases[case_name] = run_case(
                name, case_args, device, dtype, initial_state,
                train_images, validation_images,
            )
    return {
        "status": "passed",
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "data_dir": args.data_dir,
        "train_samples": len(train_images),
        "validation_samples": len(validation_images),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "rank": args.rank,
        "ranks": list(rank_values),
        "refresh": {
            "mode": args.refresh_mode,
            "interval": args.refresh_interval,
            "window": args.refresh_window,
            "mix": args.refresh_mix,
            "orthogonal_rate": args.orthogonal_refresh_rate,
            "orthogonal_direction": getattr(
                args, "orthogonal_refresh_direction", "random",
            ),
            "orthogonal_signal": getattr(
                args, "orthogonal_refresh_signal", "gradient",
            ),
        },
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
