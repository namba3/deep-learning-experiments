"""Compare low-rank adapters on a deterministic CIFAR-10 ImageAE subset.

The probe uses the actual window-transformer ``ImageAE`` and real CIFAR-10
images, while keeping the initial weights, subset, optimizer, and validation
boundary matched across adapter types.  It is a short experiment for
parameterization screening; it is not a replacement for a full training run.

Example::

    python3 -m verify.image_ae_adapter_dataset_comparison \
        --data-dir cifar10/data --seeds 0,1,2 --epochs 3 \
        --max-train-samples 512 --max-validation-samples 256 \
        --output output/image-ae-adapter-dataset-comparison.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.low_rank import (  # noqa: E402
    canonicalize_adapter_type,
    inject_adapter,
    iter_adapter_modules,
    mark_only_adapter_trainable,
    merge_adapter,
    unmerge_adapter,
)
from verify.adapter_metrics import merge_is_equivalent, merge_tolerances  # noqa: E402
from image_ae.train import ImageAE  # noqa: E402


ADAPTERS = ("lora", "loha", "dora", "glu_lora", "rglu_lora")
TARGET_PATTERNS = (
    r"\.attention\.(qkv|output)$",
    r"\.ffn\.3$",
)


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "values must be comma-separated integers",
        ) from error
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("values must be non-negative integers")
    return values


def _parse_adapters(value: str) -> tuple[str, ...]:
    adapters = tuple(
        canonicalize_adapter_type(part.strip())
        for part in value.split(",")
        if part.strip()
    )
    if not adapters or any(adapter not in ADAPTERS for adapter in adapters):
        raise argparse.ArgumentTypeError(
            "adapters must be a comma-separated subset of: " + ",".join(ADAPTERS),
        )
    return adapters


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="cifar10/data")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seeds", type=_parse_csv_ints, default=(0, 1, 2))
    parser.add_argument("--adapters", type=_parse_adapters, default=ADAPTERS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-validation-samples", type=int, default=256)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--adapter-init", choices=("identity", "lora_warm"), default="identity",
        help="GLU-LoRA family initialization mode.",
    )
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--bottleneck-channels", type=int, default=64)
    parser.add_argument("--encoder-layers", type=int, default=1)
    parser.add_argument("--decoder-layers", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    if args.max_train_samples <= 0 or args.max_validation_samples <= 0:
        parser.error("sample limits must be positive")
    if args.rank <= 0 or args.learning_rate <= 0.0:
        parser.error("--rank and --learning-rate must be positive")
    if args.alpha is not None and args.alpha <= 0.0:
        parser.error("--alpha must be positive")
    if args.weight_decay < 0.0:
        parser.error("--weight-decay must be non-negative")
    if args.latent_channels <= 0 or args.bottleneck_channels <= 0:
        parser.error("channel counts must be positive")
    if args.encoder_layers <= 0 or args.decoder_layers <= 0 or args.window_size <= 0:
        parser.error("layer counts and window size must be positive")
    if args.adapter_init != "identity" and not set(args.adapters) <= {"glu_lora", "rglu_lora"}:
        parser.error("--adapter-init lora_warm requires only glu_lora/rglu_lora")
    return args


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


def _state_metrics(optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    state_bytes = 0
    state_elements = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                state_elements += value.numel()
                state_bytes += value.numel() * value.element_size()
    return state_bytes, state_elements


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_model(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> ImageAE:
    return ImageAE(
        latent_channels=args.latent_channels,
        bottleneck_channels=args.bottleneck_channels,
        encoder_type="window_transformer",
        decoder_type="window_transformer",
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        encoder_window_size=args.window_size,
        downsample_stages=3,
        vae=False,
    ).to(device=device, dtype=dtype)


def _make_initial_state(args: argparse.Namespace, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    model = _make_model(args, torch.device("cpu"), dtype)
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _build_model(
    args: argparse.Namespace,
    adapter: str,
    initial_state: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[ImageAE, int, list[str]]:
    model = _make_model(args, device, dtype)
    model.load_state_dict(
        {key: value.to(device=device, dtype=dtype) for key, value in initial_state.items()},
        strict=True,
    )
    matched = inject_adapter(
        model,
        adapter,
        TARGET_PATTERNS,
        rank=args.rank,
        alpha=args.alpha,
        init_mode=args.adapter_init,
    )
    trainable = mark_only_adapter_trainable(model)
    return model, trainable, matched


def _loaders(args: argparse.Namespace, seed: int):
    transform = transforms.ToTensor()
    try:
        train_dataset = datasets.CIFAR10(
            root=args.data_dir, train=True, download=False, transform=transform,
        )
        validation_dataset = datasets.CIFAR10(
            root=args.data_dir, train=False, download=False, transform=transform,
        )
    except RuntimeError as error:
        raise FileNotFoundError(
            f"CIFAR-10 files were not found below {args.data_dir!r}; "
            "download the dataset first or pass --data-dir",
        ) from error
    train_count = min(args.max_train_samples, len(train_dataset))
    validation_count = min(args.max_validation_samples, len(validation_dataset))
    generator = torch.Generator(device="cpu").manual_seed(seed + 1000)
    train_loader = DataLoader(
        Subset(train_dataset, range(train_count)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, range(validation_count)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, validation_loader


@torch.no_grad()
def _evaluate(
    model: ImageAE,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    model.eval()
    total_loss = 0.0
    total_images = 0
    for images, _ in loader:
        images = images.to(device=device, dtype=dtype)
        reconstruction, _ = model(images)
        loss = (reconstruction.float() - images.float()).square().mean()
        total_loss += loss.item() * images.shape[0]
        total_images += images.shape[0]
    return total_loss / total_images


def _gate_stats(model: nn.Module) -> dict[str, float]:
    for module in iter_adapter_modules(model):
        if module.adapter_type not in {"glu_lora", "rglu_lora"}:
            continue
        with torch.no_grad():
            gate = torch.nn.functional.silu(
                module.lora_B2 @ module.lora_A2,
            ).float()
            if module.adapter_type == "rglu_lora":
                gate = gate + 1.0
            gate = gate.reshape(-1)
            quantiles = torch.quantile(gate, gate.new_tensor([0.01, 0.5, 0.99]))
        return {
            "gate_mean": gate.mean().item(),
            "gate_std": gate.std(unbiased=False).item(),
            "gate_min": gate.min().item(),
            "gate_max": gate.max().item(),
            "gate_p01": quantiles[0].item(),
            "gate_p50": quantiles[1].item(),
            "gate_p99": quantiles[2].item(),
        }
    return {}


def run_case(
    args: argparse.Namespace,
    adapter: str,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    initial_state: dict[str, torch.Tensor],
) -> dict[str, object]:
    torch.manual_seed(seed + 10_000)
    model, trainable, matched = _build_model(
        args, adapter, initial_state, device, dtype,
    )
    train_loader, validation_loader = _loaders(args, seed)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    initial_loss = _evaluate(model, validation_loader, device, dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = perf_counter()
    steps = 0
    initial_gradient_norm = None
    for _ in range(args.epochs):
        model.train()
        for images, _ in train_loader:
            images = images.to(device=device, dtype=dtype)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(images)
            loss = (reconstruction.float() - images.float()).square().mean()
            loss.backward()
            if initial_gradient_norm is None:
                gradients = [
                    parameter.grad.detach().float().reshape(-1)
                    for parameter in parameters if parameter.grad is not None
                ]
                initial_gradient_norm = torch.linalg.vector_norm(
                    torch.cat(gradients),
                ).item()
            optimizer.step()
            steps += 1
    _synchronize(device)
    elapsed = perf_counter() - started
    validation_loss = _evaluate(model, validation_loader, device, dtype)
    model.eval()
    images, _ = next(iter(validation_loader))
    images = images.to(device=device, dtype=dtype)
    with torch.no_grad():
        before_merge, _ = model(images)
        merge_adapter(model)
        after_merge, _ = model(images)
        merge_error = (before_merge - after_merge).abs().max().item()
        unmerge_adapter(model)
    merge_atol, merge_rtol = merge_tolerances(dtype)
    state_bytes, state_elements = _state_metrics(optimizer)
    result: dict[str, object] = {
        "status": "passed",
        "seed": seed,
        "adapter": adapter,
        "rank": args.rank,
        "alpha": args.alpha if args.alpha is not None else args.rank,
        "adapter_init": args.adapter_init,
        "matched_modules": matched,
        "trainable_parameters": trainable,
        "epochs": args.epochs,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(validation_loader.dataset),
        "steps": steps,
        "initial_validation_loss": initial_loss,
        "validation_loss": validation_loss,
        "validation_loss_delta": validation_loss - initial_loss,
        "initial_gradient_norm": initial_gradient_norm,
        "optimizer_state_bytes": state_bytes,
        "optimizer_state_elements": state_elements,
        "seconds_per_step": elapsed / max(steps, 1),
        "merge_max_abs_error": merge_error,
        "merge_atol": merge_atol,
        "merge_rtol": merge_rtol,
        "merge_equivalent": merge_is_equivalent(before_merge, after_merge, dtype),
    }
    result.update(_gate_stats(model))
    if device.type == "cuda":
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    cases: dict[str, dict[str, object]] = {}
    for seed in args.seeds:
        initial_state = _make_initial_state(args, seed, dtype)
        for adapter in args.adapters:
            key = f"seed={seed}/{adapter}"
            cases[key] = run_case(
                args, adapter, seed, device, dtype, initial_state,
            )
    return {
        "status": "passed",
        "script": "verify.image_ae_adapter_dataset_comparison",
        "data_dir": os.path.abspath(args.data_dir),
        "device": str(device),
        "dtype": args.dtype,
        "seeds": list(args.seeds),
        "adapters": list(args.adapters),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "rank": args.rank,
        "alpha": args.alpha if args.alpha is not None else args.rank,
        "adapter_init": args.adapter_init,
        "model": {
            "latent_channels": args.latent_channels,
            "bottleneck_channels": args.bottleneck_channels,
            "encoder_type": "window_transformer",
            "decoder_type": "window_transformer",
            "encoder_layers": args.encoder_layers,
            "decoder_layers": args.decoder_layers,
            "window_size": args.window_size,
            "downsample_stages": 3,
        },
        "loss": "MSE",
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
