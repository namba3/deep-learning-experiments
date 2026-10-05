"""Compare low-rank adapters on real CIFAR-10 images.

This runtime probe uses the real ``CIFAR10ViT`` and local CIFAR-10 files, but
keeps the model small and the sample count configurable.  It is intended for
matched short experiments before a full ``cifar10.train`` run.

Example::

    python3 -m verify.cifar10_adapter_dataset_comparison \
        --data-dir cifar10/data --seeds 0,1,2 --epochs 2 \
        --max-train-samples 512 --max-validation-samples 256 \
        --output output/cifar10-adapter-dataset-comparison.json
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

from verify.cifar10_adapter_comparison import (  # noqa: E402
    ADAPTERS,
    _build_model,
    _make_initial_state,
    _state_metrics,
    _synchronize,
    resolve_device,
    resolve_dtype,
)
from verify.adapter_metrics import merge_is_equivalent, merge_tolerances  # noqa: E402
from core.low_rank import (  # noqa: E402
    canonicalize_adapter_type,
    iter_adapter_modules,
    merge_adapter,
    unmerge_adapter,
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
            "adapters must be a comma-separated subset of: "
            + ",".join(ADAPTERS),
        )
    return adapters


def _parse_adapter_map(
    value: str,
    value_type,
    option_name: str,
) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"{option_name} entries must use adapter=value format",
            )
        adapter_name, raw_value = (part.strip() for part in item.split("=", 1))
        adapter = canonicalize_adapter_type(adapter_name)
        if adapter not in ADAPTERS:
            raise argparse.ArgumentTypeError(
                f"unknown adapter in {option_name}: {adapter_name}",
            )
        try:
            parsed_value = value_type(raw_value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"{option_name} values must be numeric",
            ) from error
        if parsed_value <= 0:
            raise argparse.ArgumentTypeError(
                f"{option_name} values must be positive",
            )
        result[adapter] = parsed_value
    if not result:
        raise argparse.ArgumentTypeError(f"{option_name} must not be empty")
    return result


def _parse_rank_map(value: str) -> dict[str, int | float]:
    return _parse_adapter_map(value, int, "--rank-map")


def _parse_alpha_map(value: str) -> dict[str, int | float]:
    return _parse_adapter_map(value, float, "--alpha-map")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="cifar10/data")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seeds", type=_parse_csv_ints, default=(0, 1, 2))
    parser.add_argument("--adapters", type=_parse_adapters, default=ADAPTERS)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-validation-samples", type=int, default=256)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument(
        "--rank-map", type=_parse_rank_map, default=None,
        help="Optional per-adapter ranks, e.g. lora=16,loha=8.",
    )
    parser.add_argument(
        "--alpha-map", type=_parse_alpha_map, default=None,
        help="Optional per-adapter alpha values, e.g. lora=16,loha=8.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--adapter-init", choices=("identity", "lora_warm"),
        default="identity",
        help="GLU-LoRA family initialization mode.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_train_samples <= 0 or args.max_validation_samples <= 0:
        parser.error("sample limits must be positive")
    if args.rank <= 0:
        parser.error("--rank must be positive")
    if args.alpha is not None and args.alpha <= 0:
        parser.error("--alpha must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    return args


def _build_loaders(args: argparse.Namespace, seed: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])
    try:
        train_dataset = datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            download=False,
            transform=transform,
        )
        validation_dataset = datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            download=False,
            transform=transform,
        )
    except RuntimeError as error:
        raise FileNotFoundError(
            f"CIFAR-10 files were not found below {args.data_dir!r}; "
            "download the dataset first or pass --data-dir",
        ) from error

    train_count = min(args.max_train_samples, len(train_dataset))
    validation_count = min(args.max_validation_samples, len(validation_dataset))
    train_dataset = Subset(train_dataset, range(train_count))
    validation_dataset = Subset(validation_dataset, range(validation_count))
    generator = torch.Generator(device="cpu").manual_seed(seed + 1000)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, validation_loader


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device=device, dtype=dtype)
        labels = labels.to(device=device)
        logits = model(images)
        total_loss += criterion(logits.float(), labels).item() * labels.numel()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    return total_loss / total, 100.0 * correct / total


def run_case(
    adapter: str,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    initial_state: dict[str, torch.Tensor],
) -> dict[str, object]:
    torch.manual_seed(seed + 10_000)
    model, trainable, matched = _build_model(
        initial_state, adapter, args, device, dtype,
    )
    train_loader, validation_loader = _build_loaders(args, seed)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    criterion = nn.CrossEntropyLoss()
    initial_loss, initial_accuracy = _evaluate(
        model, validation_loader, device, dtype,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    start = perf_counter()
    steps = 0
    for _ in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device=device, dtype=dtype)
            labels = labels.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images).float(), labels)
            loss.backward()
            optimizer.step()
            steps += 1
    _synchronize(device)
    elapsed = perf_counter() - start
    validation_loss, validation_accuracy = _evaluate(
        model, validation_loader, device, dtype,
    )

    model.eval()
    images, _ = next(iter(validation_loader))
    images = images.to(device=device, dtype=dtype)
    with torch.no_grad():
        before_merge = model(images)
        merge_adapter(model)
        after_merge = model(images)
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
        "matched_modules": matched,
        "trainable_parameters": trainable,
        "epochs": args.epochs,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(validation_loader.dataset),
        "steps": steps,
        "initial_validation_loss": initial_loss,
        "initial_validation_accuracy": initial_accuracy,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "validation_loss_delta": validation_loss - initial_loss,
        "optimizer_state_bytes": state_bytes,
        "optimizer_state_elements": state_elements,
        "seconds_per_step": elapsed / max(steps, 1),
        "merge_max_abs_error": merge_error,
        "merge_atol": merge_atol,
        "merge_rtol": merge_rtol,
        "merge_equivalent": merge_is_equivalent(before_merge, after_merge, dtype),
    }
    if device.type == "cuda":
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    for module in iter_adapter_modules(model):
        if module.adapter_type in {"glu_lora", "rglu_lora"}:
            with torch.no_grad():
                gate = torch.nn.functional.silu(
                    module.lora_B2 @ module.lora_A2,
                ).float()
                if module.adapter_type == "rglu_lora":
                    gate = gate + 1.0
                quantiles = torch.quantile(
                    gate.reshape(-1), gate.new_tensor([0.01, 0.5, 0.99]),
                )
                result.update(
                    gate_mean=gate.mean().item(),
                    gate_std=gate.std(unbiased=False).item(),
                    gate_min=gate.min().item(),
                    gate_max=gate.max().item(),
                    gate_p01=quantiles[0].item(),
                    gate_p50=quantiles[1].item(),
                    gate_p99=quantiles[2].item(),
                )
            break
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    cases: dict[str, dict[str, object]] = {}
    for seed in args.seeds:
        initial_state = _make_initial_state(seed, dtype)
        for adapter in args.adapters:
            adapter_args = argparse.Namespace(**vars(args))
            adapter_args.rank = (args.rank_map or {}).get(adapter, args.rank)
            adapter_args.alpha = (args.alpha_map or {}).get(adapter, args.alpha)
            key = f"seed={seed}/{adapter}"
            cases[key] = run_case(
                adapter, adapter_args, seed, device, dtype, initial_state,
            )
    return {
        "status": "passed",
        "script": "verify.cifar10_adapter_dataset_comparison",
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
        "rank_map": args.rank_map,
        "alpha_map": args.alpha_map,
        "adapter_init": args.adapter_init,
        "learning_rate": args.learning_rate,
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
