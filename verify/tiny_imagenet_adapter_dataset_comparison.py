"""Compare low-rank adapters on the Hugging Face TinyImageNet-200 dataset.

The default dataset is ``zh-plus/tiny-imagenet``.  ``datasets`` resolves it
from the normal Hugging Face cache, so the comparison does not depend on a
manually unpacked TinyImageNet directory.

This is a runtime probe, not a full training entrypoint.  It keeps the
classifier small while using the native 64x64 images and all 200 classes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from time import perf_counter

from datasets import load_dataset
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.low_rank import (  # noqa: E402
    iter_adapter_modules,
    merge_adapter,
    unmerge_adapter,
)
from verify.adapter_metrics import (  # noqa: E402
    merge_is_equivalent,
    merge_tolerances,
)
from verify.cifar10_adapter_comparison import (  # noqa: E402
    ADAPTERS,
    _build_model,
    _make_initial_state,
    _state_metrics,
    _synchronize,
    resolve_device,
    resolve_dtype,
)
from verify.cifar10_adapter_dataset_comparison import (  # noqa: E402
    _parse_adapters,
    _parse_alpha_map,
    _parse_csv_ints,
    _parse_rank_map,
)


NUM_CLASSES = 200
IMAGE_SIZE = 64
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class HFDatasetImageClassification(Dataset):
    """Apply a torchvision transform to images from a HF dataset split."""

    def __init__(
        self,
        dataset,
        transform,
    ) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        row = self.dataset[index]
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        return self.transform(image), int(row["label"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="zh-plus/tiny-imagenet")
    parser.add_argument(
        "--cache-dir", default=None,
        help="Optional Hugging Face datasets cache directory; omit to use the default.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seeds", type=_parse_csv_ints, default=(0, 1, 2))
    parser.add_argument("--adapters", type=_parse_adapters, default=ADAPTERS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--max-validation-samples", type=int, default=512)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--rank-map", type=_parse_rank_map, default=None)
    parser.add_argument("--alpha-map", type=_parse_alpha_map, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--train-classifier-head", action="store_true",
        help="Also train the common classifier head; useful with a random base model.",
    )
    parser.add_argument(
        "--adapter-init", choices=("identity", "lora_warm"), default="identity",
        help="GLU-LoRA family initialization mode.",
    )
    parser.add_argument("--output", default=None)
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
    if args.adapter_init != "identity" and not set(args.adapters) <= {
        "glu_lora", "rglu_lora",
    }:
        parser.error("--adapter-init lora_warm requires only glu_lora/rglu_lora")
    return args


def _enable_classifier_head(model: nn.Module) -> int:
    """Enable the common classification head while keeping the backbone frozen."""
    trainable = 0
    for name, parameter in model.named_parameters():
        if name.startswith("head."):
            parameter.requires_grad_(True)
            trainable += parameter.numel()
    if trainable == 0:
        raise ValueError("model does not contain a classifier head")
    return trainable


def _build_loaders(args: argparse.Namespace, seed: int):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize(72),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    load_kwargs = {}
    if args.cache_dir is not None:
        load_kwargs["cache_dir"] = args.cache_dir
    dataset = load_dataset(args.dataset_name, **load_kwargs)
    for split_name in ("train", "valid"):
        if split_name not in dataset:
            raise ValueError(
                f"{args.dataset_name!r} must provide train and valid splits; "
                f"missing {split_name!r}",
            )
    train_split = dataset["train"]
    validation_split = dataset["valid"]
    label_feature = train_split.features.get("label")
    if label_feature is None or getattr(label_feature, "num_classes", None) != NUM_CLASSES:
        raise ValueError(
            f"expected a label feature with {NUM_CLASSES} classes in "
            f"{args.dataset_name!r}",
        )
    train_dataset = HFDatasetImageClassification(
        train_split, train_transform,
    )
    validation_dataset = HFDatasetImageClassification(
        validation_split, validation_transform,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 1000)
    train_count = min(args.max_train_samples, len(train_dataset))
    validation_count = min(args.max_validation_samples, len(validation_dataset))
    train_indices = torch.randperm(len(train_dataset), generator=generator)[:train_count]
    validation_indices = torch.randperm(
        len(validation_dataset), generator=generator,
    )[:validation_count]
    train_dataset = Subset(train_dataset, train_indices.tolist())
    validation_dataset = Subset(validation_dataset, validation_indices.tolist())
    loader_generator = torch.Generator(device="cpu").manual_seed(seed + 2000)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        generator=loader_generator, num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
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
        img_size=IMAGE_SIZE, num_classes=NUM_CLASSES,
    )
    if args.train_classifier_head:
        trainable += _enable_classifier_head(model)
    train_loader, validation_loader = _build_loaders(args, seed)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=0.0,
    )
    criterion = nn.CrossEntropyLoss()
    initial_loss, initial_accuracy = _evaluate(
        model, validation_loader, device, dtype,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    train_start = perf_counter()
    steps = 0
    epoch_metrics: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = perf_counter()
        batch_losses: list[float] = []
        for images, labels in train_loader:
            images = images.to(device=device, dtype=dtype)
            labels = labels.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images).float(), labels)
            loss.backward()
            optimizer.step()
            steps += 1
            batch_losses.append(loss.item())
        _synchronize(device)
        epoch_train_seconds = perf_counter() - epoch_start
        validation_loss, validation_accuracy = _evaluate(
            model, validation_loader, device, dtype,
        )
        train_loss = sum(batch_losses) / len(batch_losses)
        train_loss_std = math.sqrt(
            sum((value - train_loss) ** 2 for value in batch_losses)
            / len(batch_losses),
        )
        epoch_metrics.append({
            "epoch": epoch,
            "steps": steps,
            "train_loss": train_loss,
            "train_loss_std": train_loss_std,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "validation_loss_delta": validation_loss - initial_loss,
            "epoch_train_seconds": epoch_train_seconds,
            "epoch_seconds_per_step": epoch_train_seconds / len(batch_losses),
        })
    _synchronize(device)
    train_elapsed = perf_counter() - train_start

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
        "epoch_metrics": epoch_metrics,
        "optimizer_state_bytes": state_bytes,
        "optimizer_state_elements": state_elements,
        "seconds_per_step": train_elapsed / max(steps, 1),
        "merge_max_abs_error": merge_error,
        "merge_atol": merge_atol,
        "merge_rtol": merge_rtol,
        "merge_equivalent": merge_is_equivalent(before_merge, after_merge, dtype),
    }
    if device.type == "cuda":
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    for module in iter_adapter_modules(model):
        if module.adapter_type not in {"glu_lora", "rglu_lora"}:
            continue
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
        initial_state = _make_initial_state(
            seed, dtype, img_size=IMAGE_SIZE, num_classes=NUM_CLASSES,
        )
        for adapter in args.adapters:
            adapter_args = argparse.Namespace(**vars(args))
            adapter_args.rank = (args.rank_map or {}).get(adapter, args.rank)
            adapter_args.alpha = (args.alpha_map or {}).get(adapter, args.alpha)
            cases[f"seed={seed}/{adapter}"] = run_case(
                adapter, adapter_args, seed, device, dtype, initial_state,
            )
    return {
        "status": "passed",
        "script": "verify.tiny_imagenet_adapter_dataset_comparison",
        "dataset_name": args.dataset_name,
        "cache_dir": args.cache_dir,
        "dataset": "tiny-imagenet-200",
        "num_classes": NUM_CLASSES,
        "image_size": IMAGE_SIZE,
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
        "train_classifier_head": args.train_classifier_head,
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
