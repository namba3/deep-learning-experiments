"""Compare low-rank adapters on a fixed CIFAR-10 ViT-shaped probe.

The probe uses the real ``CIFAR10ViT`` module but fixed random images and
labels.  It is intended to compare adapter parameterization and short-term
optimization behavior before launching a full dataset run.

Example::

    python3 -m verify.cifar10_adapter_comparison \
        --device cpu --steps 10 --output output/cifar10-adapters.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from time import perf_counter

import torch
from torch import nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cifar10.train import CIFAR10ViT, init_weights  # noqa: E402
from verify.adapter_metrics import merge_is_equivalent, merge_tolerances  # noqa: E402
from core.low_rank import (  # noqa: E402
    canonicalize_adapter_type,
    inject_adapter,
    iter_adapter_modules,
    mark_only_adapter_trainable,
    merge_adapter,
    unmerge_adapter,
)


ADAPTERS = ("lora", "loha", "dora", "glu_lora", "rglu_lora")
TARGET_PATTERNS = (
    r"\.attention\.(qkv|output)$",
    r"^pooling\.attn\.(q_proj|kv_proj|out_proj)$",
)


def _parse_adapters(value: str) -> tuple[str, ...]:
    adapters = tuple(
        canonicalize_adapter_type(part.strip())
        for part in value.split(",")
        if part.strip()
    )
    if not adapters or any(adapter not in ADAPTERS for adapter in adapters):
        raise argparse.ArgumentTypeError(
            "adapters must be a comma-separated subset of: "
            + ",".join(ADAPTERS)
        )
    return adapters


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--adapter-init", choices=("identity", "lora_warm"),
        default="identity",
        help="GLU-LoRA family initialization mode.",
    )
    parser.add_argument(
        "--adapters", type=_parse_adapters, default=ADAPTERS,
        help="Comma-separated adapter names. Default: all.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional JSON output path.",
    )
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.rank <= 0:
        parser.error("--rank must be positive")
    if args.alpha is not None and args.alpha <= 0:
        parser.error("--alpha must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
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


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _state_metrics(optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    state_bytes = 0
    state_elements = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                state_elements += value.numel()
                state_bytes += value.numel() * value.element_size()
    return state_bytes, state_elements


def _make_initial_state(
    seed: int,
    dtype: torch.dtype,
    *,
    img_size: int = 32,
    num_classes: int = 10,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    model = CIFAR10ViT(
        patch_size=2,
        embed_dim=32,
        num_layers=1,
        num_heads=4,
        attention_type="full",
        window_size=4,
        img_size=img_size,
        num_classes=num_classes,
    )
    model.apply(init_weights)
    return {
        key: value.detach().clone().to(dtype=dtype)
        for key, value in model.state_dict().items()
    }


def _make_probe_data(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    images = torch.randn(
        args.batch_size, 3, 32, 32, generator=generator, dtype=dtype,
    )
    labels = torch.randint(0, 10, (args.batch_size,), generator=generator)
    return images.to(device), labels.to(device)


def _build_model(
    initial_state: dict[str, torch.Tensor],
    adapter: str,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    *,
    img_size: int = 32,
    num_classes: int = 10,
) -> tuple[nn.Module, int, list[str]]:
    model = CIFAR10ViT(
        patch_size=2,
        embed_dim=32,
        num_layers=1,
        num_heads=4,
        compute_dtype=dtype,
        attention_type="full",
        window_size=4,
        img_size=img_size,
        num_classes=num_classes,
    ).to(device=device, dtype=dtype)
    model.load_state_dict(
        {key: value.to(device=device, dtype=dtype) for key, value in initial_state.items()}
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


def run_case(
    adapter: str,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    initial_state: dict[str, torch.Tensor],
    images: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, object]:
    model, trainable, matched = _build_model(
        initial_state, adapter, args, device, dtype,
    )
    model.eval()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        initial_loss = criterion(model(images), labels).item()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    start = perf_counter()
    initial_gradient_norm = None
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        if step == 0:
            initial_gradient_norm = torch.linalg.vector_norm(
                torch.cat([
                    parameter.grad.detach().float().reshape(-1)
                    for parameter in parameters
                    if parameter.grad is not None
                ])
            ).item()
        optimizer.step()
    _synchronize(device)
    elapsed = perf_counter() - start

    with torch.no_grad():
        final_loss = criterion(model(images), labels).item()
        before_merge = model(images)
        merge_adapter(model)
        after_merge = model(images)
        merge_error = (before_merge - after_merge).abs().max().item()
        unmerge_adapter(model)
    merge_atol, merge_rtol = merge_tolerances(dtype)

    state_bytes, state_elements = _state_metrics(optimizer)
    result: dict[str, object] = {
        "status": "passed",
        "adapter": adapter,
        "rank": args.rank,
        "alpha": args.alpha if args.alpha is not None else args.rank,
        "matched_modules": matched,
        "trainable_parameters": trainable,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_delta": final_loss - initial_loss,
        "initial_gradient_norm": initial_gradient_norm,
        "optimizer_state_bytes": state_bytes,
        "optimizer_state_elements": state_elements,
        "seconds_per_step": elapsed / args.steps,
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
    initial_state = _make_initial_state(args.seed, dtype)
    images, labels = _make_probe_data(args, device, dtype)
    cases = {
        adapter: run_case(
            adapter, args, device, dtype, initial_state, images, labels,
        )
        for adapter in args.adapters
    }
    return {
        "status": "passed",
        "script": "verify.cifar10_adapter_comparison",
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "rank": args.rank,
        "alpha": args.alpha if args.alpha is not None else args.rank,
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
