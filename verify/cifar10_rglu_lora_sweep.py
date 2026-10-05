"""Sweep Residual GLU-LoRA rank and alpha on real CIFAR-10 subsets."""

from __future__ import annotations

import argparse
import json

from verify.cifar10_adapter_comparison import (
    _make_initial_state,
    resolve_device,
    resolve_dtype,
)
from verify.cifar10_adapter_dataset_comparison import _parse_csv_ints, run_case


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive")
    return values


def _parse_positive_floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="cifar10/data")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seeds", type=_parse_csv_ints, default=(0, 1, 2))
    parser.add_argument("--ranks", type=_parse_positive_ints, default=(1, 2, 4, 8))
    parser.add_argument("--alphas", type=_parse_positive_floats, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int, default=256)
    parser.add_argument("--max-validation-samples", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--adapter-init", choices=("identity", "lora_warm"), default="identity",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    if args.max_train_samples <= 0 or args.max_validation_samples <= 0:
        parser.error("sample limits must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.alphas is not None and len(args.alphas) == 0:
        parser.error("--alphas must not be empty")
    return args


def run(args: argparse.Namespace) -> dict[str, object]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    alpha_values = args.alphas
    cases: dict[str, dict[str, object]] = {}
    for seed in args.seeds:
        initial_state = _make_initial_state(seed, dtype)
        for rank in args.ranks:
            for alpha in (alpha_values or (float(rank),)):
                case_args = argparse.Namespace(
                    data_dir=args.data_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    max_train_samples=args.max_train_samples,
                    max_validation_samples=args.max_validation_samples,
                    rank=rank,
                    alpha=alpha,
                    learning_rate=args.learning_rate,
                    adapter_init=args.adapter_init,
                )
                key = f"seed={seed}/rank={rank}/alpha={alpha:g}"
                cases[key] = run_case(
                    "rglu_lora",
                    case_args,
                    seed,
                    device,
                    dtype,
                    initial_state,
                )
    return {
        "status": "passed",
        "script": "verify.cifar10_rglu_lora_sweep",
        "device": str(device),
        "dtype": args.dtype,
        "seeds": list(args.seeds),
        "ranks": list(args.ranks),
        "alphas": list(alpha_values) if alpha_values is not None else "rank",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "learning_rate": args.learning_rate,
        "adapter_init": args.adapter_init,
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
