"""Compare text LM decoder architectures on a common synthetic token batch.

The benchmark reports deduplicated parameter count, forward/backward timing,
training-step timing, and CUDA peak allocator usage when CUDA is available.
On CPU, CUDA VRAM is reported as ``null`` and process RSS is reported as an
approximate fallback.  The short training comparison uses synthetic tokens;
its loss is not a language-model quality evaluation.

Example::

    PYTHONPATH=. python3 -m benchmarks.benchmark_text_lm_architectures \
        --device cpu --dim 256 --num-layers 4 --tokens 128 \
        --warmup 2 --repeats 5 --train-steps 10 \
        --json benchmark_results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from text_lm.train import (
    ARCHITECTURE_CHOICES,
    TinyTextLM,
)
from core.utils import convert_linear_to_bf16


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    return torch.device(value)


def resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "fp32":
        return torch.float32
    if value == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 is not supported by the CUDA device")
        return torch.bfloat16
    if value == "fp16":
        if device.type == "cpu":
            raise ValueError("fp16 benchmark requires CUDA")
        return torch.float16
    raise ValueError(f"unknown dtype: {value}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rss_mib() -> float:
    # Linux reports KiB; macOS reports bytes.  The environment is Linux, but
    # keeping the branch makes the fallback less surprising elsewhere.
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        value /= 1024.0
    else:
        value *= 1024.0
    return value / (1024.0 ** 2)


def parse_seeds(value: str | None, fallback: int) -> tuple[int, ...]:
    if value is None:
        if fallback < 0:
            raise ValueError("--seed must be non-negative")
        return (fallback,)
    seeds = tuple(
        int(part.strip()) for part in value.split(",") if part.strip()
    )
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("--seeds must contain one or more non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must not contain duplicates")
    return seeds


def summarize_results(results: list[dict], value_keys: tuple[str, ...]):
    grouped: dict[str, list[dict]] = {}
    for result in results:
        grouped.setdefault(result["architecture"], []).append(result)
    summary = []
    for architecture, entries in grouped.items():
        item = {
            "architecture": architecture,
            "seeds": [entry["seed"] for entry in entries],
        }
        for key in value_keys:
            raw_values = [entry[key] for entry in entries]
            if all(value is None for value in raw_values):
                item[f"{key}_mean"] = None
                item[f"{key}_std"] = None
                continue
            values = [float(value) for value in raw_values]
            item[f"{key}_mean"] = statistics.mean(values)
            item[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        for key in (
            "parameters", "trainable_parameters", "decoder_layers",
            "decoder_physical_blocks", "loop_count",
        ):
            if key in entries[0]:
                item[key] = entries[0][key]
        summary.append(item)
    return summary


def build_model(
    args,
    architecture: str,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    torch.manual_seed(seed)
    model = TinyTextLM(
        vocab_size=args.vocab_size,
        max_seq_len=args.tokens,
        embed_dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        kv_heads=args.kv_heads,
        condition_dim=args.condition_dim,
        transform_rank=args.transform_rank,
        architecture=architecture,
        looped_blocks=args.looped_blocks,
        looped_prefix_layers=args.looped_prefix_layers,
        looped_repeats=args.looped_repeats,
        looped_suffix_layers=args.looped_suffix_layers,
        mhla_looped_prefix_cycles=args.mhla_looped_prefix_cycles,
        mhla_looped_repeats=args.mhla_looped_repeats,
        mhla_looped_suffix_cycles=args.mhla_looped_suffix_cycles,
    ).to(device)
    if args.dtype == "bf16":
        skip_modules = ()
        depth_embedding = getattr(model.decoder, "depth_embedding", None)
        if depth_embedding is not None:
            skip_modules = (depth_embedding,)
        convert_linear_to_bf16(model.decoder, skip_modules=skip_modules)
    elif args.dtype == "fp16":
        model = model.to(dtype=dtype)
    return model


def loss_for(model, input_ids, attention_mask):
    logits = model(input_ids, attention_mask)
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.size(-1)),
        input_ids[:, 1:].contiguous().view(-1),
    )


def measure_step(model, optimizer, input_ids, attention_mask, device):
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    started = time.perf_counter()
    logits = model(input_ids, attention_mask)
    synchronize(device)
    forward_ms = (time.perf_counter() - started) * 1000.0
    loss = F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.size(-1)),
        input_ids[:, 1:].contiguous().view(-1),
    )
    synchronize(device)
    started = time.perf_counter()
    loss.backward()
    optimizer.step()
    synchronize(device)
    backward_and_update_ms = (time.perf_counter() - started) * 1000.0
    return forward_ms, backward_and_update_ms, float(loss.item())


def benchmark_architecture(args, architecture, device, dtype, batches, seed):
    model = build_model(args, architecture, device, dtype, seed)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    input_ids, attention_mask = batches[0]

    model.train()
    for batch_index in range(args.warmup):
        measure_step(
            model,
            optimizer,
            batches[batch_index % len(batches)][0],
            batches[batch_index % len(batches)][1],
            device,
        )
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    forward_times = []
    update_times = []
    losses = []
    for batch_index in range(args.repeats):
        forward_ms, update_ms, loss = measure_step(
            model,
            optimizer,
            batches[(args.warmup + batch_index) % len(batches)][0],
            batches[(args.warmup + batch_index) % len(batches)][1],
            device,
        )
        forward_times.append(forward_ms)
        update_times.append(update_ms)
        losses.append(loss)

    physical_blocks = len(getattr(model.decoder, "layers", ()))
    if physical_blocks == 0:
        physical_blocks = sum(
            len(getattr(model.decoder, name, ()))
            for name in ("prefix_layers", "looped_layers", "suffix_layers")
        )
    result = {
        "architecture": architecture,
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "parameter_storage_fp32_mib": parameter_count * 4 / (1024.0 ** 2),
        "parameter_storage_bf16_mib": parameter_count * 2 / (1024.0 ** 2),
        "decoder_layers": int(getattr(model.decoder, "num_layers", args.num_layers)),
        "decoder_physical_blocks": int(physical_blocks),
        "loop_count": int(getattr(model.decoder, "num_loops", 1)),
        "forward_median_ms": statistics.median(forward_times),
        "backward_update_median_ms": statistics.median(update_times),
        "step_median_ms": statistics.median(
            forward + update
            for forward, update in zip(forward_times, update_times, strict=True)
        ),
        "measurement_loss_last": losses[-1],
        "cuda_peak_allocated_mib": None,
        "cuda_peak_reserved_mib": None,
        "cpu_maxrss_mib": rss_mib() if device.type == "cpu" else None,
    }
    if device.type == "cuda":
        result["cuda_peak_allocated_mib"] = (
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        )
        result["cuda_peak_reserved_mib"] = (
            torch.cuda.max_memory_reserved(device) / (1024.0 ** 2)
        )
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def short_train(args, architecture, device, dtype, batches, seed):
    model = build_model(args, architecture, device, dtype, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    model.train()
    started = time.perf_counter()
    losses = []
    for step in range(args.train_steps):
        _, _, loss = measure_step(
            model,
            optimizer,
            batches[step % len(batches)][0],
            batches[step % len(batches)][1],
            device,
        )
        losses.append(loss)
    synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "architecture": architecture,
        "steps": args.train_steps,
        "total_seconds": elapsed,
        "mean_step_seconds": elapsed / max(args.train_steps, 1),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
    }
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--architectures", default=",".join(ARCHITECTURE_CHOICES))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--looped-blocks", type=int, default=1)
    parser.add_argument("--looped-prefix-layers", type=int, default=1)
    parser.add_argument("--looped-repeats", type=int, default=2)
    parser.add_argument("--looped-suffix-layers", type=int, default=1)
    parser.add_argument("--mhla-looped-prefix-cycles", type=int, default=0)
    parser.add_argument("--mhla-looped-repeats", type=int, default=1)
    parser.add_argument("--mhla-looped-suffix-cycles", type=int, default=0)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--condition-dim", type=int, default=16)
    parser.add_argument("--transform-rank", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--train-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds for repeated measurements; defaults to --seed.",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    try:
        seeds = parse_seeds(args.seeds, args.seed)
    except ValueError as error:
        parser.error(str(error))

    architectures = tuple(
        part.strip() for part in args.architectures.split(",") if part.strip()
    )
    if not architectures or any(
        architecture not in ARCHITECTURE_CHOICES
        for architecture in architectures
    ):
        parser.error(
            "--architectures must contain only: "
            + ",".join(ARCHITECTURE_CHOICES)
        )
    for name in (
        "batch_size", "tokens", "vocab_size", "dim", "num_layers",
        "num_heads", "kv_heads", "condition_dim", "transform_rank",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_heads % args.kv_heads != 0:
        parser.error("--kv-heads must divide --num-heads")
    if not 1 <= args.looped_blocks <= args.num_layers:
        parser.error("--looped-blocks must be in [1, --num-layers]")
    if args.num_layers % args.looped_blocks != 0:
        parser.error("--num-layers must be divisible by --looped-blocks")
    if args.looped_prefix_layers < 0 or args.looped_suffix_layers < 0:
        parser.error("looped prefix/suffix layers must be >= 0")
    if args.looped_repeats <= 0:
        parser.error("--looped-repeats must be positive")
    if "looped-hybrid" in architectures:
        expected_layers = (
            args.looped_prefix_layers
            + args.looped_blocks * args.looped_repeats
            + args.looped_suffix_layers
        )
        if expected_layers != args.num_layers:
            parser.error(
                "--num-layers must equal prefix + looped-blocks * repeats + suffix"
            )
    if any(
        architecture == "mhla3-gqa-looped-hybrid"
        for architecture in architectures
    ):
        if (
            args.mhla_looped_prefix_cycles < 0
            or args.mhla_looped_suffix_cycles < 0
            or args.mhla_looped_repeats <= 0
        ):
            parser.error("MHLA looped prefix/suffix must be >= 0 and repeats > 0")
        expected_layers = 4 * (
            args.mhla_looped_prefix_cycles
            + args.mhla_looped_repeats
            + args.mhla_looped_suffix_cycles
        )
        if expected_layers != args.num_layers:
            parser.error(
                "--num-layers must equal 4 * "
                "(mhla prefix-cycles + repeats + suffix-cycles)"
            )
    if args.warmup < 0 or args.repeats <= 0 or args.train_steps <= 0:
        parser.error("--warmup must be >= 0; repeats and train-steps must be > 0")
    if args.threads <= 0:
        parser.error("--threads must be positive")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    torch.set_num_threads(args.threads)
    print(
        f"device={device} dtype={dtype} batch={args.batch_size} tokens={args.tokens} "
        f"vocab={args.vocab_size} dim={args.dim} num_layers={args.num_layers} "
        f"heads={args.num_heads} kv_heads={args.kv_heads} warmup={args.warmup} "
        f"repeats={args.repeats} train_steps={args.train_steps} "
        f"seeds={','.join(str(seed) for seed in seeds)} threads={args.threads}",
        flush=True,
    )
    benchmark_results = []
    training_results = []
    for seed in seeds:
        torch.manual_seed(seed)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        batches = []
        for _ in range(max(args.warmup + args.repeats, args.train_steps)):
            input_ids = torch.randint(
                args.vocab_size,
                (args.batch_size, args.tokens),
                generator=generator,
                device=device,
            )
            batches.append((input_ids, torch.ones_like(input_ids, dtype=torch.bool)))

        for architecture in architectures:
            result = benchmark_architecture(
                args, architecture, device, dtype, batches, seed,
            )
            training = short_train(
                args, architecture, device, dtype, batches, seed,
            )
            result["seed"] = seed
            training["seed"] = seed
            benchmark_results.append(result)
            training_results.append(training)
            print(
                f"seed={seed} {architecture:15s} params={result['parameters']:,} "
                f"layers={result['decoder_layers']} "
                f"forward={result['forward_median_ms']:.3f}ms "
                f"backward+update={result['backward_update_median_ms']:.3f}ms "
                f"step={result['step_median_ms']:.3f}ms "
                f"short_train={training['mean_step_seconds'] * 1000.0:.3f}ms/step",
                flush=True,
            )

    benchmark_summary = summarize_results(
        benchmark_results,
        (
            "forward_median_ms", "backward_update_median_ms", "step_median_ms",
            "measurement_loss_last", "cuda_peak_allocated_mib",
            "cuda_peak_reserved_mib", "cpu_maxrss_mib",
        ),
    )
    training_summary = summarize_results(
        training_results,
        ("total_seconds", "mean_step_seconds", "loss_first", "loss_last"),
    )
    for item in training_summary:
        print(
            f"summary {item['architecture']:15s} "
            f"step={item['mean_step_seconds_mean'] * 1000.0:.3f}"
            f"+/-{item['mean_step_seconds_std'] * 1000.0:.3f}ms "
            f"loss_last={item['loss_last_mean']:.6f}"
            f"+/-{item['loss_last_std']:.6f}",
            flush=True,
        )

    config = vars(args).copy()
    if config["json"] is not None:
        config["json"] = str(config["json"])
    output = {
        "config": config,
        "device": str(device),
        "dtype": str(dtype),
        "seeds": list(seeds),
        "benchmark": benchmark_results,
        "benchmark_summary": benchmark_summary,
        "short_training": training_results,
        "short_training_summary": training_summary,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
