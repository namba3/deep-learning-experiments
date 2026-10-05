"""Benchmark Full Attention and MHLA under the same MMDiT conditions.

The benchmark warms up every configuration before timing it, so Triton
compilation and autotune startup are excluded from the reported values.

Example::

    python3 -m image_gen.benchmark_mhla \
        --backend triton --patterns full,mhla3-full1 \
        --no-autotune --warmup 2 --repeats 10

The default model dimensions are close to the training model:
``dim=1024, heads=16, kv_heads=8`` (head dimension 64).
"""

import argparse
import gc
import os
import statistics
import time

import torch

DiT = None
SUPPORTED_PATTERNS = ("full", "mhla", "mhla3-full1")


def parse_patterns(value):
    patterns = tuple(part.strip() for part in value.split(",") if part.strip())
    if not patterns or any(pattern not in SUPPORTED_PATTERNS for pattern in patterns):
        raise argparse.ArgumentTypeError(
            "patterns must be a comma-separated subset of: full,mhla,mhla3-full1"
        )
    return patterns


def percentile(values, fraction):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def make_model(args, pattern):
    model = DiT(
        image_channels=args.image_channels,
        context_dim=args.context_dim,
        dim=args.model_dim,
        depth=args.depth,
        heads=args.heads,
        patch_size=2,
        reference_height=args.latent_height,
        reference_width=args.latent_width,
        context_depth=args.context_depth,
        context_heads=args.context_heads,
        gradient_checkpointing=args.gradient_checkpointing,
        kv_heads=args.kv_heads,
        context_kv_heads=args.context_kv_heads,
        use_head_gate=True,
        attention_pattern=pattern,
        mhla_latent_blocks=args.mhla_latent_blocks,
        mhla_image_blocks=args.mhla_image_blocks,
        mhla_text_blocks=args.mhla_text_blocks,
        mhla_backend=args.backend,
        mhla_recompute_output=args.recompute_output,
    )
    return model.to(device="cuda", dtype=args.dtype).train()


def make_inputs(args):
    latent = torch.randn(
        args.batch_size,
        args.image_channels,
        args.latent_height,
        args.latent_width,
        device="cuda",
        dtype=args.dtype,
    )
    text = torch.randn(
        args.batch_size,
        args.text_tokens,
        args.context_dim,
        device="cuda",
        dtype=args.dtype,
    )
    if args.empty_text:
        text_mask = torch.zeros(
            args.batch_size, args.text_tokens,
            device="cuda", dtype=torch.bool,
        )
    else:
        text_mask = torch.ones(
            args.batch_size, args.text_tokens,
            device="cuda", dtype=torch.bool,
        )
        if args.text_tokens >= 2:
            text_mask[:, -2:] = False
    timestep = torch.rand(
        args.batch_size, device="cuda", dtype=torch.float32,
    )
    return latent, timestep, text, text_mask


def run_model(args, pattern):
    torch.cuda.empty_cache()
    gc.collect()
    print(f"[{pattern}] building model", flush=True)
    model = make_model(args, pattern)
    latent, timestep, text, text_mask = make_inputs(args)

    # Compile/autotune and populate allocator caches before measuring.
    print(
        f"[{pattern}] warmup {args.warmup} iteration(s) "
        f"(compile/autotune may take time)",
        flush=True,
    )
    for warmup_index in range(args.warmup):
        print(
            f"[{pattern}] warmup {warmup_index + 1}/{args.warmup}",
            flush=True,
        )
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=args.dtype):
            output = model(latent, timestep, text, text_mask)
            loss = output.float().square().mean()
        if not args.forward_only:
            loss.backward()
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    print(f"[{pattern}] timed repetitions={args.repeats}", flush=True)

    forward_times = []
    backward_times = []
    for _ in range(args.repeats):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=args.dtype):
            output = model(latent, timestep, text, text_mask)
            loss = output.float().square().mean()
        torch.cuda.synchronize()
        forward_times.append((time.perf_counter() - started) * 1000.0)

        if not args.forward_only:
            started = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            backward_times.append((time.perf_counter() - started) * 1000.0)

    allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    result = {
        "pattern": pattern,
        "forward_median": statistics.median(forward_times),
        "forward_p90": percentile(forward_times, 0.90),
        "forward_mean": statistics.mean(forward_times),
        "backward_median": statistics.median(backward_times) if backward_times else None,
        "backward_p90": percentile(backward_times, 0.90) if backward_times else None,
        "backward_mean": statistics.mean(backward_times) if backward_times else None,
        "peak_allocated": allocated,
        "peak_reserved": reserved,
    }
    del model, latent, timestep, text, text_mask, output, loss
    torch.cuda.empty_cache()
    gc.collect()
    return result


def print_result(result):
    backward = "forward-only"
    if result["backward_median"] is not None:
        backward = (
            f"backward median={result['backward_median']:.2f}ms "
            f"p90={result['backward_p90']:.2f}ms"
        )
    print(
        f"pattern={result['pattern']:10s} "
        f"forward median={result['forward_median']:.2f}ms "
        f"p90={result['forward_p90']:.2f}ms "
        f"{backward} "
        f"peak_allocated={result['peak_allocated']:.1f}MiB "
        f"peak_reserved={result['peak_reserved']:.1f}MiB"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patterns", type=parse_patterns, default=("full", "mhla3-full1"),
        help=(
            "Comma-separated patterns to benchmark. Supported: full,mhla,"
            "mhla3-full1. Default: full,mhla3-full1."
        ),
    )
    parser.add_argument(
        "--backend", choices=["auto", "vectorized", "triton"], default="triton",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    autotune_group = parser.add_mutually_exclusive_group()
    autotune_group.add_argument(
        "--autotune", action="store_true",
        help="Enable MHLA Triton autotune (slower startup).",
    )
    autotune_group.add_argument(
        "--no-autotune", action="store_true",
        help="Disable MHLA Triton autotune (the default).",
    )
    parser.add_argument(
        "--triton-backward", action="store_true",
        help="Use the custom Triton MHLA backward (slower startup).",
    )
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--recompute-output", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing", action="store_true",
        help="Enable DiT gradient checkpointing while benchmarking.",
    )
    parser.add_argument(
        "--dtype", choices=["bf16", "fp16"], default="bf16",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-channels", type=int, default=4)
    parser.add_argument("--latent-height", type=int, default=64)
    parser.add_argument("--latent-width", type=int, default=48)
    parser.add_argument("--text-tokens", type=int, default=256)
    parser.add_argument("--model-dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--context-dim", type=int, default=1024)
    parser.add_argument("--context-depth", type=int, default=2)
    parser.add_argument("--context-heads", type=int, default=16)
    parser.add_argument("--context-kv-heads", type=int, default=8)
    parser.add_argument("--mhla-latent-blocks", type=int, default=16)
    parser.add_argument("--mhla-image-blocks", type=int, default=4)
    parser.add_argument("--mhla-text-blocks", type=int, default=4)
    parser.add_argument("--empty-text", action="store_true")
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats must be positive")
    positive_options = (
        "batch_size", "image_channels", "latent_height", "latent_width",
        "text_tokens", "model_dim", "depth", "heads", "kv_heads",
        "context_dim", "context_depth", "context_heads", "context_kv_heads",
        "mhla_latent_blocks", "mhla_image_blocks", "mhla_text_blocks",
    )
    invalid_options = [
        f"--{name.replace('_', '-')} must be positive"
        for name in positive_options
        if getattr(args, name) <= 0
    ]
    if invalid_options:
        parser.error("; ".join(invalid_options))
    if not torch.cuda.is_available():
        parser.error("CUDA is required for this benchmark")
    args.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    if args.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        parser.error("BF16 is not supported on this CUDA device")

    # Benchmark startup should be short and deterministic.  Training keeps
    # the normal autotune default; only this benchmark disables it by default.
    os.environ["MHLA_AUTOTUNE"] = "1" if args.autotune else "0"
    os.environ["MHLA_TRITON_BACKWARD"] = "1" if args.triton_backward else "0"

    # Import after setting MHLA_AUTOTUNE: train.py creates the Triton kernels
    # at import time, so changing the environment afterwards is too late.
    global DiT
    try:
        from .train import DiT as imported_dit
    except ImportError:
        from train import DiT as imported_dit
    DiT = imported_dit

    torch.manual_seed(0)
    print(
        f"benchmark: batch={args.batch_size} latent={args.latent_height}x{args.latent_width} "
        f"dim={args.model_dim} heads={args.heads}/{args.kv_heads} "
        f"context={args.context_dim} heads={args.context_heads}/{args.context_kv_heads} "
        f"depth={args.depth} context_depth={args.context_depth} "
        f"patterns={','.join(args.patterns)} "
        f"dtype={args.dtype} warmup={args.warmup} repeats={args.repeats} "
        f"empty_text={args.empty_text} recompute={args.recompute_output} "
        f"gradient_checkpointing={args.gradient_checkpointing} "
        f"autotune={args.autotune} triton_backward={args.triton_backward}"
    )
    for pattern in args.patterns:
        result = run_model(args, pattern)
        print_result(result)


if __name__ == "__main__":
    main()
