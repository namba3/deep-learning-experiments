"""CUDA validation and small benchmark for the MMDiT attention paths.

This intentionally exercises the combinations that are easy to miss when
changing MHLA kernels: BF16 activations, GQA, rectangular resolutions,
batch>1, and an entirely masked text stream.

Example::

    python image_gen/validate_mhla.py --backend triton

The script requires the same environment as ``train.py`` and a CUDA GPU for
the BF16/Triton checks.
"""

import argparse
import time

import torch

try:
    from .train import DiT
except ImportError:
    from train import DiT


def make_model(pattern, backend, recompute_output):
    # Small dimensions keep this validation practical while retaining GQA
    # (8 Q heads / 2 KV heads) and the required 2D-RoPE head dimension.
    model = DiT(
        image_channels=4,
        context_dim=32,
        dim=64,
        depth=1,
        heads=8,
        patch_size=2,
        reference_height=32,
        reference_width=32,
        context_depth=1,
        context_heads=4,
        gradient_checkpointing=False,
        kv_heads=2,
        context_kv_heads=2,
        use_head_gate=True,
        attention_pattern=pattern,
        mhla_latent_blocks=4,
        mhla_image_blocks=2,
        mhla_text_blocks=2,
        mhla_backend=backend,
        mhla_recompute_output=recompute_output,
    )
    return model.to(device="cuda", dtype=torch.bfloat16).train()


def run_case(pattern, backend, batch, height, width, empty_text,
             recompute_output, skip_backward):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = make_model(pattern, backend, recompute_output)
    latent = torch.randn(
        batch, 4, height, width, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )
    text = torch.randn(
        batch, 9, 32, device="cuda", dtype=torch.bfloat16,
    )
    if empty_text:
        text_mask = torch.zeros(batch, 9, device="cuda", dtype=torch.bool)
    else:
        text_mask = torch.ones(batch, 9, device="cuda", dtype=torch.bool)
        text_mask[:, -2:] = False
    timestep = torch.rand(batch, device="cuda", dtype=torch.float32)

    # Compile/autotune overhead must not be included in the steady-state
    # measurement.  Warm up both forward and backward when requested.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        warmup_output = model(latent, timestep, text, text_mask)
        warmup_loss = warmup_output.float().square().mean()
    if not skip_backward:
        warmup_loss.backward()
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    latent.grad = None
    del warmup_output, warmup_loss
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(latent, timestep, text, text_mask)
        loss = output.float().square().mean()
    torch.cuda.synchronize()
    forward_ms = (time.perf_counter() - start) * 1000.0
    if not torch.isfinite(output).all():
        raise AssertionError(f"non-finite {pattern} output")

    backward_ms = None
    if not skip_backward:
        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward_ms = (time.perf_counter() - start) * 1000.0
        finite_grads = [
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ]
        if not all(finite_grads):
            raise AssertionError(f"non-finite {pattern} gradient")

    allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    result = (
        f"pattern={pattern:5s} backend={backend:10s} "
        f"batch={batch} resolution={height}x{width} "
        f"empty_text={empty_text} recompute={recompute_output} "
        f"forward={forward_ms:.1f}ms "
        + (f"backward={backward_ms:.1f}ms " if backward_ms is not None else "")
    )
    print(
        result
        + f"peak_allocated={allocated:.1f}MiB "
        + f"peak_reserved={reserved:.1f}MiB"
    )
    del model, latent, text, text_mask, timestep, output, loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["auto", "vectorized", "triton"], default="triton",
    )
    parser.add_argument(
        "--pattern", choices=["full", "mhla", "both"], default="both",
    )
    parser.add_argument(
        "--recompute-output", action="store_true",
        help="also validate the MHLA backward activation-recompute path",
    )
    parser.add_argument(
        "--skip-backward", action="store_true",
        help="only validate forward outputs",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for MHLA runtime validation")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("This validation requires BF16 support on the CUDA device")

    torch.manual_seed(0)
    patterns = ["full", "mhla"] if args.pattern == "both" else [args.pattern]
    cases = (
        (2, 64, 48, False),  # batch>1, rectangular resolution, partially masked text
        (2, 48, 64, True),   # another rectangular layout, all text masked
    )
    for pattern in patterns:
        for batch, height, width, empty_text in cases:
            run_case(
                pattern, args.backend, batch, height, width, empty_text,
                args.recompute_output, args.skip_backward,
            )
    print("MHLA runtime validation passed")


if __name__ == "__main__":
    main()
