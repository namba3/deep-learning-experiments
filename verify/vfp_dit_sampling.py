"""Run real Qwen/VAE T2I and TI2I sampling with an untrained tiny DiT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vfp_dit_runtime.qwen35 import DEFAULT_QWEN35_MODEL, load_qwen35_encoder  # noqa: E402
from vfp_dit.encoders import (  # noqa: E402
    DEFAULT_VAE_MODEL,
    load_qwen_image_vae,
)
from vfp_dit.generate_samples import generate_one  # noqa: E402
from vfp_dit.model import NoVFCBDiT  # noqa: E402


DEFAULT_PROMPT = "Change the background to a clear blue sky."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-model", default=DEFAULT_QWEN35_MODEL)
    parser.add_argument("--vae-model", default=DEFAULT_VAE_MODEL)
    parser.add_argument("--condition-layer", default="final")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.resolution <= 0 or args.resolution % 16:
        raise ValueError("resolution must be a positive multiple of 16")
    if args.steps <= 0 or args.guidance_scale < 0 or args.threads <= 0:
        raise ValueError("steps/threads must be positive and guidance-scale non-negative")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    vae = load_qwen_image_vae(
        model_id=args.vae_model,
        device=device,
        dtype=torch.float32,
    )
    encoder = load_qwen35_encoder(
        model_id=args.vlm_model,
        device=device,
        dtype=torch.float32,
    )
    model = NoVFCBDiT(
        qwen_dim=encoder.condition_dim,
        latent_channels=16,
        width=24,
        depth=1,
        heads=2,
        kv_heads=1,
        adapter_depth=1,
        ff_mult=1.5,
    ).eval()
    reference = Image.new("RGB", (64, 64), (128, 96, 64))
    outputs = []
    for index, (mode, source) in enumerate((
        ("t2i", None),
        ("ti2i", reference),
    )):
        image = generate_one(
            args.prompt,
            source,
            model=model,
            vae=vae,
            encoder=encoder,
            condition_layer=args.condition_layer,
            resolution=args.resolution,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            device=device,
            generator=torch.Generator(device=device).manual_seed(args.seed + index),
        )
        finite = bool(torch.isfinite(image).all())
        expected_shape = (3, args.resolution, args.resolution)
        outputs.append({
            "mode": mode,
            "image_shape": list(image.shape),
            "finite": finite,
            "in_range": bool((image >= -1.0).all() and (image <= 1.0).all()),
            "ok": image.shape == expected_shape and finite,
        })
    return {
        "device": "cpu",
        "dtype": "float32",
        "vlm_model": args.vlm_model,
        "vae_model": args.vae_model,
        "condition_layer": args.condition_layer,
        "resolution": args.resolution,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "untrained_tiny_model": True,
        "outputs": outputs,
        "ok": all(output["ok"] for output in outputs),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except Exception as error:
        print(
            f"VFP-DiT simple sampling verification failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
