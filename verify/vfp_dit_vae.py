"""Verify the original Qwen-Image VAE path used by vfp_dit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vfp_dit.encoders import (  # noqa: E402
    DEFAULT_VAE_MODEL,
    decode_qwen_image_latents,
    encode_qwen_image_latents,
    load_qwen_image_vae,
)


def parse_size(value: str) -> tuple[int, int]:
    try:
        height, width = (int(part) for part in value.lower().split("x", 1))
    except (AttributeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"expected HEIGHTxWIDTH, got {value!r}") from error
    if height <= 0 or width <= 0 or height % 8 or width % 8:
        raise argparse.ArgumentTypeError("height and width must be positive multiples of 8")
    return height, width


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-model", default=DEFAULT_VAE_MODEL)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--size", type=parse_size, action="append", default=None,
        help="Image size as HEIGHTxWIDTH; repeat to add cases. Default: 16x16 and 32x40.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = torch.float32 if args.dtype == "fp32" or device.type == "cpu" else torch.bfloat16
    vae = load_qwen_image_vae(model_id=args.vae_model, device=device, dtype=dtype)
    sizes = args.size or [(16, 16), (32, 40)]
    results = []
    for height, width in sizes:
        case: dict[str, Any] = {"size": [height, width]}
        try:
            images = torch.zeros((1, 3, height, width), device=device, dtype=dtype)
            started = perf_counter()
            latents = encode_qwen_image_latents(vae, images)
            decoded = decode_qwen_image_latents(vae, latents)
            case.update({
                "latent_shape": list(latents.shape),
                "decoded_shape": list(decoded.shape),
                "latent_stride": [height // latents.shape[-2], width // latents.shape[-1]],
                "finite": bool(torch.isfinite(latents).all() and torch.isfinite(decoded).all()),
                "seconds": perf_counter() - started,
            })
            case["ok"] = (
                latents.ndim == 4
                and latents.shape[:2] == (1, 16)
                and decoded.shape == images.shape
                and case["finite"]
            )
        except Exception as error:
            case.update({"ok": False, "error_type": type(error).__name__, "error": str(error)})
        results.append(case)
    return {
        "vae_model": args.vae_model,
        "vae_class": vae.__class__.__name__,
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "latent_channels": int(getattr(vae.config, "z_dim", -1)),
        "results": results,
        "ok": all(result["ok"] for result in results),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except Exception as error:
        print(
            f"VFP-DiT simple VAE verification failed: {type(error).__name__}: {error}",
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
