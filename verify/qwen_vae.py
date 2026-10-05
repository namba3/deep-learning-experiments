"""Probe a real diffusers VAE outside the regular unit-test suite.

This command is intentionally separate from pytest.  It loads an external VAE,
may use a local model cache or network access, and reports runtime-specific
shape, dtype, timing, and device behavior.

Example::

    python3 -m verify.qwen_vae \
        --vae-model Qwen/Qwen-Image \
        --vae-dtype fp32
"""

import argparse
import json
import os
import sys
from time import perf_counter

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from image_gen.inference import (  # noqa: E402
    decode_images,
    encode_images,
    resolve_vae_dtype,
    resolve_vae_latent_scale,
    validate_bucket_shapes_with_vae,
)
from image_gen.data import make_bucket_shapes  # noqa: E402


DEFAULT_PROBE_SIZES = ("8x8", "16x16", "32x40")
DEFAULT_ROUNDTRIP_SIZES = ("16x16", "32x40")


def parse_size(value):
    """Parse a positive ``HEIGHTxWIDTH`` argument into an ``(H, W)`` tuple."""
    try:
        height_text, width_text = value.lower().split("x", 1)
        height, width = int(height_text), int(width_text)
    except (AttributeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"expected HEIGHTxWIDTH, got {value!r}"
        ) from error
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError(
            f"height and width must be positive, got {value!r}"
        )
    return height, width


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vae-model", required=True,
        help="Hugging Face model id or local directory containing the vae subfolder.",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="Execution device. Default: CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--vae-dtype", choices=("auto", "bf16", "fp32"), default="auto",
        help="VAE parameter dtype. Default: BF16 on CUDA and FP32 on CPU.",
    )
    parser.add_argument(
        "--latent-scale", type=float, default=None,
        help="Override the VAE config scaling_factor; omitted means use the config or 1.0.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--bucket-step", type=int, default=32)
    parser.add_argument(
        "--probe-size", type=parse_size, action="append", default=None,
        metavar="HEIGHTxWIDTH",
        help="Encode probe size; repeat to add cases. Default: 8x8, 16x16, 32x40.",
    )
    parser.add_argument(
        "--roundtrip-size", type=parse_size, action="append", default=None,
        metavar="HEIGHTxWIDTH",
        help="Encode/decode size; repeat to add cases. Default: 16x16, 32x40.",
    )
    parser.add_argument(
        "--skip-roundtrip", action="store_true",
        help="Only run encode and bucket-stride probes.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(requested)


def load_vae(model_name, device, dtype):
    """Load the VAE using the same Qwen/non-Qwen selection as image_gen."""
    from diffusers import AutoencoderKL

    if "qwen" in model_name.lower():
        from diffusers import AutoencoderKLQwenImage

        vae_class = AutoencoderKLQwenImage
    else:
        vae_class = AutoencoderKL
    return vae_class.from_pretrained(
        model_name, subfolder="vae", torch_dtype=dtype,
    ).to(device).eval()


def tensor_shape(value):
    return [int(dimension) for dimension in value.shape]


def encode_probe(vae, size, latent_scale):
    height, width = size
    parameter = next(vae.parameters())
    images = torch.zeros(
        1, 3, height, width, device=parameter.device, dtype=parameter.dtype,
    )
    started = perf_counter()
    with torch.inference_mode():
        latents = encode_images(vae, images, latent_scale)
    elapsed = perf_counter() - started
    latent_height, latent_width = (int(value) for value in latents.shape[-2:])
    stride = None
    if latent_height and latent_width and height % latent_height == 0 and width % latent_width == 0:
        stride = [height // latent_height, width // latent_width]
    return {
        "input_shape": tensor_shape(images),
        "latent_shape": tensor_shape(latents),
        "stride": stride,
        "finite": bool(torch.isfinite(latents).all()),
        "seconds": elapsed,
    }


def roundtrip_probe(vae, size, latent_scale, seed):
    height, width = size
    parameter = next(vae.parameters())
    generator = torch.Generator(device=parameter.device)
    generator.manual_seed(seed)
    images = torch.randn(
        1, 3, height, width, device=parameter.device,
        dtype=parameter.dtype, generator=generator,
    ).clamp(-1.0, 1.0)
    started = perf_counter()
    with torch.inference_mode():
        latents = encode_images(vae, images, latent_scale)
        decoded = decode_images(vae, latents, latent_scale)
    elapsed = perf_counter() - started
    return {
        "input_shape": tensor_shape(images),
        "latent_shape": tensor_shape(latents),
        "decoded_shape": tensor_shape(decoded),
        "finite": bool(torch.isfinite(latents).all() and torch.isfinite(decoded).all()),
        "decoded_min": float(decoded.amin()),
        "decoded_max": float(decoded.amax()),
        "seconds": elapsed,
    }


def run(args):
    if args.image_size <= 0 or args.bucket_step <= 0:
        raise ValueError("--image-size and --bucket-step must be positive")
    device = resolve_device(args.device)
    requested_dtype = "bf16" if args.vae_dtype == "auto" and device.type == "cuda" else "fp32"
    if args.vae_dtype != "auto":
        requested_dtype = args.vae_dtype
    vae_dtype = resolve_vae_dtype(device, requested_dtype)
    vae = load_vae(args.vae_model, device, vae_dtype)
    latent_scale = resolve_vae_latent_scale(vae, args.latent_scale)

    result = {
        "vae_model": args.vae_model,
        "vae_class": vae.__class__.__name__,
        "device": str(device),
        "dtype": str(vae_dtype).replace("torch.", ""),
        "latent_scale": latent_scale,
        "image_size": args.image_size,
        "bucket_step": args.bucket_step,
    }

    probe_sizes = args.probe_size or [parse_size(value) for value in DEFAULT_PROBE_SIZES]
    probes = []
    for size in probe_sizes:
        try:
            probes.append({"size": list(size), "ok": True, **encode_probe(vae, size, latent_scale)})
        except Exception as error:  # Report boundary failures without hiding them.
            probes.append({
                "size": list(size), "ok": False,
                "error_type": type(error).__name__, "error": str(error),
            })
    result["encode_probes"] = probes

    bucket_shapes = make_bucket_shapes(args.image_size, args.bucket_step)
    bucket_result = {"shapes": [list(shape) for shape in bucket_shapes]}
    try:
        bucket_result["ok"] = True
        bucket_result["stride"] = list(
            validate_bucket_shapes_with_vae(vae, bucket_shapes, latent_scale)
        )
    except Exception as error:
        bucket_result.update({
            "ok": False, "error_type": type(error).__name__, "error": str(error),
        })
    result["buckets"] = bucket_result

    if not args.skip_roundtrip:
        roundtrip_sizes = args.roundtrip_size or [
            parse_size(value) for value in DEFAULT_ROUNDTRIP_SIZES
        ]
        roundtrips = []
        for index, size in enumerate(roundtrip_sizes):
            try:
                roundtrips.append({
                    "size": list(size), "ok": True,
                    **roundtrip_probe(vae, size, latent_scale, args.seed + index),
                })
            except Exception as error:
                roundtrips.append({
                    "size": list(size), "ok": False,
                    "error_type": type(error).__name__, "error": str(error),
                })
        result["roundtrips"] = roundtrips

    result["ok"] = (
        all(probe["ok"] for probe in result["encode_probes"])
        and result["buckets"]["ok"]
        and all(roundtrip["ok"] for roundtrip in result.get("roundtrips", ()))
    )
    return result


def main(argv=None):
    args = parse_args(argv)
    try:
        result = run(args)
    except Exception as error:
        print(f"verification failed before probes: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
