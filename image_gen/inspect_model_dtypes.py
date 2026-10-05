"""Inspect runtime and safetensors dtypes of Qwen3.5 and Qwen-Image VAE.

Examples::

    python image_gen/inspect_model_dtypes.py \
        --vae-model Qwen/Qwen-Image

The runtime report reflects the dtype after ``from_pretrained`` and before any
explicit conversion by this script.  The safetensors report is best-effort:
for Hugging Face model IDs it inspects an already cached snapshot and does not
download weight files unless ``--download-file-weights`` is specified.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch


DEFAULT_VISION_MODEL = "Qwen/Qwen3.5-0.8B"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL)
    parser.add_argument("--vae-model", required=True)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Runtime inspection device. Default: auto.",
    )
    parser.add_argument(
        "--runtime", action=argparse.BooleanOptionalAction, default=True,
        help="Load models and inspect runtime dtypes. Default: enabled.",
    )
    parser.add_argument(
        "--disk", action=argparse.BooleanOptionalAction, default=True,
        help="Inspect cached/local safetensors dtypes. Default: enabled.",
    )
    parser.add_argument(
        "--download-file-weights", action="store_true",
        help="Allow downloading safetensors when the Hugging Face snapshot is not cached.",
    )
    return parser.parse_args()


def dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def format_numel(numel):
    return f"{numel:,}"


def summarize_dtypes(items):
    counts = Counter()
    for dtype, numel in items:
        counts[dtype_name(dtype)] += int(numel)
    return counts


def print_dtype_summary(title, counts):
    total = sum(counts.values())
    print(f"{title}:")
    if not total:
        print("  (no tensors found)")
        return
    for dtype, numel in counts.most_common():
        ratio = 100.0 * numel / total
        print(f"  {dtype}: {format_numel(numel)} elements ({ratio:.2f}%)")
    print(f"  total: {format_numel(total)} elements")


def runtime_module_summary(module, title):
    parameter_items = [
        (parameter.dtype, parameter.numel())
        for parameter in module.parameters()
    ]
    buffer_items = [
        (buffer.dtype, buffer.numel())
        for buffer in module.buffers()
        if buffer.is_floating_point()
    ]
    print_dtype_summary(f"{title} parameters", summarize_dtypes(parameter_items))
    print_dtype_summary(f"{title} floating-point buffers", summarize_dtypes(buffer_items))


def find_visual_module(model):
    visual = getattr(model, "visual", None)
    if visual is not None:
        return visual
    for name, module in model.named_modules():
        if name.endswith("visual"):
            return module
    return None


def resolve_snapshot(model_ref, download_file_weights):
    local_path = Path(model_ref).expanduser()
    if local_path.is_dir():
        return local_path
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(f"disk {model_ref}: huggingface_hub is unavailable")
        return None
    try:
        return Path(snapshot_download(
            model_ref,
            local_files_only=not download_file_weights,
        ))
    except Exception as error:
        mode = "cached snapshot" if not download_file_weights else "snapshot"
        print(f"disk {model_ref}: no {mode} available ({error})")
        return None


def inspect_safetensors(root, title, subdirectory=None):
    search_root = root / subdirectory if subdirectory else root
    if not search_root.is_dir():
        print(f"disk {title}: directory not found: {search_root}")
        return
    files = sorted(search_root.rglob("*.safetensors"))
    if not files:
        print(f"disk {title}: no safetensors files under {search_root}")
        return
    items = []
    file_dtypes = Counter()
    tensor_count = 0
    for path in files:
        try:
            # The safetensors header contains dtype and shape for every tensor;
            # reading it avoids materializing the actual model weights.
            with path.open("rb") as file:
                header_size_bytes = file.read(8)
                if len(header_size_bytes) != 8:
                    raise ValueError("missing 8-byte header length")
                header_size = int.from_bytes(header_size_bytes, "little")
                header = json.loads(file.read(header_size))
            for key, metadata in header.items():
                if key == "__metadata__":
                    continue
                dtype = metadata["dtype"]
                numel = math.prod(metadata["shape"])
                items.append((dtype, numel))
                file_dtypes[dtype_name(dtype)] += 1
                tensor_count += 1
        except Exception as error:
            print(f"disk {title}: failed to inspect {path}: {error}")
    print_dtype_summary(f"{title} safetensors", summarize_dtypes(items))
    print(f"  files: {len(files)}; tensors: {tensor_count}")
    print("  tensor counts by dtype: " + ", ".join(
        f"{dtype}={count}" for dtype, count in file_dtypes.most_common()
    ))


def load_runtime_models(args, device):
    from transformers import AutoModel
    from diffusers import AutoencoderKL

    if "qwen" in args.vae_model.lower():
        from diffusers import AutoencoderKLQwenImage
        vae = AutoencoderKLQwenImage.from_pretrained(
            args.vae_model, subfolder="vae",
        )
    else:
        vae = AutoencoderKL.from_pretrained(args.vae_model, subfolder="vae")
    vision = AutoModel.from_pretrained(args.vision_model)
    vae = vae.to(device).eval()
    vision = vision.to(device).eval()
    return vision, vae


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    print(f"device for runtime inspection: {device}")
    print(f"vision model: {args.vision_model}")
    print(f"vae model: {args.vae_model}")

    if args.disk:
        vision_root = resolve_snapshot(args.vision_model, args.download_file_weights)
        if vision_root is not None:
            inspect_safetensors(vision_root, "Qwen3.5")
        vae_root = resolve_snapshot(args.vae_model, args.download_file_weights)
        if vae_root is not None:
            inspect_safetensors(vae_root, "VAE", subdirectory="vae")

    if args.runtime:
        vision, vae = load_runtime_models(args, device)
        runtime_module_summary(vision, "Qwen3.5")
        visual = find_visual_module(vision)
        if visual is not None:
            runtime_module_summary(visual, "Qwen3.5 visual encoder")
        else:
            print("Qwen3.5 visual encoder: module named 'visual' was not found")
        runtime_module_summary(vae, "VAE")


if __name__ == "__main__":
    main()
