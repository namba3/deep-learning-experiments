"""Verify real Qwen3.5 T2I/TI2I conditions for the VFCB-free DiT."""

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

from vfp_dit_runtime.qwen35 import (  # noqa: E402
    DEFAULT_QWEN35_MODEL,
    load_qwen35_encoder,
    parse_condition_layer,
)
from vfp_dit.model import build_qwen_condition_positions  # noqa: E402

DEFAULT_PROMPT = "Change the background to a plain blue sky."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_QWEN35_MODEL)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--tap", type=parse_condition_layer, default="final")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def _processor_inputs(encoder, prompt: str, source: Image.Image | None) -> dict[str, Any]:
    content = []
    if source is not None:
        content.extend((
            {"type": "text", "text": "Source image:"},
            {"type": "image", "image": source.convert("RGB")},
        ))
    content.append({"type": "text", "text": "Instruction: " + prompt})
    return encoder.processor.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def _probe_condition(encoder, *, name: str, prompt: str,
                     condition_layer: int | str,
                     source: Image.Image | None) -> dict[str, Any]:
    hidden, mask, positions = encoder.encode_condition(
        prompt,
        source_image=source,
        condition_layer=condition_layer,
        include_positions=True,
    )
    inputs = _processor_inputs(encoder, prompt, source)
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")
    if input_ids is None:
        raise ValueError("processor omitted input_ids")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    modality_types = inputs.get("mm_token_type_ids")
    if modality_types is None:
        image_token_id = getattr(encoder.model.config, "image_token_id", None)
        if image_token_id is None:
            raise ValueError("processor omitted modality types and model image_token_id")
        modality_types = (input_ids == int(image_token_id)).to(dtype=torch.int32)
    vision_config = getattr(encoder.model.config, "vision_config", None)
    merge_size = int(getattr(vision_config, "spatial_merge_size", 0))
    expected = build_qwen_condition_positions(
        modality_types,
        attention_mask.to(dtype=torch.bool),
        inputs.get("image_grid_thw"),
        spatial_merge_size=merge_size,
    )[0].to(device=positions.device)
    mask_matches_processor = torch.equal(
        mask,
        attention_mask[0].to(device=mask.device, dtype=torch.bool),
    )
    positions_match_processor = torch.equal(positions, expected)
    image_token_count = int(((modality_types == 1) & attention_mask.bool()).sum().item())
    has_expected_image_mode = (image_token_count == 0) if source is None else image_token_count > 0
    finite = bool(torch.isfinite(hidden).all() and torch.isfinite(positions).all())
    ok = (
        hidden.ndim == 2
        and hidden.shape == (mask.numel(), encoder.condition_dim)
        and mask.dtype == torch.bool
        and mask.any().item()
        and positions.shape == (mask.numel(), 3)
        and mask_matches_processor
        and positions_match_processor
        and has_expected_image_mode
        and finite
    )
    return {
        "mode": name,
        "hidden_shape": list(hidden.shape),
        "mask_shape": list(mask.shape),
        "position_shape": list(positions.shape),
        "image_token_count": image_token_count,
        "mask_matches_processor": mask_matches_processor,
        "positions_match_processor": positions_match_processor,
        "finite": finite,
        "ok": bool(ok),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.image_size <= 0 or not args.prompt.strip():
        raise ValueError("image-size must be positive and prompt must be non-empty")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = torch.float32 if args.dtype == "fp32" or device.type == "cpu" else torch.bfloat16
    encoder = load_qwen35_encoder(
        model_id=args.model_id,
        device=device,
        dtype=dtype,
    )
    source = Image.new("RGB", (args.image_size, args.image_size), (128, 96, 64))
    results = [
        _probe_condition(
            encoder, name="t2i", prompt=args.prompt,
            condition_layer=args.tap, source=None,
        ),
        _probe_condition(
            encoder, name="ti2i", prompt=args.prompt,
            condition_layer=args.tap, source=source,
        ),
    ]
    return {
        "model_id": args.model_id,
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "tap": str(args.tap),
        "source_image_shape": [3, args.image_size, args.image_size],
        "results": results,
        "ok": all(result["ok"] for result in results),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except Exception as error:
        print(
            f"VFP-DiT simple condition verification failed: {type(error).__name__}: {error}",
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
