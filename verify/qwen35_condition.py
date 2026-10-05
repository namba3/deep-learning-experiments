"""Verify Qwen3.5 condition taps against a full multimodal forward.

The external-model probe is separate from pytest. It defaults to CPU so it can
run alongside GPU training without competing for VRAM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from PIL import Image
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vfp_dit_runtime.qwen35 import (  # noqa: E402
    DEFAULT_QWEN35_MODEL,
    load_qwen35_encoder,
    parse_condition_layer,
)

DEFAULT_PROMPT = "Change the background to a plain blue sky."


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(value)


def resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {"fp32": torch.float32, "bf16": torch.bfloat16}[value]


def tap_name(tap: int | str) -> str:
    return str(tap)


def make_inputs(encoder, prompt: str, image: Image.Image) -> dict[str, Any]:
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Source image:"},
            {"type": "image", "image": image.convert("RGB")},
            {"type": "text", "text": "Instruction: " + prompt},
        ],
    }]
    inputs = encoder.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {
        key: value.to(encoder.device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def run_tap_probe(encoder, *, tap: int | str, prompt: str, image: Image.Image,
                  inputs: dict[str, Any]) -> dict[str, Any]:
    base = getattr(encoder.model, "model", encoder.model)
    text_model = getattr(base, "language_model", None)
    layers = getattr(text_model, "layers", None)
    if text_model is None or layers is None or len(layers) != encoder.layer_count:
        raise RuntimeError("Could not locate the expected Qwen3.5 language-model blocks")

    is_final = tap == "final"
    selected = text_model.norm if is_final else layers[tap - 1]
    observed_calls = [0] * len(layers)
    observer_handles = [
        layer.register_forward_hook(
            lambda _module, _inputs, _output, i=i: observed_calls.__setitem__(
                i, observed_calls[i] + 1,
            )
        )
        for i, layer in enumerate(layers)
    ]
    norm_calls = [0]
    norm_handle = (
        text_model.norm.register_forward_hook(
            lambda _module, _inputs, _output: norm_calls.__setitem__(0, norm_calls[0] + 1)
        )
        if is_final else None
    )
    lm_head = getattr(encoder.model, "lm_head", None)
    lm_head_calls = [0]
    lm_head_handle = (
        lm_head.register_forward_hook(
            lambda _module, _inputs, _output: lm_head_calls.__setitem__(
                0, lm_head_calls[0] + 1,
            )
        )
        if is_final and isinstance(lm_head, nn.Module) else None
    )
    try:
        started = perf_counter()
        hidden, mask = encoder.encode_condition(
            prompt, source_image=image, condition_layer=tap,
        )
        elapsed = perf_counter() - started
    finally:
        for handle in observer_handles:
            handle.remove()
        if norm_handle is not None:
            norm_handle.remove()
        if lm_head_handle is not None:
            lm_head_handle.remove()

    if is_final:
        tap_call_ok = norm_calls[0] == 1
        decoder_execution_ok = all(count == 1 for count in observed_calls)
        later_blocks_skipped = None
        lm_head_skipped = (
            lm_head_calls[0] == 0 if isinstance(lm_head, nn.Module) else None
        )
    else:
        tap_call_ok = observed_calls[tap - 1] == 1
        decoder_execution_ok = (
            all(count == 1 for count in observed_calls[:tap])
            and all(count == 0 for count in observed_calls[tap:])
        )
        later_blocks_skipped = all(count == 0 for count in observed_calls[tap:])
        lm_head_skipped = None

    reference_outputs: list[torch.Tensor] = []
    reference_handle = selected.register_forward_hook(
        lambda _module, _inputs, output: reference_outputs.append(output)
    )
    try:
        with torch.inference_mode():
            encoder.model(
                **inputs,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
    finally:
        reference_handle.remove()

    if len(reference_outputs) != 1 or not isinstance(reference_outputs[0], torch.Tensor):
        raise RuntimeError(
            f"Expected one tensor from full-forward tap {tap_name(tap)}, "
            f"got {len(reference_outputs)} outputs"
        )
    reference = reference_outputs[0][0].float()
    same_shape = tuple(hidden.shape) == tuple(reference.shape)
    exact_match = same_shape and torch.equal(hidden, reference)
    max_abs_error = (
        float((hidden - reference).abs().max().item()) if same_shape else None
    )
    attention_mask = inputs.get("attention_mask")
    mask_matches = (
        attention_mask is not None
        and torch.equal(mask, attention_mask[0].to(device=mask.device, dtype=torch.bool))
    )
    finite = bool(torch.isfinite(hidden).all())
    ok = (
        exact_match and mask_matches and finite and tap_call_ok
        and decoder_execution_ok and lm_head_skipped is not False
    )
    return {
        "tap": tap_name(tap),
        "early_shape": list(hidden.shape),
        "full_forward_shape": list(reference.shape),
        "mask_shape": list(mask.shape),
        "mask_matches_processor": mask_matches,
        "finite": finite,
        "exact_match": exact_match,
        "max_abs_error": max_abs_error,
        "selected_module_called_once": tap_call_ok,
        "decoder_execution_contract_ok": decoder_execution_ok,
        "later_decoder_blocks_skipped": later_blocks_skipped,
        "lm_head_skipped": lm_head_skipped,
        "early_exit_seconds": elapsed,
        "ok": ok,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_QWEN35_MODEL)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--dtype", choices=("auto", "fp32", "bf16"), default="auto")
    parser.add_argument(
        "--tap", action="append", type=parse_condition_layer, default=None,
        help="Tap to compare; repeat as needed. Default: 6 and final.",
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if not args.prompt.strip():
        raise ValueError("--prompt must contain non-whitespace text")
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    taps = args.tap or [6, "final"]
    if len({tap_name(tap) for tap in taps}) != len(taps):
        raise ValueError("--tap values must be unique")

    encoder = load_qwen35_encoder(model_id=args.model_id, device=device, dtype=dtype)
    image = Image.new("RGB", (args.image_size, args.image_size), (128, 96, 64))
    inputs = make_inputs(encoder, args.prompt, image)
    results = [
        run_tap_probe(
            encoder, tap=tap, prompt=args.prompt, image=image, inputs=inputs,
        )
        for tap in taps
    ]
    return {
        "model_id": args.model_id,
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "layer_count": encoder.layer_count,
        "condition_dim": encoder.condition_dim,
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
            f"Qwen3.5 condition verification failed: {type(error).__name__}: {error}",
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
