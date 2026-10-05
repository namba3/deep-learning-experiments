"""Run one real-data CPU training step through the VFCB-free DiT stack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vfp_dit_runtime.hf_data import collate_hf_image_examples, load_hf_image_sources  # noqa: E402
from vfp_dit_runtime.qwen35 import (  # noqa: E402
    DEFAULT_QWEN35_MODEL,
    load_qwen35_encoder,
    parse_condition_layer,
)
from vfp_dit.encoders import (  # noqa: E402
    DEFAULT_VAE_MODEL,
    load_qwen_image_vae,
)
from vfp_dit.model import NoVFCBDiT  # noqa: E402
from vfp_dit.train import _flow_step  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multi-edit-data-root", default="data/MultiEdit")
    parser.add_argument("--coco-split", default="val")
    parser.add_argument("--edit-split", default="train")
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--vlm-model", default=DEFAULT_QWEN35_MODEL)
    parser.add_argument("--vae-model", default=DEFAULT_VAE_MODEL)
    parser.add_argument("--condition-layer", type=parse_condition_layer, default="final")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.resolution <= 0 or args.resolution % 16:
        raise ValueError("resolution must be a positive multiple of 16")
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    coco, edits = load_hf_image_sources(
        multi_edit_data_root=args.multi_edit_data_root,
        edit_split=args.edit_split,
        coco_split=args.coco_split,
        cache_dir=args.hf_cache_dir,
    )
    batch = collate_hf_image_examples([coco[0], edits[0]])
    vae = load_qwen_image_vae(
        model_id=args.vae_model,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    encoder = load_qwen35_encoder(
        model_id=args.vlm_model,
        device=torch.device("cpu"),
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
    )
    step_args = argparse.Namespace(
        condition_layer=args.condition_layer,
        resolution=args.resolution,
        vae_latent_mode="mode",
        condition_dropout=0.0,
    )
    loss, metrics = _flow_step(
        model,
        batch,
        step_args,
        torch.device("cpu"),
        vae=vae,
        encoder=encoder,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("flow-matching loss is non-finite")
    loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    finite_gradients = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in trainable
    )
    if not finite_gradients:
        raise FloatingPointError("a trainable model parameter has no finite gradient")
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    optimizer.step()
    return {
        "device": "cpu",
        "dtype": "float32",
        "vlm_model": args.vlm_model,
        "vae_model": args.vae_model,
        "condition_layer": str(args.condition_layer),
        "resolution": args.resolution,
        "sources": batch["sources"],
        "conditioning_types": batch["conditioning_types"],
        "loss": float(loss.detach()),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "trainable_parameters": len(trainable),
        "finite_gradients": finite_gradients,
        "optimizer_step": "ok",
        "ok": True,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except Exception as error:
        print(
            f"VFP-DiT simple training-step verification failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
