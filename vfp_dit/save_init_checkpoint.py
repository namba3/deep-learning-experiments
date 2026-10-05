"""Save a seed-controlled, no-metadata initialization for matched DiT ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .encoders import LATENT_CHANNELS
from .model import NoVFCBDiT


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--condition-dim", type=int, default=1024)
    parser.add_argument("--model-width", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--adapter-depth", type=int, default=2)
    parser.add_argument("--ff-mult", type=float, default=3.0)
    parser.add_argument(
        "--reference-latent-downsample-factor", "--latent-downsample-factor",
        dest="latent_downsample_factor", type=int, choices=(1,), default=1,
    )
    parser.add_argument(
        "--target-latent-downsample-factor", type=int, choices=(1, 2, 4), default=2,
    )
    parser.add_argument("--output-refinement-depth", type=int, default=2)
    parser.add_argument("--no-fuse-reference-latent-to-vision", action="store_true")
    parser.add_argument(
        "--fuse-same-input-projections", action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    model = NoVFCBDiT(
        qwen_dim=args.condition_dim,
        latent_channels=LATENT_CHANNELS,
        width=args.model_width,
        depth=args.depth,
        heads=args.heads,
        kv_heads=args.kv_heads,
        adapter_depth=args.adapter_depth,
        ff_mult=args.ff_mult,
        latent_downsample_factor=args.latent_downsample_factor,
        target_latent_downsample_factor=args.target_latent_downsample_factor,
        output_refinement_depth=args.output_refinement_depth,
        output_skip_fusion_mode="add",
        fuse_reference_latent_to_vision=not args.no_fuse_reference_latent_to_vision,
        reference_latent_fusion_mode=(
            "legacy_latent_to_qwen" if args.no_fuse_reference_latent_to_vision
            else "qwen_to_half_latent_add"
        ),
        fuse_same_input_projections=args.fuse_same_input_projections,
        metadata_conditioning="none",
    )
    config = {
        **vars(args),
        "output": str(args.output),
        "latent_channels": LATENT_CHANNELS,
        "target_latent_downsample_factor": args.target_latent_downsample_factor,
        "output_refinement_depth": args.output_refinement_depth,
        "output_skip_fusion_mode": "add",
        "output_head_ada_scale": False,
        "adapter_type": "ffn",
        "fuse_reference_latent_to_vision": not args.no_fuse_reference_latent_to_vision,
        "reference_latent_fusion_mode": (
            "legacy_latent_to_qwen" if args.no_fuse_reference_latent_to_vision
            else "qwen_to_half_latent_add"
        ),
        "metadata_conditioning": "none",
        "metadata_scale_mapping": "linear",
        "metadata_shift": False,
        "attention_head_gate": "input_silu",
        "metadata_ffn_residual_gate": False,
        "metadata_ffn_gate_mapping": "linear",
        "fuse_same_input_projections": args.fuse_same_input_projections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    weights = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    save_file(weights, str(args.output), metadata={
        "vfp_dit.checkpoint": json.dumps({
            "network_version": 1,
            "stage": "vfp_dit.train",
            "stage_label": "no-metadata initialization",
        }, sort_keys=True),
        "vfp_dit.config": json.dumps(config, sort_keys=True),
    })
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"saved initialization checkpoint: {args.output}")
    print(f"seed={args.seed} parameters={parameter_count:,} tensors={len(weights)}")


if __name__ == "__main__":
    main()
