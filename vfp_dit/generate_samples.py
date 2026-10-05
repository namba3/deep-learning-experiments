"""Generate T2I or single-reference TI2I samples from a checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from safetensors.torch import load_file
from safetensors import safe_open

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.device import resolve_device  # noqa: E402
from vfp_dit_runtime.qwen35 import (  # noqa: E402
    load_qwen35_encoder,
    parse_condition_layer,
    validate_condition_layer,
)
from .encoders import (  # noqa: E402
    LATENT_DOWNSAMPLE_FACTOR,
    decode_qwen_image_latents,
    encode_qwen_image_latents,
    load_qwen_image_vae,
)
from .model import ConditionKVCache, NoVFCBDiT, grid_positions  # noqa: E402
from .samplers import SCHEDULERS, SOLVERS, sample_flow_matching  # noqa: E402
from runtime.profiling import component_timer  # noqa: E402
from flow_sampling import apply_guidance, available_guidance_methods  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", action="append", default=None,
                        help="Prompt; repeat for multiple samples.")
    parser.add_argument("--prompt-file", help="UTF-8 file with one prompt per line.")
    parser.add_argument("--reference-image", action="append", default=None,
                        help="Optional source image per prompt; repeat once for every prompt.")
    parser.add_argument("--output", default="vfp_dit/generated_samples.png")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--solver", "--sampler", dest="solver",
        choices=SOLVERS, default="euler",
        help=(
            "Update rule (default: euler). Non-default rf_* solvers are "
            "experimental; see flow_sampling/README.md for validation limits."
        ),
    )
    parser.add_argument(
        "--scheduler", choices=SCHEDULERS,
        default="flow_match_euler",
    )
    parser.add_argument("--flow-shift", type=float, default=1.0)
    parser.add_argument("--er-sde-sigma-max", type=float, default=80.0)
    parser.add_argument("--rf-er-sde-eta", type=float, default=0.2)
    parser.add_argument("--rf-er-sde-cutoff", type=float, default=0.9)
    parser.add_argument("--rf-2m-warp-type", choices=("identity", "rational"), default="identity")
    parser.add_argument("--rf-2m-warp-shift", type=float, default=1.0)
    parser.add_argument(
        "--rf-er-sde-warp-type", choices=("identity", "rational"), default="identity",
    )
    parser.add_argument("--rf-er-sde-warp-shift", type=float, default=1.0)
    parser.add_argument("--rf-er-sde-warp-trust-lambda", type=float, default=4.0)
    parser.add_argument("--rf-er-sde-warp-trust-error-c", type=float, default=0.1)
    parser.add_argument("--rf-er-sde-warp-trust-epsilon-abs", type=float, default=1e-8)
    parser.add_argument("--rf-trust-lambda", type=float, default=4.0,
                        help="Trust gate strength for rf_trust_region")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--guidance-method", choices=available_guidance_methods(), default="cfg",
        help="Guidance transform applied to conditional/unconditional model outputs",
    )
    parser.add_argument("--resolution", type=int, default=None,
                        help="Nominal square-equivalent output resolution; defaults to the checkpoint.")
    parser.add_argument(
        "--aspect-ratio", type=_parse_aspect_ratio, default=1.0,
        help="Output W:H aspect ratio, such as 16:9 or 2:1.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "fp32", "bf16"), default="auto")
    parser.add_argument("--vlm-model", default=None, help="Override checkpoint Qwen model.")
    parser.add_argument("--vae-model", default=None, help="Override checkpoint VAE model.")
    parser.add_argument("--condition-layer", default=None,
                        help="Override checkpoint Qwen hidden-state tap.")
    parser.add_argument("--profile-components", action="store_true",
                        help="Print per-sample Qwen/VAE/DiT component timings.")
    return parser.parse_args(argv)


def _parse_aspect_ratio(value: str) -> float:
    try:
        if ":" in value:
            numerator, denominator = value.split(":", 1)
            ratio = float(numerator) / float(denominator)
        else:
            ratio = float(value)
    except (ValueError, ZeroDivisionError) as error:
        raise argparse.ArgumentTypeError("aspect-ratio must be a positive number or W:H") from error
    if not np.isfinite(ratio) or ratio <= 0:
        raise argparse.ArgumentTypeError("aspect-ratio must be finite and positive")
    return ratio


def build_sampling_metadata(
    *,
    solver: str,
    scheduler: str,
    flow_shift: float,
    steps: int,
    guidance_scale: float,
    seed: int,
    num_samples: int,
    er_sde_sigma_max: float,
    output: str,
    checkpoint: str,
    prompts: list[str],
    reference_images: list[str | None],
    resolution: int,
    output_size: tuple[int, int] | None = None,
    device: str,
    dtype: str,
    rf_er_sde_eta: float = 0.2,
    rf_er_sde_cutoff: float = 0.9,
    rf_2m_warp_type: str = "identity",
    rf_2m_warp_shift: float = 1.0,
    rf_er_sde_warp_type: str = "identity",
    rf_er_sde_warp_shift: float = 1.0,
    rf_er_sde_warp_trust_lambda: float = 4.0,
    rf_er_sde_warp_trust_error_c: float = 0.1,
    rf_er_sde_warp_trust_epsilon: float = 1e-8,
    rf_trust_lambda: float = 4.0,
    guidance_method: str = "cfg",
) -> dict:
    """Build reproducible solver metadata for a generated image grid."""
    callback_evaluations = {
        "euler": steps,
        "fireflow": steps + 1,
        "abm2": steps + 1,
        "er_sde": steps,
        "rf_ab2": steps,
        "rf_2m_warp": steps,
        "rf_trust_region": steps,
        "rf_er_sde_1": steps,
        "rf_er_sde_2m": steps,
        "rf_er_sde_warp_1": steps,
        "rf_er_sde_warp_2m": steps,
        "rf_er_sde_warp_trust": steps,
    }[solver]
    return {
        "solver": solver,
        "solver_version": (
            "0.1.1" if solver.startswith("rf_er_sde_warp") else None
        ),
        "scheduler": scheduler,
        "solver_grid": (
            "uniform_tau"
            if solver in {
                "rf_2m_warp", "rf_er_sde_warp_1", "rf_er_sde_warp_2m",
                "rf_er_sde_warp_trust",
            }
            else None
        ),
        "flow_shift": flow_shift,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "guidance_method": guidance_method,
        "seed": seed,
        "sample_seeds": [seed + index for index in range(num_samples)],
        "num_samples": num_samples,
        "callback_evaluations_per_sample": callback_evaluations,
        "callback_evaluations_total": callback_evaluations * num_samples,
        "cfg_network_forward_multiplier": 2 if guidance_scale != 1.0 else 1,
        "er_sde_sigma_max": er_sde_sigma_max if solver == "er_sde" else None,
        "rf_er_sde_eta": rf_er_sde_eta if solver.startswith("rf_er_sde") else None,
        "rf_er_sde_cutoff": rf_er_sde_cutoff if solver.startswith("rf_er_sde") else None,
        "rf_2m_warp_type": rf_2m_warp_type if solver == "rf_2m_warp" else None,
        "rf_2m_warp_shift": rf_2m_warp_shift if solver == "rf_2m_warp" else None,
        "rf_er_sde_warp_type": (
            rf_er_sde_warp_type if solver.startswith("rf_er_sde_warp") else None
        ),
        "rf_er_sde_warp_shift": (
            rf_er_sde_warp_shift if solver.startswith("rf_er_sde_warp") else None
        ),
        "rf_er_sde_warp_q_min": (
            1e-3 if solver.startswith("rf_er_sde_warp") else None
        ),
        "rf_er_sde_warp_q_max": (
            1e3 if solver.startswith("rf_er_sde_warp") else None
        ),
        "rf_er_sde_warp_trust_lambda": (
            rf_er_sde_warp_trust_lambda
            if solver == "rf_er_sde_warp_trust" else None
        ),
        "rf_er_sde_warp_trust_error_c": (
            rf_er_sde_warp_trust_error_c
            if solver == "rf_er_sde_warp_trust" else None
        ),
        "rf_er_sde_warp_trust_epsilon_abs": (
            rf_er_sde_warp_trust_epsilon
            if solver == "rf_er_sde_warp_trust" else None
        ),
        "rf_er_sde_warp_sde_gate": (
            "trust_error" if solver == "rf_er_sde_warp_trust" else None
        ),
        "rf_er_sde_warp_startup": (
            "deterministic_first_two_intervals"
            if solver == "rf_er_sde_warp_trust" else None
        ),
        "rf_er_sde_time_gate": "sin_pi" if solver.startswith("rf_er_sde") else None,
        "rf_trust_lambda": rf_trust_lambda if solver == "rf_trust_region" else None,
        "output": output,
        "checkpoint": checkpoint,
        "prompts": prompts,
        "reference_images": reference_images,
        "resolution": resolution,
        "output_size": list(output_size or (resolution, resolution)),
        "device": device,
        "dtype": dtype,
    }


def read_checkpoint(path: Path) -> tuple[dict, dict]:
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
    try:
        network = json.loads(metadata["vfp_dit.checkpoint"])
        config = json.loads(metadata["vfp_dit.config"])
    except (KeyError, json.JSONDecodeError) as error:
        raise ValueError("Checkpoint is missing VFP-DiT metadata") from error
    if network.get("network_version") != 1 or network.get("stage") not in {
        "vfp_dit.train", "vfp_dit_simple.train",
    }:
        raise ValueError("Checkpoint is not a supported VFP-DiT trainer checkpoint")
    return network, config


def build_model(config: dict, device: torch.device) -> NoVFCBDiT:
    adapter_type = config.get("adapter_type")
    if adapter_type != "ffn":
        raise ValueError(
            "This sampler supports only GatedFFN VLM adapter checkpoints; "
            f"checkpoint adapter_type={adapter_type!r} is retired."
        )
    model = NoVFCBDiT(
        qwen_dim=int(config["condition_dim"]),
        latent_channels=int(config["latent_channels"]),
        latent_downsample_factor=int(config.get("latent_downsample_factor", 1)),
        target_latent_downsample_factor=int(config.get(
            "target_latent_downsample_factor",
            config.get("latent_downsample_factor", 1),
        )),
        output_refinement_depth=int(config.get("output_refinement_depth", 0)),
        output_refinement_conditioning=str(
            config.get("output_refinement_conditioning", "none")
        ),
        output_skip_fusion_mode=str(
            config.get("output_skip_fusion_mode", "concat_linear")
        ),
        output_head_ada_scale=bool(config.get("output_head_ada_scale", False)),
        fuse_reference_latent_to_vision=bool(
            config.get("fuse_reference_latent_to_vision", False)
        ),
        # Older checkpoints fused a downsampled latent into the Qwen grid.
        reference_latent_fusion_mode=str(config.get(
            "reference_latent_fusion_mode", "legacy_latent_to_qwen",
        )),
        width=int(config["model_width"]),
        depth=int(config["depth"]),
        heads=int(config["heads"]),
        kv_heads=int(config["kv_heads"]),
        adapter_depth=int(config["adapter_depth"]),
        metadata_conditioning=str(config.get("metadata_conditioning", "none")),
        metadata_scale_mapping=str(config.get("metadata_scale_mapping", "linear")),
        metadata_ffn_gate_mapping=str(config.get("metadata_ffn_gate_mapping", "linear")),
        metadata_shift=bool(config.get("metadata_shift", False)),
        attention_head_gate=str(config.get("attention_head_gate", "timestep_sigmoid")),
        metadata_ffn_residual_gate=bool(
            config.get("metadata_ffn_residual_gate", False)
        ),
        fuse_same_input_projections=bool(
            config.get("fuse_same_input_projections", False)
        ),
        ff_mult=float(config["ff_mult"]),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
    ).to(device).eval()
    return model


def fit_image(image: Image.Image, resolution: int | tuple[int, int]) -> Image.Image:
    height, width = (resolution, resolution) if isinstance(resolution, int) else resolution
    return ImageOps.fit(image.convert("RGB"), (width, height),
                        method=Image.Resampling.LANCZOS)


def image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


def _autocast(device: torch.device, dtype: str):
    if dtype == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast(device.type, enabled=False)


@torch.inference_mode()
def sample_latents(
    model: NoVFCBDiT,
    condition_cache: ConditionKVCache,
    *,
    latent_height: int,
    latent_width: int,
    steps: int,
    device: torch.device,
    guidance_scale: float = 1.0,
    guidance_method: str = "cfg",
    unconditional_cache: ConditionKVCache | None = None,
    generator: torch.Generator | None = None,
    solver: str = "euler",
    scheduler: str = "flow_match_euler",
    flow_shift: float = 1.0,
    er_sde_sigma_max: float = 80.0,
    rf_er_sde_eta: float = 0.2,
    rf_er_sde_cutoff: float = 0.9,
    rf_2m_warp_type: str = "identity",
    rf_2m_warp_shift: float = 1.0,
    rf_er_sde_warp_type: str = "identity",
    rf_er_sde_warp_shift: float = 1.0,
    rf_er_sde_warp_trust_lambda: float = 4.0,
    rf_er_sde_warp_trust_error_c: float = 0.1,
    rf_er_sde_warp_trust_epsilon: float = 1e-8,
    rf_trust_lambda: float = 4.0,
    metadata: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample the trained velocity field from Gaussian noise at t=1 to t=0."""
    if guidance_scale < 0:
        raise ValueError("guidance_scale must be non-negative")
    if guidance_method not in available_guidance_methods():
        raise ValueError(
            "guidance_method must be one of "
            f"{', '.join(available_guidance_methods())}"
        )
    if guidance_scale != 1.0 and unconditional_cache is None:
        raise ValueError("CFG requires an unconditional condition cache")
    noise = torch.randn(
        (1, model.latent_channels, latent_height, latent_width),
        device=device, generator=generator,
    )

    def predict_velocity(state: torch.Tensor, timestep_values: torch.Tensor) -> torch.Tensor:
        timestep = timestep_values.to(device=state.device, dtype=torch.float32)
        velocity = model(state, timestep, condition_cache, metadata)
        if guidance_scale != 1.0:
            unconditional = model(state, timestep, unconditional_cache, metadata)
            velocity = apply_guidance(
                unconditional, velocity, guidance_scale, method=guidance_method,
            )
        return velocity

    return sample_flow_matching(
        predict_velocity,
        noise,
        steps=steps,
        solver=solver,
        scheduler=scheduler,
        flow_shift=flow_shift,
        generator=generator,
        er_sde_sigma_max=er_sde_sigma_max,
        rf_er_sde_eta=rf_er_sde_eta,
        rf_er_sde_cutoff=rf_er_sde_cutoff,
        rf_2m_warp_type=rf_2m_warp_type,
        rf_2m_warp_shift=rf_2m_warp_shift,
        rf_er_sde_warp_type=rf_er_sde_warp_type,
        rf_er_sde_warp_shift=rf_er_sde_warp_shift,
        rf_er_sde_warp_trust_lambda=rf_er_sde_warp_trust_lambda,
        rf_er_sde_warp_trust_error_c=rf_er_sde_warp_trust_error_c,
        rf_er_sde_warp_trust_epsilon=rf_er_sde_warp_trust_epsilon,
        rf_trust_lambda=rf_trust_lambda,
    )


def _condition_cache(
    model: NoVFCBDiT,
    hidden: torch.Tensor,
    mask: torch.Tensor,
    positions: torch.Tensor,
    reference_latent: torch.Tensor | None = None,
    qwen_vision_mask: torch.Tensor | None = None,
) -> ConditionKVCache:
    if reference_latent is None:
        return model.prepare_condition(hidden[None], mask[None], positions[None])
    height, width = reference_latent.shape[-2:]
    ref_positions = grid_positions(
        1, height, width, device=reference_latent.device, reference_id=1,
    )
    return model.prepare_condition(
        hidden[None], mask[None], positions[None], reference_latent[None],
        torch.ones((1, height * width), dtype=torch.bool, device=hidden.device),
        ref_positions,
        qwen_vision_mask,
    )


def _unconditional_cache(model: NoVFCBDiT, *, device: torch.device) -> ConditionKVCache:
    dtype = model.adapter.qwen_in.weight.dtype
    hidden = torch.zeros((1, 1, model.adapter.qwen_dim), device=device, dtype=dtype)
    mask = torch.ones((1, 1), device=device, dtype=torch.bool)
    positions = torch.zeros((1, 1, 3), device=device)
    return model.prepare_condition(hidden, mask, positions)


def _read_prompts(args) -> list[str]:
    prompts = list(args.prompt or [])
    if args.prompt_file:
        prompts.extend(
            line.strip() for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not prompts:
        raise ValueError("Provide at least one --prompt or --prompt-file")
    if any(not prompt.strip() for prompt in prompts):
        raise ValueError("prompts must not be empty")
    return prompts


@torch.inference_mode()
def generate_one(
    prompt: str,
    reference: Image.Image | None,
    *,
    model: NoVFCBDiT,
    vae: torch.nn.Module,
    encoder,
    condition_layer: int | str,
    resolution: int | tuple[int, int],
    steps: int,
    guidance_scale: float,
    device: torch.device,
    guidance_method: str = "cfg",
    solver: str = "euler",
    scheduler: str = "flow_match_euler",
    flow_shift: float = 1.0,
    er_sde_sigma_max: float = 80.0,
    generator: torch.Generator,
    profile_components: bool = False,
    component_seconds: dict[str, float] | None = None,
    rf_er_sde_eta: float = 0.2,
    rf_er_sde_cutoff: float = 0.9,
    rf_2m_warp_type: str = "identity",
    rf_2m_warp_shift: float = 1.0,
    rf_er_sde_warp_type: str = "identity",
    rf_er_sde_warp_shift: float = 1.0,
    rf_er_sde_warp_trust_lambda: float = 4.0,
    rf_er_sde_warp_trust_error_c: float = 0.1,
    rf_er_sde_warp_trust_epsilon: float = 1e-8,
    rf_trust_lambda: float = 4.0,
) -> torch.Tensor:
    if isinstance(resolution, int):
        output_height = output_width = resolution
    else:
        output_height, output_width = resolution
    reference = None if reference is None else fit_image(reference, (output_height, output_width))
    with component_timer(profile_components, device) as qwen_timer:
        hidden, mask, positions, qwen_vision_mask = encoder.encode_condition(
            prompt, source_image=reference, condition_layer=condition_layer,
            include_positions=True, include_vision_mask=True,
        )
    if component_seconds is not None and profile_components:
        component_seconds["qwen_encode"] = qwen_timer["seconds"]
    hidden = hidden.to(device=device, dtype=model.adapter.qwen_in.weight.dtype)
    mask = mask.to(device=device, dtype=torch.bool)
    positions = positions.to(device=device, dtype=torch.float32)
    qwen_vision_mask = qwen_vision_mask.to(device=device, dtype=torch.bool)
    reference_latent = None
    if reference is not None:
        with component_timer(profile_components, device) as reference_timer:
            reference_latent = encode_qwen_image_latents(
                vae, image_tensor(reference)[None], sample_mode="mode",
            )[0]
        reference_latent = reference_latent.to(device=device)
        if component_seconds is not None and profile_components:
            component_seconds["vae_reference_encode"] = reference_timer["seconds"]
    with component_timer(profile_components, device) as condition_timer:
        condition = _condition_cache(
            model, hidden, mask, positions, reference_latent, qwen_vision_mask,
        )
        unconditional = (
            _unconditional_cache(model, device=device) if guidance_scale != 1.0 else None
        )
    if component_seconds is not None and profile_components:
        component_seconds["condition_prepare"] = condition_timer["seconds"]
    sample_metadata = (
        torch.tensor([[
            np.log(float(np.sqrt(output_height * output_width))),
            np.log(output_width / output_height),
        ]], device=device, dtype=torch.float32)
        if model.metadata_conditioning != "none" else None
    )
    with component_timer(profile_components, device) as dit_timer:
        latent = sample_latents(
            model, condition,
            latent_height=output_height // LATENT_DOWNSAMPLE_FACTOR,
            latent_width=output_width // LATENT_DOWNSAMPLE_FACTOR,
            metadata=sample_metadata,
            steps=steps,
            device=device,
            guidance_scale=guidance_scale,
            guidance_method=guidance_method,
            unconditional_cache=unconditional,
            generator=generator,
            solver=solver,
            scheduler=scheduler,
            flow_shift=flow_shift,
            er_sde_sigma_max=er_sde_sigma_max,
            rf_er_sde_eta=rf_er_sde_eta,
            rf_er_sde_cutoff=rf_er_sde_cutoff,
            rf_2m_warp_type=rf_2m_warp_type,
            rf_2m_warp_shift=rf_2m_warp_shift,
            rf_er_sde_warp_type=rf_er_sde_warp_type,
            rf_er_sde_warp_shift=rf_er_sde_warp_shift,
            rf_er_sde_warp_trust_lambda=rf_er_sde_warp_trust_lambda,
            rf_er_sde_warp_trust_error_c=rf_er_sde_warp_trust_error_c,
            rf_er_sde_warp_trust_epsilon=rf_er_sde_warp_trust_epsilon,
            rf_trust_lambda=rf_trust_lambda,
        )
    if component_seconds is not None and profile_components:
        component_seconds["dit_sampling"] = dit_timer["seconds"]
    with component_timer(profile_components, device) as decode_timer:
        image = decode_qwen_image_latents(vae, latent)[0].cpu()
    if component_seconds is not None and profile_components:
        component_seconds["vae_decode"] = decode_timer["seconds"]
    return image


def save_grid(images: list[torch.Tensor], prompts: list[str], output: Path) -> None:
    from torchvision.transforms.functional import to_pil_image

    tiles = [to_pil_image((image.clamp(-1, 1) + 1) * 0.5) for image in images]
    if not tiles:
        raise ValueError("No images to save")
    font = ImageFont.load_default()
    label_height = 48
    cols = min(2, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    width, height = tiles[0].size
    canvas = Image.new("RGB", (cols * width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (tile, prompt) in enumerate(zip(tiles, prompts, strict=True)):
        x, y = (index % cols) * width, (index // cols) * (height + label_height)
        canvas.paste(tile, (x, y))
        draw.text((x + 4, y + height + 4), f"{index + 1}: {prompt[:120]}",
                  fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main(argv=None) -> None:
    args = parse_args(argv)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if args.steps <= 0 or args.resolution is not None and args.resolution <= 0:
        raise ValueError("steps and resolution must be positive")
    if not np.isfinite(args.flow_shift) or args.flow_shift <= 0:
        raise ValueError("--flow-shift must be finite and positive")
    warped_solver_names = {
        "rf_2m_warp", "rf_er_sde_warp_1", "rf_er_sde_warp_2m",
        "rf_er_sde_warp_trust",
    }
    if args.solver in warped_solver_names and args.flow_shift != 1.0:
        raise ValueError(
            "warped RF solvers own their time transform and require --flow-shift 1"
        )
    if args.solver == "rf_trust_region" and (
        not np.isfinite(args.rf_trust_lambda) or args.rf_trust_lambda < 0
    ):
        raise ValueError("--rf-trust-lambda must be finite and non-negative")
    if not np.isfinite(args.er_sde_sigma_max) or args.er_sde_sigma_max <= 0:
        raise ValueError("--er-sde-sigma-max must be finite and positive")
    if not np.isfinite(args.rf_2m_warp_shift) or args.rf_2m_warp_shift <= 0:
        raise ValueError("--rf-2m-warp-shift must be finite and positive")
    if not np.isfinite(args.rf_er_sde_warp_shift) or args.rf_er_sde_warp_shift <= 0:
        raise ValueError("--rf-er-sde-warp-shift must be finite and positive")
    if (
        not np.isfinite(args.rf_er_sde_warp_trust_lambda)
        or args.rf_er_sde_warp_trust_lambda < 0
    ):
        raise ValueError("--rf-er-sde-warp-trust-lambda must be finite and non-negative")
    if (
        not np.isfinite(args.rf_er_sde_warp_trust_error_c)
        or args.rf_er_sde_warp_trust_error_c <= 0
    ):
        raise ValueError("--rf-er-sde-warp-trust-error-c must be finite and positive")
    if (
        not np.isfinite(args.rf_er_sde_warp_trust_epsilon_abs)
        or args.rf_er_sde_warp_trust_epsilon_abs <= 0
    ):
        raise ValueError("--rf-er-sde-warp-trust-epsilon-abs must be finite and positive")
    if not np.isfinite(args.rf_er_sde_eta) or args.rf_er_sde_eta < 0:
        raise ValueError("--rf-er-sde-eta must be finite and non-negative")
    if (
        not np.isfinite(args.rf_er_sde_cutoff)
        or not 0 <= args.rf_er_sde_cutoff < 1
    ):
        raise ValueError("--rf-er-sde-cutoff must be in [0, 1)")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    _network, config = read_checkpoint(checkpoint_path)
    prompts = _read_prompts(args)
    references = list(args.reference_image or [])
    if references and len(references) != len(prompts):
        raise ValueError("repeat --reference-image once per prompt (use an empty value for T2I)")
    reference_images = [Image.open(path).convert("RGB") if path else None for path in references]
    if not reference_images:
        reference_images = [None] * len(prompts)
    resolution = args.resolution or int(config["resolution"])
    reference_factor = int(config.get("latent_downsample_factor", 1))
    target_factor = int(config.get(
        "target_latent_downsample_factor", reference_factor,
    ))
    required_resolution_multiple = max(
        16, LATENT_DOWNSAMPLE_FACTOR * max(reference_factor, target_factor),
    )
    if resolution % required_resolution_multiple:
        raise ValueError(
            f"resolution must be a multiple of {required_resolution_multiple} "
            "for the checkpoint latent downsample factor"
        )
    from vfp_dit_runtime.hf_data import resolution_bucket_size
    output_size = resolution_bucket_size(
        resolution, args.aspect_ratio, required_resolution_multiple,
    )
    condition_layer = parse_condition_layer(
        args.condition_layer or str(config["condition_layer"]),
    )
    dtype_name = args.dtype
    if dtype_name == "auto":
        dtype_name = "bf16" if device.type == "cuda" else "fp32"
    model_dtype = torch.bfloat16 if dtype_name == "bf16" and device.type == "cuda" else torch.float32
    model = build_model(config, device).to(dtype=model_dtype)
    weights = load_file(str(checkpoint_path), device="cpu")
    model.load_checkpoint_state_dict(weights, adapter_type=config.get("adapter_type"))
    del weights
    vae = load_qwen_image_vae(
        model_id=args.vae_model or str(config["vae_model"]),
        device=device, dtype=model_dtype,
    )
    encoder = load_qwen35_encoder(
        model_id=args.vlm_model or str(config["vlm_model"]),
        device=device, dtype=model_dtype,
    )
    validate_condition_layer(condition_layer, encoder.layer_count)
    if encoder.condition_dim != model.adapter.qwen_dim:
        raise ValueError("Qwen model hidden width does not match checkpoint architecture")
    images = []
    for index, (prompt, reference) in enumerate(zip(prompts, reference_images, strict=True)):
        generator = torch.Generator(device=device).manual_seed(args.seed + index)
        component_seconds: dict[str, float] = {}
        with _autocast(device, dtype_name):
            image = generate_one(
                prompt, reference, model=model, vae=vae, encoder=encoder,
                condition_layer=condition_layer, resolution=output_size,
                steps=args.steps, guidance_scale=args.guidance_scale,
                guidance_method=args.guidance_method,
                device=device, generator=generator, solver=args.solver,
                scheduler=args.scheduler, flow_shift=args.flow_shift,
                er_sde_sigma_max=args.er_sde_sigma_max,
                rf_er_sde_eta=args.rf_er_sde_eta,
                rf_er_sde_cutoff=args.rf_er_sde_cutoff,
                rf_2m_warp_type=args.rf_2m_warp_type,
                rf_2m_warp_shift=args.rf_2m_warp_shift,
                rf_er_sde_warp_type=args.rf_er_sde_warp_type,
                rf_er_sde_warp_shift=args.rf_er_sde_warp_shift,
                rf_er_sde_warp_trust_lambda=args.rf_er_sde_warp_trust_lambda,
                rf_er_sde_warp_trust_error_c=args.rf_er_sde_warp_trust_error_c,
                rf_er_sde_warp_trust_epsilon=args.rf_er_sde_warp_trust_epsilon_abs,
                rf_trust_lambda=args.rf_trust_lambda,
                profile_components=args.profile_components,
                component_seconds=component_seconds,
            )
        images.append(image)
        print(f"generated {index + 1}/{len(prompts)}")
        if args.profile_components:
            print("component_seconds " + " ".join(
                f"{name}={seconds:.4f}" for name, seconds in component_seconds.items()
            ))
    output = Path(args.output).expanduser().resolve()
    save_grid(images, prompts, output)
    metadata = build_sampling_metadata(
        solver=args.solver,
        scheduler=args.scheduler,
        flow_shift=args.flow_shift,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        guidance_method=args.guidance_method,
        seed=args.seed,
        num_samples=len(prompts),
        er_sde_sigma_max=args.er_sde_sigma_max,
        rf_er_sde_eta=args.rf_er_sde_eta,
        rf_er_sde_cutoff=args.rf_er_sde_cutoff,
        rf_2m_warp_type=args.rf_2m_warp_type,
        rf_2m_warp_shift=args.rf_2m_warp_shift,
        rf_er_sde_warp_type=args.rf_er_sde_warp_type,
        rf_er_sde_warp_shift=args.rf_er_sde_warp_shift,
        rf_er_sde_warp_trust_lambda=args.rf_er_sde_warp_trust_lambda,
        rf_er_sde_warp_trust_error_c=args.rf_er_sde_warp_trust_error_c,
        rf_er_sde_warp_trust_epsilon=args.rf_er_sde_warp_trust_epsilon_abs,
        rf_trust_lambda=args.rf_trust_lambda,
        output=str(output),
        checkpoint=str(checkpoint_path),
        prompts=prompts,
        reference_images=references if references else [None] * len(prompts),
        resolution=resolution,
        output_size=output_size,
        device=str(device),
        dtype=dtype_name,
    )
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"saved {output}")
    print(f"saved {metadata_path}")


if __name__ == "__main__":
    main()
