"""Train the single-stage, VFCB-free multimodal flow-matching DiT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.device import resolve_device  # noqa: E402
from runtime.config import cli_option_provided  # noqa: E402
from vfp_dit_runtime.qwen35 import (  # noqa: E402
    DEFAULT_QWEN35_MODEL,
    load_qwen35_encoder,
    parse_condition_layer,
    validate_condition_layer,
)
from vfp_dit_runtime.training import (  # noqa: E402
    add_common_arguments,
    autocast_context,
    run_training,
    seed_everything,
    validate_common_training_args,
)
from .encoders import (  # noqa: E402
    DEFAULT_VAE_MODEL,
    LATENT_CHANNELS,
    encode_qwen_image_latents,
    load_qwen_image_vae,
)
from .model import (  # noqa: E402
    METADATA_FFN_GATE_MAPPINGS,
    METADATA_SCALE_MAPPINGS,
    NoVFCBDiT,
    grid_positions,
)
from .samplers import SCHEDULERS, SOLVERS  # noqa: E402
from flow_sampling import available_guidance_methods  # noqa: E402
from runtime.profiling import component_timer  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a VFCB-free VFP-DiT with multimodal cached conditions",
    )
    add_common_arguments(
        parser,
        default_output="vfp_dit/output",
        default_optimizer="APOLLO",
    )
    parser.set_defaults(apollo_rank=32, lr=1e-3)
    parser.add_argument(
        "--resolution-levels", type=_parse_resolution_levels, default=None,
        help="Comma-separated nominal square-equivalent resolutions; default is 0.5x, 0.75x, and 1x.",
    )
    parser.add_argument(
        "--aspect-ratios", type=_parse_aspect_ratios,
        default=(0.5, 0.5625, 2 / 3, 0.75, 1.0, 4 / 3, 1.5, 16 / 9, 2.0),
        help="Target W:H buckets; default covers portrait and landscape through 2:1.",
    )
    parser.add_argument("--vlm-model", default=DEFAULT_QWEN35_MODEL)
    parser.add_argument("--vae-model", default=DEFAULT_VAE_MODEL)
    parser.add_argument(
        "--encoder-device", choices=("training", "cpu"), default="training",
        help="Place the frozen Qwen encoder on the training device or CPU.",
    )
    parser.add_argument(
        "--vae-device", choices=("training", "cpu"), default=None,
        help="Place the frozen VAE on the training device or CPU; defaults to --encoder-device.",
    )
    parser.add_argument(
        "--encoder-prefetch-batches", type=int, default=2,
        help=(
            "Bounded number of raw/encoded HF batches held ahead by the CPU encoder thread; "
            "used only when both Qwen and VAE are on CPU. Set 0 to encode synchronously."
        ),
    )
    parser.add_argument("--condition-layer", type=parse_condition_layer, default="final")
    parser.add_argument("--condition-dim", type=int, default=1024)
    parser.add_argument("--latent-channels", type=int, default=LATENT_CHANNELS)
    parser.add_argument(
        "--reference-latent-downsample-factor", "--latent-downsample-factor",
        dest="latent_downsample_factor", type=int, choices=(1,), default=1,
        help=(
            "Reference-latent path factor. The Qwen-fused reference path retains "
            "its full-grid tokens; this setting applies to the legacy alignment path."
        ),
    )
    parser.add_argument(
        "--target-latent-downsample-factor", type=int, choices=(1, 2, 4), default=2,
        help="Spatial factor for target input patchification and output upsampling.",
    )
    parser.add_argument(
        "--output-refinement-depth", type=int, default=2,
        help="Number of shallow full-resolution GQA refinement blocks after target upsampling.",
    )
    parser.add_argument(
        "--output-refinement-conditioning",
        choices=("none", "cross_attention"), default=None,
        help=(
            "Add target-query cross-attention to adapted Qwen/reference tokens in the "
            "full-resolution output blocks."
        ),
    )
    parser.add_argument(
        "--output-skip-fusion-mode", choices=("add", "concat_linear"), default=None,
        help="Combine upsampled target features and the latent skip by addition or legacy concat+linear.",
    )
    parser.add_argument(
        "--output-head-ada-scale", action=argparse.BooleanOptionalAction, default=None,
        help=(
            "Apply unit-initialized metadata Ada scale to the nonlinear velocity-head "
            "correction branch only; defaults on for new Ada runs with output refinement."
        ),
    )
    parser.add_argument(
        "--fuse-reference-latent-to-vision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="downsample reference latents by 2, align Qwen vision features, and fuse their 1024-channel projections",
    )
    parser.add_argument(
        "--reference-latent-fusion-mode",
        choices=(
            "legacy_latent_to_qwen", "qwen_to_full_latent",
            "qwen_to_half_latent", "qwen_to_half_latent_add",
        ),
        default=None,
        help="Reference fusion architecture; fresh runs default to additive half-grid fusion.",
    )
    parser.add_argument("--vae-latent-mode", choices=("mode", "sample"), default="mode")
    parser.add_argument("--model-width", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--adapter-depth", type=int, default=2)
    parser.add_argument("--ff-mult", type=float, default=3.0)
    parser.add_argument(
        "--metadata-conditioning",
        choices=("none", "ada_attn_ffn"),
        default=None,
        help=(
            "none disables metadata; ada_attn_ffn applies metadata Ada scale and shift "
            "to both attention and FFN branches. New runs default to ada_attn_ffn; "
            "resume runs inherit the checkpoint setting."
        ),
    )
    parser.add_argument(
        "--metadata-scale-mapping", choices=METADATA_SCALE_MAPPINGS, default=None,
        help=(
            "Unit-initialized Ada scale function for ada_attn_ffn; defaults to "
            "softplus1_normalized when Ada is enabled and linear otherwise."
        ),
    )
    parser.add_argument(
        "--metadata-shift", action=argparse.BooleanOptionalAction, default=None,
        help=(
            "Add zero-initialized metadata bias to both Ada-conditioned branches. "
            "Defaults on for new Ada runs and inherits the checkpoint on resume."
        ),
    )
    parser.add_argument(
        "--metadata-ffn-residual-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use a zero-initialized, meta-conditioned per-channel FFN residual gate; "
            "defaults on for ada_attn_ffn and off otherwise."
        ),
    )
    parser.add_argument(
        "--metadata-ffn-gate-mapping", choices=METADATA_FFN_GATE_MAPPINGS,
        default=None,
        help=(
            "Map the Ada FFN residual gate with linear raw values or SiLU; fresh runs "
            "default to silu and older resume configs retain linear."
        ),
    )
    parser.add_argument(
        "--attention-head-gate",
        choices=("input_silu", "timestep_sigmoid"),
        default="input_silu",
        help=(
            "Attention head gate; input_silu uses 1+SiLU from Ada-modulated target "
            "features, timestep_sigmoid restores the legacy checkpoint behavior."
        ),
    )
    parser.add_argument(
        "--fuse-same-input-projections",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Combine target QKV+head-gate, condition KV, and metadata projections. "
            "Enabled by default for new runs; omitted resume/init settings are inferred from the checkpoint."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction,
        default=True,
        help="Trade extra compute for reduced DiT activation memory (default: enabled).",
    )
    parser.add_argument(
        "--log-adapter-gradients", action="store_true",
        help="Record per-branch adapter gradient norms after each optimizer step",
    )
    parser.add_argument("--timesteps-per-image", type=int, default=4)
    parser.add_argument(
        "--observe-interval", type=int, default=1000,
        help="Write interval-mean loss to metrics.jsonl every N optimizer steps and sample images; 0 disables observations",
    )
    parser.add_argument("--sample-steps", type=int, default=30,
                        help="Number of solver steps for observe-interval image sampling")
    parser.add_argument(
        "--sample-solver", "--sample-sampler", dest="sample_solver",
        choices=SOLVERS, default="euler",
        help=(
            "Update rule for observation samples (default: euler). "
            "Non-default rf_* solvers are experimental; see "
            "flow_sampling/README.md for validation limits."
        ),
    )
    parser.add_argument(
        "--sample-scheduler", choices=SCHEDULERS,
        default="flow_match_euler",
    )
    parser.add_argument("--sample-flow-shift", type=float, default=1.0)
    parser.add_argument("--sample-er-sde-sigma-max", type=float, default=80.0)
    parser.add_argument("--sample-rf-er-sde-eta", type=float, default=0.2)
    parser.add_argument("--sample-rf-er-sde-cutoff", type=float, default=0.9)
    parser.add_argument("--sample-rf-2m-warp-type", choices=("identity", "rational"), default="identity")
    parser.add_argument("--sample-rf-2m-warp-shift", type=float, default=1.0)
    parser.add_argument(
        "--sample-rf-er-sde-warp-type", choices=("identity", "rational"),
        default="identity",
    )
    parser.add_argument("--sample-rf-er-sde-warp-shift", type=float, default=1.0)
    parser.add_argument("--sample-rf-er-sde-warp-trust-lambda", type=float, default=4.0)
    parser.add_argument("--sample-rf-er-sde-warp-trust-error-c", type=float, default=0.1)
    parser.add_argument("--sample-rf-er-sde-warp-trust-epsilon-abs", type=float, default=1e-8)
    parser.add_argument("--sample-rf-trust-lambda", type=float, default=4.0)
    parser.add_argument("--sample-guidance-scale", type=float, default=4.0)
    parser.add_argument(
        "--sample-guidance-method", choices=available_guidance_methods(), default="cfg",
        help="Guidance transform used for observe-interval image sampling",
    )
    parser.add_argument(
        "--sample-prompt", action="append", default=None,
        help="Fixed T2I prompt for observation samples; repeat to save multiple images",
    )
    parser.add_argument(
        "--no-observe-samples", action="store_true",
        help="Keep interval loss JSONL observations but skip image sampling",
    )
    parser.add_argument("--profile-components", action="store_true")
    parser.add_argument(
        "--log-metadata-diagnostics", action="store_true",
        help="Log Ada scale/shift ranges across DiT blocks into progress and metrics",
    )
    args = parser.parse_args(argv)
    args.metadata_scale_mapping_explicit = args.metadata_scale_mapping is not None
    return args


def _resolve_metadata_conditioning_args(args, saved_config: dict | None) -> None:
    """Resolve fresh-run Ada defaults while preserving resume checkpoint settings."""
    resume_config = saved_config if args.resume and saved_config is not None else None
    requested_conditioning = args.metadata_conditioning
    if requested_conditioning is None:
        args.metadata_conditioning = (
            str(resume_config.get("metadata_conditioning", "none"))
            if resume_config is not None else "ada_attn_ffn"
        )
    inherit_saved_ada = (
        resume_config is not None
        and resume_config.get("metadata_conditioning", "none")
        == args.metadata_conditioning
    )

    if args.metadata_scale_mapping is None:
        if inherit_saved_ada:
            args.metadata_scale_mapping = str(
                resume_config.get("metadata_scale_mapping", "linear")
            )
        else:
            args.metadata_scale_mapping = (
                "softplus1_normalized"
                if args.metadata_conditioning == "ada_attn_ffn" else "linear"
            )
    if args.metadata_shift is None:
        args.metadata_shift = (
            bool(resume_config.get("metadata_shift", False))
            if inherit_saved_ada else args.metadata_conditioning == "ada_attn_ffn"
        )
    if args.metadata_ffn_residual_gate is None:
        args.metadata_ffn_residual_gate = (
            bool(resume_config.get("metadata_ffn_residual_gate", False))
            if inherit_saved_ada else args.metadata_conditioning == "ada_attn_ffn"
        )
    if args.metadata_ffn_gate_mapping is None:
        args.metadata_ffn_gate_mapping = (
            str(resume_config.get("metadata_ffn_gate_mapping", "linear"))
            if resume_config is not None else "silu"
        )

    scale_mapping_was_explicit = args.metadata_scale_mapping_explicit
    del args.metadata_scale_mapping_explicit
    if args.metadata_shift and args.metadata_conditioning != "ada_attn_ffn":
        raise ValueError("--metadata-shift requires --metadata-conditioning ada_attn_ffn")
    if args.metadata_ffn_residual_gate and args.metadata_conditioning != "ada_attn_ffn":
        raise ValueError(
            "--metadata-ffn-residual-gate requires --metadata-conditioning ada_attn_ffn"
        )
    if (
        scale_mapping_was_explicit
        and args.metadata_scale_mapping != "linear"
        and args.metadata_conditioning != "ada_attn_ffn"
    ):
        raise ValueError("nonlinear --metadata-scale-mapping requires ada_attn_ffn conditioning")


def _parse_resolution_levels(value: str) -> tuple[int, ...]:
    try:
        levels = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("resolution-levels must be comma-separated integers") from error
    if not levels or any(level <= 0 for level in levels) or len(set(levels)) != len(levels):
        raise argparse.ArgumentTypeError("resolution-levels must be unique positive integers")
    return tuple(sorted(levels))


def _parse_aspect_ratios(value: str) -> tuple[float, ...]:
    ratios = []
    try:
        for part in value.split(","):
            if ":" in part:
                numerator, denominator = part.split(":", 1)
                ratio = float(numerator.strip()) / float(denominator.strip())
            else:
                ratio = float(part.strip())
            ratios.append(ratio)
    except (ValueError, ZeroDivisionError) as error:
        raise argparse.ArgumentTypeError(
            "aspect-ratios must be comma-separated numbers or W:H values"
        ) from error
    if not ratios or any(not np.isfinite(ratio) or ratio <= 0 for ratio in ratios):
        raise argparse.ArgumentTypeError("aspect-ratios must be finite and positive")
    if len(set(ratios)) != len(ratios):
        raise argparse.ArgumentTypeError("aspect-ratios must be unique")
    return tuple(sorted(ratios))


def _fit_rgb(image: Image.Image, resolution: int | tuple[int, int]) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("dataset images must be PIL.Image.Image instances")
    height, width = (resolution, resolution) if isinstance(resolution, int) else resolution
    return ImageOps.fit(
        image.convert("RGB"),
        (width, height),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def _image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


@torch.no_grad()
def encode_hf_batch(
    batch: dict,
    *,
    vae: torch.nn.Module,
    encoder,
    condition_layer: int | str,
    resolution: int,
    vae_latent_mode: str,
    profile_components: bool = False,
) -> dict[str, Any]:
    """Encode target/ref images, Qwen hidden, masks, and modality positions."""
    targets = batch.get("target_images")
    sources = batch.get("source_images")
    prompts = batch.get("prompts")
    if not isinstance(targets, list) or not targets:
        raise ValueError("HF batch must contain target_images")
    if not isinstance(sources, list) or len(sources) != len(targets):
        raise ValueError("source_images must match target_images")
    if not isinstance(prompts, list) or len(prompts) != len(targets):
        raise ValueError("prompts must match target_images")

    bucket_size = tuple(batch.get("bucket_size", (resolution, resolution)))
    target_images = [_fit_rgb(image, bucket_size) for image in targets]
    profile_enabled = profile_components
    # The VAE is an nn.Module, while Qwen35FeatureEncoder is a lightweight
    # wrapper around its model and exposes the placement as ``.device``.
    # Resolve each independently so profiling also works when only one encoder
    # is on CUDA.
    encoder_devices: list[torch.device] = []
    vae_parameter = next(vae.parameters(), None)
    if vae_parameter is not None:
        encoder_devices.append(vae_parameter.device)
    qwen_device = getattr(encoder, "device", None)
    if qwen_device is not None:
        encoder_devices.append(torch.device(qwen_device))
    profile_device = next(
        (device for device in encoder_devices if device.type == "cuda"),
        encoder_devices[0] if encoder_devices else torch.device("cpu"),
    )
    component_seconds: dict[str, float] = {}
    with component_timer(profile_enabled, profile_device) as target_vae_timer:
        clean_latent = encode_qwen_image_latents(
            vae,
            torch.stack([_image_tensor(image) for image in target_images]),
            sample_mode=vae_latent_mode,
        )
    if profile_enabled:
        component_seconds["vae_target_encode"] = target_vae_timer["seconds"]
    reference_latent = None
    source_rows = [
        (index, _fit_rgb(image, bucket_size))
        for index, image in enumerate(sources) if image is not None
    ]
    if source_rows:
        with component_timer(profile_enabled, profile_device) as reference_vae_timer:
            source_batch = encode_qwen_image_latents(
                vae,
                torch.stack([_image_tensor(image) for _, image in source_rows]),
                sample_mode=vae_latent_mode,
            )
        if profile_enabled:
            component_seconds["vae_reference_encode"] = reference_vae_timer["seconds"]
        reference_latent = clean_latent.new_zeros(
            (len(targets), LATENT_CHANNELS, *source_batch.shape[-2:]),
        )
        for row, (index, _image) in enumerate(source_rows):
            reference_latent[index] = source_batch[row]

    hidden_rows = []
    mask_rows = []
    position_rows = []
    vision_mask_rows = []
    with component_timer(profile_enabled, profile_device) as qwen_timer:
        for prompt, source in zip(prompts, sources, strict=True):
            if not isinstance(prompt, str):
                raise TypeError("every prompt must be a string")
            encoded = encoder.encode_condition(
                prompt,
                source_image=None if source is None else _fit_rgb(source, bucket_size),
                condition_layer=condition_layer,
                include_positions=True,
                include_vision_mask=True,
            )
            hidden, mask, positions, vision_mask = encoded
            if mask.shape != (hidden.shape[0],) or positions.shape != (hidden.shape[0], 3):
                raise ValueError("Qwen hidden, mask, and positions have inconsistent lengths")
            hidden_rows.append(hidden.float())
            mask_rows.append(mask.bool())
            position_rows.append(positions.float())
            vision_mask_rows.append(vision_mask.bool())
    if profile_enabled:
        component_seconds["qwen_encode"] = qwen_timer["seconds"]
    qwen_hidden = pad_sequence(hidden_rows, batch_first=True)
    qwen_positions = pad_sequence(position_rows, batch_first=True)
    qwen_mask = torch.zeros(qwen_hidden.shape[:2], device=qwen_hidden.device, dtype=torch.bool)
    qwen_vision_mask = torch.zeros_like(qwen_mask)
    for row, (mask, vision_mask) in enumerate(zip(mask_rows, vision_mask_rows, strict=True)):
        qwen_mask[row, :mask.numel()] = mask
        qwen_vision_mask[row, :vision_mask.numel()] = vision_mask

    result = {
        "clean_latent": clean_latent.float(),
        "metadata": batch.get("metadata"),
        "qwen_hidden": qwen_hidden,
        "qwen_mask": qwen_mask,
        "qwen_positions": qwen_positions,
        "qwen_vision_mask": qwen_vision_mask,
        "profile_component_seconds": component_seconds,
    }
    if reference_latent is not None:
        batch_size, _channels, height, width = reference_latent.shape
        reference_mask = torch.zeros(
            (batch_size, height * width), device=reference_latent.device, dtype=torch.bool,
        )
        for row, source in enumerate(sources):
            if source is not None:
                reference_mask[row] = True
        result.update({
            "reference_latent": reference_latent.float(),
            "reference_mask": reference_mask,
            "reference_positions": grid_positions(
                batch_size, height, width,
                device=reference_latent.device,
                reference_id=1,
            ),
        })
    return result


def _drop_conditions(
    encoded: dict[str, torch.Tensor],
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    hidden = encoded["qwen_hidden"]
    mask = encoded["qwen_mask"]
    positions = encoded["qwen_positions"]
    batch = hidden.shape[0]
    dropped = torch.rand(batch, device=hidden.device) < probability
    if probability:
        hidden = hidden.clone()
        mask = mask.clone()
        positions = positions.clone()
        hidden[dropped] = 0
        mask[dropped] = False
        mask[dropped, 0] = True
        positions[dropped] = 0

    reference_latent = encoded.get("reference_latent")
    reference_mask = encoded.get("reference_mask")
    if reference_mask is not None and probability:
        reference_latent = reference_latent.clone()
        reference_mask = reference_mask.clone()
        reference_latent[dropped] = 0
        reference_mask[dropped] = False
    return hidden, mask, positions, reference_latent, reference_mask, dropped.float().mean()


def sample_stratified_timesteps(
    batch_size: int,
    samples_per_image: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Draw one independent t from each equal-width interval per image."""
    if batch_size <= 0 or samples_per_image <= 0:
        raise ValueError("batch_size and samples_per_image must be positive")
    intervals = torch.arange(samples_per_image, device=device, dtype=torch.float32)
    jitter = torch.rand((batch_size, samples_per_image), device=device)
    return (intervals[None, :] + jitter) / samples_per_image


def _finite_tensor_summary(name: str, tensor: torch.Tensor) -> str:
    values = tensor.detach().float()
    finite = torch.isfinite(values)
    finite_values = values[finite]
    details = (
        f"finite_min={finite_values.min().item():.6g} "
        f"finite_max={finite_values.max().item():.6g}"
        if finite_values.numel() else "finite_min=n/a finite_max=n/a"
    )
    return (
        f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"nonfinite={int((~finite).sum().item())}/{tensor.numel()} {details}"
    )


def _raise_nonfinite_flow_loss(batch, tensors: dict[str, torch.Tensor]) -> None:
    tensor_details = "\n  ".join(
        _finite_tensor_summary(name, tensor)
        for name, tensor in tensors.items()
        if isinstance(tensor, torch.Tensor)
    )
    identity = " ".join(
        f"{key}={batch[key]!r}"
        for key in ("conditioning_types", "sample_ids", "sources")
        if key in batch
    ) or "batch_identity=unavailable"
    raise FloatingPointError(
        f"Non-finite flow loss ({identity})\n  {tensor_details}"
    )


def _flow_step(model, batch, args, _device, *, vae, encoder):
    encoded = batch.get("_encoded_features")
    if encoded is None:
        encoded = encode_hf_batch(
            batch,
            vae=vae,
            encoder=encoder,
            condition_layer=args.condition_layer,
            resolution=args.resolution,
            vae_latent_mode=args.vae_latent_mode,
            profile_components=getattr(args, "profile_components", False),
        )
    encoded = {
        key: value.to(device=_device) if isinstance(value, torch.Tensor) else value
        for key, value in encoded.items()
    }
    clean = encoded["clean_latent"]
    hidden, mask, positions, reference, reference_mask, drop_fraction = _drop_conditions(
        encoded, args.condition_dropout,
    )
    profile_enabled = bool(getattr(args, "profile_components", False))
    component_seconds = encoded.pop("profile_component_seconds")
    with component_timer(profile_enabled, clean.device) as condition_timer:
        cache = model.prepare_condition(
            hidden,
            mask,
            positions,
            reference,
            reference_mask,
            qwen_vision_mask=encoded["qwen_vision_mask"],
        )
    if profile_enabled:
        component_seconds["condition_prepare"] = condition_timer["seconds"]

    batch_size = clean.shape[0]
    timestep_count = getattr(args, "timesteps_per_image", 1)
    with component_timer(profile_enabled, clean.device) as sampling_timer:
        timestep_grid = sample_stratified_timesteps(
            batch_size, timestep_count, device=clean.device,
        )
        timestep = timestep_grid.reshape(batch_size * timestep_count)
        clean_repeated = clean.repeat_interleave(timestep_count, dim=0)
        noise = torch.randn_like(clean_repeated)
        timestep_image = timestep.to(clean_repeated.dtype).reshape(
            batch_size * timestep_count, 1, 1, 1,
        )
        noisy = (1.0 - timestep_image) * clean_repeated + timestep_image * noise
        target_velocity = noise - clean_repeated
        repeated_cache = cache.repeat_interleave(timestep_count)
    if profile_enabled:
        component_seconds["stratified_noise_sampling"] = sampling_timer["seconds"]

    with component_timer(profile_enabled, clean.device) as dit_timer:
        metadata = encoded["metadata"]
        repeated_metadata = (
            metadata.to(device=clean.device).repeat_interleave(timestep_count, dim=0)
            if model.metadata_conditioning != "none" else None
        )
        prediction = model(noisy, timestep, repeated_cache, repeated_metadata)
    if profile_enabled:
        component_seconds["dit_forward"] = dit_timer["seconds"]
    if getattr(args, "log_metadata_diagnostics", False):
        metrics_metadata = model.metadata_diagnostics(metadata)
    else:
        metrics_metadata = {}
    with component_timer(profile_enabled, clean.device) as loss_timer:
        squared_error = (prediction.float() - target_velocity.float()).square()
        per_timestep_loss = squared_error.mean(dim=(1, 2, 3))
        per_example_loss = per_timestep_loss.reshape(batch_size, timestep_count).mean(dim=1)
        flow_mse = per_timestep_loss.mean()
    if (
        getattr(args, "check_finite_updates", False)
        and not torch.isfinite(flow_mse)
    ):
        _raise_nonfinite_flow_loss(batch, {
            "vae_clean_latent": clean,
            "qwen_condition_hidden": encoded["condition_hidden"],
            "condition_hidden_after_dropout": hidden,
            "reference_condition": reference,
            "noisy_target": noisy,
            "target_velocity": target_velocity,
            "dit_prediction": prediction,
            "squared_error": squared_error,
            "flow_mse": flow_mse,
        })
    if profile_enabled:
        component_seconds["loss"] = loss_timer["seconds"]

    metrics = {
        "loss": flow_mse.detach(),
        "flow_mse": flow_mse.detach(),
        "per_example_loss": per_example_loss.detach(),
        "condition_drop_fraction": drop_fraction.detach(),
    }
    metrics.update(metrics_metadata)
    if profile_enabled:
        metrics.update({
            f"profile_{name}_seconds": torch.tensor(
                seconds, device=clean.device, dtype=torch.float32,
            )
            for name, seconds in component_seconds.items()
        })
    return flow_mse, metrics


def _make_step_fn(vae, encoder):
    def step(model, batch, args, device):
        return _flow_step(
            model, batch, args, device, vae=vae, encoder=encoder,
        )
    return step


def _make_prepare_batch_fn(args, vae, encoder):
    """Run frozen feature extraction before the trainable DiT step."""
    def prepare(batch):
        encoded = encode_hf_batch(
            batch,
            vae=vae,
            encoder=encoder,
            condition_layer=args.condition_layer,
            resolution=args.resolution,
            vae_latent_mode=args.vae_latent_mode,
            profile_components=getattr(args, "profile_components", False),
        )
        return {
            key: batch[key]
            for key in ("prompts", "conditioning_types", "sample_ids", "sources")
            if key in batch
        } | {"_encoded_features": encoded}
    return prepare


def _make_observe_fn(vae, encoder):
    from .generate_samples import generate_one, save_grid

    def observe(model, args, device, output_dir: Path, global_step: int):
        prompts = list(args.sample_prompt or [
            "A small red wooden boat on a quiet lake at sunrise, realistic photography.",
            "A crowded night market on a rainy city street, bright signs reflected on wet pavement, documentary street photography.",
            "Extreme close-up macro photograph of a metallic blue beetle walking across a vivid green leaf, soft natural light.",
            "A tiny orange fox reading a book beneath oversized mushrooms in a misty forest, whimsical watercolor illustration.",
        ])
        images = []
        was_training = model.training
        model.eval()
        started = perf_counter()
        try:
            with torch.inference_mode():
                for index, prompt in enumerate(prompts):
                    generator = torch.Generator(device=device).manual_seed(
                        args.seed + index,
                    )
                    with autocast_context(device, args.amp):
                        image = generate_one(
                            prompt, None,
                            model=model,
                            vae=vae,
                            encoder=encoder,
                            condition_layer=args.condition_layer,
                            resolution=args.resolution,
                            steps=args.sample_steps,
                            guidance_scale=args.sample_guidance_scale,
                            guidance_method=args.sample_guidance_method,
                            device=device,
                            generator=generator,
                            solver=args.sample_solver,
                            scheduler=args.sample_scheduler,
                            flow_shift=args.sample_flow_shift,
                            er_sde_sigma_max=args.sample_er_sde_sigma_max,
                            rf_er_sde_eta=args.sample_rf_er_sde_eta,
                            rf_er_sde_cutoff=args.sample_rf_er_sde_cutoff,
                            rf_2m_warp_type=args.sample_rf_2m_warp_type,
                            rf_2m_warp_shift=args.sample_rf_2m_warp_shift,
                            rf_er_sde_warp_type=args.sample_rf_er_sde_warp_type,
                            rf_er_sde_warp_shift=args.sample_rf_er_sde_warp_shift,
                            rf_er_sde_warp_trust_lambda=(
                                args.sample_rf_er_sde_warp_trust_lambda
                            ),
                            rf_er_sde_warp_trust_error_c=(
                                args.sample_rf_er_sde_warp_trust_error_c
                            ),
                            rf_er_sde_warp_trust_epsilon=(
                                args.sample_rf_er_sde_warp_trust_epsilon_abs
                            ),
                            rf_trust_lambda=args.sample_rf_trust_lambda,
                        )
                    images.append(image)
            path = output_dir / f"step_{global_step:08d}.png"
            save_grid(images, prompts, path)
        finally:
            model.train(was_training)
        return {
            "path": str(path),
            "prompts": prompts,
            "steps": int(args.sample_steps),
            "solver": args.sample_solver,
            "callback_evaluations_per_sample": (
                args.sample_steps + 1
                if args.sample_solver in {"fireflow", "abm2"}
                else args.sample_steps
            ),
            "callback_evaluations_total": (
                (args.sample_steps + 1 if args.sample_solver in {"fireflow", "abm2"}
                 else args.sample_steps) * len(prompts)
            ),
            "cfg_network_forward_multiplier": (
                2 if args.sample_guidance_scale != 1.0 else 1
            ),
            "solver_version": (
                "0.1.1" if args.sample_solver.startswith("rf_er_sde_warp") else None
            ),
            "scheduler": args.sample_scheduler,
            "solver_grid": (
                "uniform_tau"
                if args.sample_solver in {
                    "rf_2m_warp", "rf_er_sde_warp_1", "rf_er_sde_warp_2m",
                    "rf_er_sde_warp_trust",
                }
                else None
            ),
            "flow_shift": float(args.sample_flow_shift),
            "er_sde_sigma_max": float(args.sample_er_sde_sigma_max),
            "rf_er_sde_eta": float(args.sample_rf_er_sde_eta),
            "rf_er_sde_cutoff": float(args.sample_rf_er_sde_cutoff),
            "rf_2m_warp_type": args.sample_rf_2m_warp_type,
            "rf_2m_warp_shift": float(args.sample_rf_2m_warp_shift),
            "rf_er_sde_warp_type": (
                args.sample_rf_er_sde_warp_type
                if args.sample_solver.startswith("rf_er_sde_warp") else None
            ),
            "rf_er_sde_warp_shift": (
                float(args.sample_rf_er_sde_warp_shift)
                if args.sample_solver.startswith("rf_er_sde_warp") else None
            ),
            "rf_er_sde_warp_q_min": (
                1e-3 if args.sample_solver.startswith("rf_er_sde_warp") else None
            ),
            "rf_er_sde_warp_q_max": (
                1e3 if args.sample_solver.startswith("rf_er_sde_warp") else None
            ),
            "rf_er_sde_warp_trust_lambda": (
                float(args.sample_rf_er_sde_warp_trust_lambda)
                if args.sample_solver == "rf_er_sde_warp_trust" else None
            ),
            "rf_er_sde_warp_trust_error_c": (
                float(args.sample_rf_er_sde_warp_trust_error_c)
                if args.sample_solver == "rf_er_sde_warp_trust" else None
            ),
            "rf_er_sde_warp_trust_epsilon_abs": (
                float(args.sample_rf_er_sde_warp_trust_epsilon_abs)
                if args.sample_solver == "rf_er_sde_warp_trust" else None
            ),
            "rf_er_sde_warp_sde_gate": (
                "trust_error" if args.sample_solver == "rf_er_sde_warp_trust" else None
            ),
            "rf_er_sde_warp_startup": (
                "deterministic_first_two_intervals"
                if args.sample_solver == "rf_er_sde_warp_trust" else None
            ),
            "rf_er_sde_time_gate": (
                "sin_pi" if args.sample_solver.startswith("rf_er_sde") else None
            ),
            "rf_trust_lambda": (
                float(args.sample_rf_trust_lambda)
                if args.sample_solver == "rf_trust_region" else None
            ),
            "guidance_scale": float(args.sample_guidance_scale),
            "guidance_method": args.sample_guidance_method,
            "seconds": perf_counter() - started,
        }

    return observe


def main(argv=None):
    args = parse_args(argv)
    args.adapter_type = "ffn"
    source_checkpoint = args.resume or args.init_checkpoint
    saved_config = None
    if source_checkpoint:
        checkpoint_path = Path(source_checkpoint).expanduser()
        if checkpoint_path.is_file():
            from safetensors import safe_open

            with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
                checkpoint_config = (checkpoint.metadata() or {}).get("vfp_dit.config")
            if checkpoint_config is not None:
                saved_config = json.loads(checkpoint_config)
    if args.resume and Path(args.resume).expanduser().is_file() and (
        saved_config is None or saved_config.get("adapter_type") != "ffn"
    ):
        raise ValueError(
            "VLM Transformer/linear adapters are retired and cannot be resumed. "
            "Use --init-checkpoint to start a fresh GatedFFN run from compatible weights."
        )
    if args.resume and saved_config is not None:
        # Preserve the architecture stored in a resume checkpoint unless the
        # caller explicitly requests a new target-grid configuration.
        if not cli_option_provided(
            sys.argv[1:] if argv is None else argv,
            "--target-latent-downsample-factor",
        ):
            args.target_latent_downsample_factor = int(
                saved_config.get(
                    "target_latent_downsample_factor",
                    saved_config.get("latent_downsample_factor", 1),
                ),
            )
        if not cli_option_provided(
            sys.argv[1:] if argv is None else argv, "--output-refinement-depth",
        ):
            args.output_refinement_depth = int(saved_config.get("output_refinement_depth", 0))
        if not cli_option_provided(
            sys.argv[1:] if argv is None else argv, "--output-refinement-conditioning",
        ):
            args.output_refinement_conditioning = str(
                saved_config.get("output_refinement_conditioning", "none")
            )
        if not cli_option_provided(
            sys.argv[1:] if argv is None else argv, "--output-skip-fusion-mode",
        ):
            args.output_skip_fusion_mode = str(
                saved_config.get("output_skip_fusion_mode", "concat_linear")
            )
        if not cli_option_provided(
            sys.argv[1:] if argv is None else argv,
            "--output-head-ada-scale", "--no-output-head-ada-scale",
        ):
            args.output_head_ada_scale = bool(
                saved_config.get("output_head_ada_scale", False)
            )
    if args.output_refinement_conditioning is None:
        args.output_refinement_conditioning = "none"
    if args.output_skip_fusion_mode is None:
        args.output_skip_fusion_mode = "add"
    _resolve_metadata_conditioning_args(args, saved_config)
    if args.output_head_ada_scale is None:
        args.output_head_ada_scale = (
            args.output_refinement_depth > 0
            and args.metadata_conditioning == "ada_attn_ffn"
        )
    if args.output_head_ada_scale:
        if args.output_refinement_depth == 0:
            raise ValueError("--output-head-ada-scale requires output refinement blocks")
        if args.metadata_conditioning != "ada_attn_ffn":
            raise ValueError(
                "--output-head-ada-scale requires --metadata-conditioning ada_attn_ffn"
            )
    if args.vae_device is None:
        if args.resume and saved_config is not None:
            args.vae_device = str(
                saved_config.get("vae_device", saved_config.get("encoder_device", args.encoder_device))
            )
        else:
            args.vae_device = args.encoder_device
    batch_encoder_prefetch = (
        args.encoder_device == "cpu" and args.vae_device == "cpu"
    )
    if not batch_encoder_prefetch and args.encoder_prefetch_batches:
        print(
            "Encoder batch prefetch disabled unless both Qwen and VAE are on CPU; "
            "mixed placement encodes synchronously to avoid concurrent GPU encoding."
        )
    if args.fuse_same_input_projections is None:
        args.fuse_same_input_projections = (
            bool(saved_config.get("fuse_same_input_projections", False))
            if saved_config is not None or source_checkpoint else True
        )
    validate_common_training_args(args)
    if args.data_mode != "hf":
        raise ValueError("VFP-DiT currently supports --data-mode=hf only")
    if args.resume and saved_config is not None:
        args.fuse_reference_latent_to_vision = bool(
            saved_config.get("fuse_reference_latent_to_vision", args.fuse_reference_latent_to_vision),
        )
        args.reference_latent_fusion_mode = str(saved_config.get(
            "reference_latent_fusion_mode", "legacy_latent_to_qwen",
        ))
    else:
        args.reference_latent_fusion_mode = args.reference_latent_fusion_mode or (
            "qwen_to_half_latent_add" if args.fuse_reference_latent_to_vision
            else "legacy_latent_to_qwen"
        )
    args.bucket_alignment = max(16, 8 * max(
        args.latent_downsample_factor, args.target_latent_downsample_factor,
    ))
    if args.resolution_levels is None:
        args.resolution_levels = tuple(sorted({
            max(args.bucket_alignment, round((args.resolution * scale) / args.bucket_alignment)
                * args.bucket_alignment)
            for scale in (0.5, 0.75, 1.0)
        }))
    if any(level % args.bucket_alignment for level in args.resolution_levels):
        raise ValueError(
            f"every resolution level must be divisible by {args.bucket_alignment}"
        )
    if args.resolution % args.bucket_alignment:
        raise ValueError(
            f"resolution must be divisible by {args.bucket_alignment}"
        )
    dimensions = (
        args.condition_dim, args.latent_channels, args.model_width, args.depth,
        args.heads, args.kv_heads, args.adapter_depth,
    )
    if min(dimensions) <= 0:
        raise ValueError("condition and model dimensions must be positive")
    if args.latent_channels != LATENT_CHANNELS:
        raise ValueError("Qwen/Qwen-Image VAE has 16 latent channels")
    if args.timesteps_per_image <= 0:
        raise ValueError("--timesteps-per-image must be positive")
    if not 0 <= args.encoder_prefetch_batches <= 8:
        raise ValueError("--encoder-prefetch-batches must be between 0 and 8")
    if args.observe_interval < 0 or args.sample_steps <= 0:
        raise ValueError("--observe-interval must be non-negative and --sample-steps positive")
    if not np.isfinite(args.sample_flow_shift) or args.sample_flow_shift <= 0:
        raise ValueError("--sample-flow-shift must be finite and positive")
    warped_solver_names = {
        "rf_2m_warp", "rf_er_sde_warp_1", "rf_er_sde_warp_2m",
        "rf_er_sde_warp_trust",
    }
    if args.sample_solver in warped_solver_names and args.sample_flow_shift != 1.0:
        raise ValueError(
            "warped RF solvers own their time transform and require "
            "--sample-flow-shift 1"
        )
    if args.sample_solver == "rf_trust_region" and (
        not np.isfinite(args.sample_rf_trust_lambda) or args.sample_rf_trust_lambda < 0
    ):
        raise ValueError("--sample-rf-trust-lambda must be finite and non-negative")
    if not np.isfinite(args.sample_er_sde_sigma_max) or args.sample_er_sde_sigma_max <= 0:
        raise ValueError("--sample-er-sde-sigma-max must be finite and positive")
    if not np.isfinite(args.sample_rf_2m_warp_shift) or args.sample_rf_2m_warp_shift <= 0:
        raise ValueError("--sample-rf-2m-warp-shift must be finite and positive")
    if not np.isfinite(args.sample_rf_er_sde_warp_shift) or args.sample_rf_er_sde_warp_shift <= 0:
        raise ValueError("--sample-rf-er-sde-warp-shift must be finite and positive")
    if (
        not np.isfinite(args.sample_rf_er_sde_warp_trust_lambda)
        or args.sample_rf_er_sde_warp_trust_lambda < 0
    ):
        raise ValueError(
            "--sample-rf-er-sde-warp-trust-lambda must be finite and non-negative"
        )
    if (
        not np.isfinite(args.sample_rf_er_sde_warp_trust_error_c)
        or args.sample_rf_er_sde_warp_trust_error_c <= 0
    ):
        raise ValueError(
            "--sample-rf-er-sde-warp-trust-error-c must be finite and positive"
        )
    if (
        not np.isfinite(args.sample_rf_er_sde_warp_trust_epsilon_abs)
        or args.sample_rf_er_sde_warp_trust_epsilon_abs <= 0
    ):
        raise ValueError(
            "--sample-rf-er-sde-warp-trust-epsilon-abs must be finite and positive"
        )
    if not np.isfinite(args.sample_rf_er_sde_eta) or args.sample_rf_er_sde_eta < 0:
        raise ValueError("--sample-rf-er-sde-eta must be finite and non-negative")
    if (
        not np.isfinite(args.sample_rf_er_sde_cutoff)
        or not 0 <= args.sample_rf_er_sde_cutoff < 1
    ):
        raise ValueError("--sample-rf-er-sde-cutoff must be in [0, 1)")
    if not np.isfinite(args.sample_guidance_scale) or args.sample_guidance_scale < 0:
        raise ValueError("--sample-guidance-scale must be finite and non-negative")
    if args.sample_prompt is not None and any(
        not prompt.strip() for prompt in args.sample_prompt
    ):
        raise ValueError("--sample-prompt values must not be empty")
    if args.model_width % args.heads or args.heads % args.kv_heads:
        raise ValueError("model-width must divide heads and heads must divide into kv-heads")
    if not np.isfinite(args.ff_mult) or args.ff_mult <= 0:
        raise ValueError("ff-mult must be finite and positive")
    vae_stride = 8
    required_resolution_multiple = max(
        16,
        vae_stride * max(args.latent_downsample_factor, args.target_latent_downsample_factor),
    )
    if args.resolution <= 0 or args.resolution % required_resolution_multiple:
        raise ValueError(
            "resolution must be a positive multiple of "
            f"{required_resolution_multiple} for the latent downsample factor"
        )

    seed_everything(args.seed)
    model_context = torch.device("meta") if args.dry_run or args.validate_only else torch.device("cpu")
    with model_context:
        model = NoVFCBDiT(
            qwen_dim=args.condition_dim,
            latent_channels=args.latent_channels,
            latent_downsample_factor=args.latent_downsample_factor,
            target_latent_downsample_factor=args.target_latent_downsample_factor,
            output_refinement_depth=args.output_refinement_depth,
            output_refinement_conditioning=args.output_refinement_conditioning,
            output_skip_fusion_mode=args.output_skip_fusion_mode,
            output_head_ada_scale=args.output_head_ada_scale,
            fuse_reference_latent_to_vision=args.fuse_reference_latent_to_vision,
            reference_latent_fusion_mode=args.reference_latent_fusion_mode,
            width=args.model_width,
            depth=args.depth,
            heads=args.heads,
            kv_heads=args.kv_heads,
            adapter_depth=args.adapter_depth,
            metadata_conditioning=args.metadata_conditioning,
            metadata_scale_mapping=args.metadata_scale_mapping,
            metadata_ffn_gate_mapping=args.metadata_ffn_gate_mapping,
            metadata_shift=args.metadata_shift,
            attention_head_gate=args.attention_head_gate,
            metadata_ffn_residual_gate=args.metadata_ffn_residual_gate,
            fuse_same_input_projections=args.fuse_same_input_projections,
            ff_mult=args.ff_mult,
            gradient_checkpointing=args.gradient_checkpointing,
        )
    if args.dry_run:
        run_training(
            args,
            script_name="vfp_dit.train",
            required_keys=("clean_latent",),
            model=model,
            step_fn=_flow_step,
        )
        return

    vae = encoder = None
    if not args.validate_only:
        device = resolve_device(args.device)
        qwen_device = (
            torch.device("cpu") if args.encoder_device == "cpu" else device
        )
        vae_device = torch.device("cpu") if args.vae_device == "cpu" else device
        def encoder_dtype(target_device: torch.device) -> torch.dtype:
            return (
                torch.bfloat16
                if target_device.type == "cuda" and args.amp == "bf16"
                else torch.float32
            )
        vae = load_qwen_image_vae(
            model_id=args.vae_model,
            device=vae_device,
            dtype=encoder_dtype(vae_device),
        )
        encoder = load_qwen35_encoder(
            model_id=args.vlm_model,
            device=qwen_device,
            dtype=encoder_dtype(qwen_device),
        )
        if encoder.condition_dim != args.condition_dim:
            raise ValueError(
                f"Qwen hidden width {encoder.condition_dim} does not match "
                f"--condition-dim={args.condition_dim}"
            )
        validate_condition_layer(args.condition_layer, encoder.layer_count)
    run_training(
        args,
        script_name="vfp_dit.train",
        required_keys=("clean_latent",),
        model=model,
        step_fn=_make_step_fn(vae, encoder),
        observe_fn=_make_observe_fn(vae, encoder),
        prepare_batch_fn=(
            _make_prepare_batch_fn(args, vae, encoder)
            if batch_encoder_prefetch else None
        ),
        prefetch_batches=(
            args.encoder_prefetch_batches if batch_encoder_prefetch else 0
        ),
    )


if __name__ == "__main__":
    main()
