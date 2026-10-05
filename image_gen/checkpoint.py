"""Image-generation model checkpoint metadata and weight helpers."""

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from runtime.checkpoint import save_training_state
from runtime.config import CONFIG_SCHEMA_VERSION, validate_config_schema

# Image-latent-only architecture; semantic-channel checkpoints are incompatible.
# Version 43 fuses grouped-query Q and KV projections into one Linear.
NETWORK_CONFIG_VERSION = "43"

def checkpoint_metadata(args, epoch, global_step=None):
    config = {key: getattr(args, key) for key in (
        "dataset_name", "dataset_split", "image_size", "patch_size",
        "model_dim", "depth", "heads",
        "kv_heads", "context_depth", "context_heads", "context_kv_heads",
        "attention_gate", "attention_pattern", "mhla_latent_blocks",
        "mhla_image_blocks", "mhla_text_blocks", "mhla_backend",
        "mhla_recompute_output",
        "gradient_checkpointing",
        "trainable_dtype",
        "reconstruction_loss_weight", "max_reconstruction_contribution",
        "latent_scale", "vae_dtype", "time_scale", "prediction_type", "text_max_length",
        "latent_channels", "bucket_step", "observe_interval", "sample_steps",
        "text_model", "optimizer", "auto_schedule", "epochs", "batch_size",
        "grad_accumulation", "lr", "weight_decay", "amp", "num_workers", "seed",
        "gc_interval", "empty_cache_interval", "compile", "compile_mode",
        "init_freeze_steps", "text_adapter_dim", "text_adapter_transformer_dims",
        "text_adapter_transformer_heads", "text_adapter_transformer_kv_heads",
        "text_adapter_transformer_ff_mult",
        "text_adapter_rope_theta",
        "null_conditioning_prob", "timestep_repeats",
        "lr_scheduler", "warmup_steps", "warmup_ratio", "min_lr_ratio",
        "lr_step_size", "lr_gamma", "lr_milestones", "lr_num_cycles", "lr_power",
        "auto_schedule_target_update_ratio", "auto_schedule_ema_beta",
        "auto_schedule_trust_alpha", "auto_schedule_min_factor",
        "auto_schedule_max_factor", "auto_schedule_max_increase",
        "auto_schedule_max_decrease", "auto_schedule_confidence_floor",
        "auto_schedule_stability_gain", "auto_schedule_limiter_gain",
        "auto_schedule_cooldown_steps", "auto_schedule_warmup_steps",
        "linear_optimizer", "linear_lr", "linear_weight_decay",
        "conv_optimizer", "conv_lr", "conv_weight_decay",
        "linear_muon_momentum", "apollo_rank", "apollo_scale",
        "apollo_mini_scale", "apollo_update_proj_gap", "apollo_projection_refresh_mode",
        "apollo_projection_refresh_window", "apollo_projection_refresh_mix",
        "apollo_projection_refresh_state", "apollo_orthogonal_refresh_rate",
        "apollo_scale_front",
        "apollo_disable_norm_growth_limiter", "apollo_norm_growth_rate",
        "apollo_fallback", "apollo_matrix_fallback",
        "apollo_fallback_state_margin", "apollo_fallback_min_savings_bytes",
        "apollo_came_backend",
        "rot_apollo_frequency", "rot_apollo_rate", "rot_apollo_exploration_ratio",
        "dual_rot_apollo_frequency", "dual_rot_apollo_rate",
        "dual_rot_apollo_exploration_ratio", "dual_rot_apollo_roughness_beta",
        "dual_rot_apollo_branch_temperature",
    )}
    config["network_version"] = NETWORK_CONFIG_VERSION
    config["config_schema_version"] = CONFIG_SCHEMA_VERSION
    config["epoch"] = epoch
    if global_step is not None:
        config["global_step"] = global_step
    return {"image_gen.config": json.dumps(config, sort_keys=True)}

RESUME_CONFIG_KEYS = (
    "dataset_name", "dataset_split", "image_size", "bucket_step", "patch_size",
    "model_dim", "depth", "heads", "kv_heads", "context_depth",
    "context_heads", "context_kv_heads", "attention_gate", "attention_pattern",
    "mhla_latent_blocks", "mhla_image_blocks", "mhla_text_blocks", "mhla_backend",
    "mhla_recompute_output", "gradient_checkpointing", "trainable_dtype",
    "reconstruction_loss_weight", "max_reconstruction_contribution", "latent_scale",
    "vae_dtype", "time_scale", "prediction_type", "text_max_length",
    "latent_channels", "observe_interval", "sample_steps", "text_model",
    "optimizer", "auto_schedule", "epochs", "batch_size", "grad_accumulation",
    "timestep_repeats", "lr", "weight_decay", "amp", "num_workers", "seed",
    "gc_interval", "empty_cache_interval", "compile", "compile_mode",
    "init_freeze_steps", "text_adapter_dim", "text_adapter_transformer_dims",
    "text_adapter_transformer_heads", "text_adapter_transformer_kv_heads",
    "text_adapter_transformer_ff_mult", "text_adapter_rope_theta",
    "null_conditioning_prob", "lr_scheduler", "warmup_steps", "warmup_ratio",
    "min_lr_ratio", "lr_step_size", "lr_gamma", "lr_milestones", "lr_num_cycles",
    "lr_power", "auto_schedule_target_update_ratio", "auto_schedule_ema_beta",
    "auto_schedule_trust_alpha", "auto_schedule_min_factor", "auto_schedule_max_factor",
    "auto_schedule_max_increase", "auto_schedule_max_decrease",
    "auto_schedule_confidence_floor", "auto_schedule_stability_gain",
    "auto_schedule_limiter_gain", "auto_schedule_cooldown_steps",
    "auto_schedule_warmup_steps", "linear_optimizer", "linear_lr",
    "linear_weight_decay", "conv_optimizer", "conv_lr", "conv_weight_decay",
    "linear_muon_momentum", "apollo_rank", "apollo_scale", "apollo_mini_scale",
    "apollo_update_proj_gap", "apollo_projection_refresh_mode",
    "apollo_projection_refresh_window", "apollo_projection_refresh_mix",
    "apollo_projection_refresh_state", "apollo_orthogonal_refresh_rate",
    "apollo_scale_front",
    "apollo_disable_norm_growth_limiter",
    "apollo_norm_growth_rate", "apollo_fallback", "apollo_matrix_fallback",
    "apollo_fallback_state_margin", "apollo_fallback_min_savings_bytes",
    "apollo_came_backend", "rot_apollo_frequency", "rot_apollo_rate",
    "rot_apollo_exploration_ratio", "dual_rot_apollo_frequency", "dual_rot_apollo_rate",
    "dual_rot_apollo_exploration_ratio", "dual_rot_apollo_roughness_beta",
    "dual_rot_apollo_branch_temperature",
)

RESUME_CONFIG_OPTION_NAMES = {
    "auto_schedule": ("--auto-schedule", "--no-auto-schedule"),
    "compile": ("--compile", "--no-compile"),
    "gradient_checkpointing": (
        "--gradient-checkpointing", "--no-gradient-checkpointing",
    ),
    "apollo_scale_front": ("--apollo-scale-front", "--no-apollo-scale-front"),
    "apollo_disable_norm_growth_limiter": (
        "--apollo-disable-norm-growth-limiter",
        "--no-apollo-disable-norm-growth-limiter",
    ),
}

def save_checkpoint(
    model, text_adapter, path, args, epoch, global_step=None, *,
    resume_state=None,
):
    state = {f"dit.{key}": value.detach().cpu() for key, value in model.state_dict().items()}
    state.update({f"text_adapter.{key}": value.detach().cpu() for key, value in text_adapter.state_dict().items()})
    save_file(state, path, metadata=checkpoint_metadata(args, epoch, global_step))
    if resume_state is not None:
        save_training_state(path, resume_state)

def resolve_resume_path(path, output_dir):
    if os.path.isfile(path):
        return path
    candidate = os.path.join(output_dir, path)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"Resume checkpoint not found: {path}")

def checkpoint_info(path):
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
    try:
        config = json.loads(metadata.get("image_gen.config", "{}"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid image_gen.config metadata in {path}") from error
    validate_config_schema(config, key="image_gen.config", path=path)
    return config

def checkpoint_epoch(path):
    return int(checkpoint_info(path).get("epoch", 0))

def load_resume_weights(path, dit, text_adapter):
    state = load_file(path, device="cpu")
    dit_prefix = "dit."
    adapter_prefix = "text_adapter."
    dit_state = {
        key[len(dit_prefix):]: value for key, value in state.items()
        if key.startswith(dit_prefix)
    }
    adapter_state = {
        key[len(adapter_prefix):]: value for key, value in state.items()
        if key.startswith(adapter_prefix)
    }
    if not dit_state or not adapter_state:
        raise ValueError(f"Checkpoint does not contain image_gen model weights: {path}")
    dit.load_state_dict(dit_state)
    text_adapter.load_state_dict(adapter_state)

def initialize_matching_weights(path, dit, text_adapter):
    """Copy only same-name, same-shape tensors from an older architecture."""
    source_state = load_file(path, device="cpu")
    targets = {
        "dit.": dit,
        "text_adapter.": text_adapter,
    }
    legacy_prefixes = {
        "dit.latent_input_projection.projection.": "dit.patch.",
        "dit.latent_output_projection.projection.": "dit.final.",
    }
    transferred = []
    with torch.no_grad():
        for prefix, module in targets.items():
            target_state = module.state_dict()
            for name, target in target_state.items():
                source_key = prefix + name
                source = source_state.get(source_key)
                if source is None and source_key.endswith(".qkv.weight"):
                    legacy_prefix = source_key[:-len("qkv.weight")]
                    legacy_q = source_state.get(legacy_prefix + "q.weight")
                    legacy_kv = source_state.get(legacy_prefix + "kv.weight")
                    if legacy_q is not None and legacy_kv is not None:
                        source = torch.cat((legacy_q, legacy_kv), dim=0)
                elif source is None and source_key.endswith(".qkv.bias"):
                    legacy_prefix = source_key[:-len("qkv.bias")]
                    legacy_q = source_state.get(legacy_prefix + "q.bias")
                    legacy_kv = source_state.get(legacy_prefix + "kv.bias")
                    if legacy_q is not None and legacy_kv is not None:
                        source = torch.cat((legacy_q, legacy_kv), dim=0)
                if source is None:
                    for target_prefix, legacy_prefix in legacy_prefixes.items():
                        if source_key.startswith(target_prefix):
                            legacy_key = legacy_prefix + source_key[len(target_prefix):]
                            source = source_state.get(legacy_key)
                            if source is not None:
                                break
                if source is None or source.shape != target.shape:
                    continue
                target.copy_(source.to(device=target.device, dtype=target.dtype))
                transferred.append(prefix + name)
    if not transferred:
        raise ValueError(f"No compatible weights found in initialization checkpoint: {path}")
    return transferred
