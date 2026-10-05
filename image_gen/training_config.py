"""Checkpoint-aware image generation training configuration validation."""

import sys

from image_gen.checkpoint import (
    NETWORK_CONFIG_VERSION,
    RESUME_CONFIG_KEYS,
    RESUME_CONFIG_OPTION_NAMES,
    checkpoint_info,
    resolve_resume_path,
)
from image_gen.layers import validate_attention_head_counts
from runtime.config import apply_saved_config


def prepare_training_config(args, *, argv=None):
    """Apply resume settings, normalize defaults, and validate CLI values."""
    argv = sys.argv[1:] if argv is None else argv
    if args.dry_run and args.validate_only:
        raise ValueError("--dry-run and --validate-only cannot be used together")
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint cannot be used together")
    resume_path = resolve_resume_path(args.resume, args.output_dir) if args.resume else None
    resume_config = checkpoint_info(resume_path) if resume_path else None
    if resume_config is not None:
        version = resume_config.get("network_version")
        if version != NETWORK_CONFIG_VERSION:
            raise ValueError(
                f"resume checkpoint network config version {version!r} is incompatible "
                f"with current version {NETWORK_CONFIG_VERSION!r}: {resume_path}"
            )
        overridden_config_keys = []
        restored_config_keys = apply_saved_config(
            args,
            resume_config,
            RESUME_CONFIG_OPTION_NAMES,
            argv=argv,
            keys=RESUME_CONFIG_KEYS,
            overridden_keys=overridden_config_keys,
        )
        if restored_config_keys:
            print(
                "Restored settings from checkpoint: "
                + ", ".join(restored_config_keys)
            )
        if overridden_config_keys:
            print(
                "CLI overrides checkpoint settings: "
                + ", ".join(overridden_config_keys)
            )
    if args.warmup_steps is None:
        args.warmup_steps = 0
    if args.warmup_ratio is None:
        args.warmup_ratio = 0.0
    args.performance = (
        args.performance
        or args.perf_backward_breakdown
        or args.perf_optimizer_breakdown
    )
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if args.bucket_step <= 0:
        raise ValueError("--bucket-step must be positive")
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be positive")
    if args.image_size % args.patch_size:
        raise ValueError("--image-size must be divisible by --patch-size")
    if args.time_scale <= 0:
        raise ValueError("--time-scale must be positive")
    if args.observe_interval <= 0 or args.sample_steps <= 0:
        raise ValueError("--observe-interval and --sample-steps must be positive")
    if args.gc_interval <= 0:
        raise ValueError("--gc-interval must be positive")
    if args.empty_cache_interval < 0:
        raise ValueError("--empty-cache-interval must be >= 0")
    if args.timestep_repeats <= 0:
        raise ValueError("--timestep-repeats must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.warmup_steps > 0 and args.warmup_ratio > 0.0:
        raise ValueError("use only one of --warmup-steps and --warmup-ratio")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must be in [0, 1]")
    if args.auto_schedule_target_update_ratio <= 0.0:
        raise ValueError("--auto-schedule-target-update-ratio must be positive")
    if not 0.0 < args.auto_schedule_ema_beta < 1.0:
        raise ValueError("--auto-schedule-ema-beta must be in (0, 1)")
    if args.auto_schedule_trust_alpha <= 0.0:
        raise ValueError("--auto-schedule-trust-alpha must be positive")
    if not 0.0 < args.auto_schedule_min_factor <= args.auto_schedule_max_factor:
        raise ValueError("invalid AutoSchedule factor range")
    if args.auto_schedule_max_increase < 1.0:
        raise ValueError("--auto-schedule-max-increase must be >= 1")
    if not 0.0 < args.auto_schedule_max_decrease <= 1.0:
        raise ValueError("--auto-schedule-max-decrease must be in (0, 1]")
    if not 0.0 <= args.auto_schedule_confidence_floor <= 1.0:
        raise ValueError("--auto-schedule-confidence-floor must be in [0, 1]")
    if args.auto_schedule_stability_gain < 0.0:
        raise ValueError("--auto-schedule-stability-gain must be >= 0")
    if args.auto_schedule_limiter_gain < 0.0:
        raise ValueError("--auto-schedule-limiter-gain must be >= 0")
    if args.auto_schedule_cooldown_steps < 0:
        raise ValueError("--auto-schedule-cooldown-steps must be >= 0")
    if args.auto_schedule_warmup_steps < 0:
        raise ValueError("--auto-schedule-warmup-steps must be >= 0")
    if args.linear_lr is not None and args.linear_lr <= 0:
        raise ValueError("--linear-lr must be positive")
    if args.linear_weight_decay is not None and args.linear_weight_decay < 0:
        raise ValueError("--linear-weight-decay must be >= 0")
    if args.conv_lr is not None and args.conv_lr <= 0:
        raise ValueError("--conv-lr must be positive")
    if args.conv_weight_decay is not None and args.conv_weight_decay < 0:
        raise ValueError("--conv-weight-decay must be >= 0")
    if not 0.0 <= args.linear_muon_momentum < 1.0:
        raise ValueError("--linear-muon-momentum must be in [0, 1)")
    if args.apollo_rank <= 0:
        raise ValueError("--apollo-rank must be positive")
    if args.apollo_scale <= 0 or args.apollo_mini_scale <= 0:
        raise ValueError("--apollo scales must be positive")
    if args.apollo_update_proj_gap <= 0:
        raise ValueError("--apollo-update-proj-gap must be positive")
    if args.apollo_projection_refresh_window < 0:
        raise ValueError("--apollo-projection-refresh-window must be non-negative")
    if args.apollo_projection_refresh_mode == "smooth" and args.apollo_projection_refresh_window <= 0:
        raise ValueError(
            "--apollo-projection-refresh-window must be positive for smooth mode"
        )
    if args.apollo_orthogonal_refresh_rate < 0.0:
        raise ValueError(
            "--apollo-orthogonal-refresh-rate must be non-negative"
        )
    if args.apollo_fallback_state_margin <= 0.0:
        raise ValueError("--apollo-fallback-state-margin must be positive")
    if args.apollo_fallback_min_savings_bytes < 0:
        raise ValueError(
            "--apollo-fallback-min-savings-bytes must be >= 0"
        )
    if args.rot_apollo_frequency <= 0:
        raise ValueError("--rot-apollo-frequency must be positive")
    if args.rot_apollo_rate < 0.0:
        raise ValueError("--rot-apollo-rate must be non-negative")
    if not 0.0 <= args.rot_apollo_exploration_ratio <= 1.0:
        raise ValueError("--rot-apollo-exploration-ratio must be in [0, 1]")
    if args.dual_rot_apollo_frequency <= 0:
        raise ValueError("--dual-rot-apollo-frequency must be positive")
    if args.dual_rot_apollo_rate < 0.0:
        raise ValueError("--dual-rot-apollo-rate must be non-negative")
    if not 0.0 <= args.dual_rot_apollo_exploration_ratio <= 1.0:
        raise ValueError("--dual-rot-apollo-exploration-ratio must be in [0, 1]")
    if not 0.0 <= args.dual_rot_apollo_roughness_beta < 1.0:
        raise ValueError("--dual-rot-apollo-roughness-beta must be in [0, 1)")
    if args.dual_rot_apollo_branch_temperature < 0.0:
        raise ValueError("--dual-rot-apollo-branch-temperature must be non-negative")
    if args.apollo_norm_growth_rate <= 1.0:
        raise ValueError("--apollo-norm-growth-rate must be greater than 1")
    if args.context_depth <= 0:
        raise ValueError("--context-depth must be positive")
    if args.context_heads <= 0:
        raise ValueError("--context-heads must be positive")
    if args.kv_heads <= 0:
        raise ValueError("--kv-heads must be positive")
    validate_attention_head_counts(args.heads, args.kv_heads, "MMDiT")
    if args.context_kv_heads <= 0:
        raise ValueError("--context-kv-heads must be positive")
    validate_attention_head_counts(
        args.context_heads, args.context_kv_heads, "context transformer",
    )
    if args.mhla_latent_blocks <= 0:
        raise ValueError("--mhla-latent-blocks must be positive")
    if args.mhla_image_blocks <= 0:
        raise ValueError("--mhla-image-blocks must be positive")
    if args.mhla_text_blocks <= 0:
        raise ValueError("--mhla-text-blocks must be positive")
    if args.model_dim <= 0:
        raise ValueError("--model-dim must be positive")
    if args.heads <= 0:
        raise ValueError("--heads must be positive")
    if args.model_dim % args.heads:
        raise ValueError("--model-dim must be divisible by --heads")
    if (args.model_dim // args.heads) % 4:
        raise ValueError(
            "--model-dim / --heads must be divisible by 4 for 2D/1D RoPE"
        )
    if args.init_freeze_steps < 0:
        raise ValueError("--init-freeze-steps must be >= 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.reconstruction_loss_weight < 0:
        raise ValueError("--reconstruction-loss-weight must be >= 0")
    if not 0.0 < args.max_reconstruction_contribution <= 1.0:
        raise ValueError(
            "--max-reconstruction-contribution must be in (0, 1]"
        )
    if args.text_adapter_dim <= 0:
        raise ValueError("--text-adapter-dim must be positive")
    if len(args.text_adapter_transformer_dims) != len(args.text_adapter_transformer_heads):
        raise ValueError(
            "--text-adapter-transformer-dims and --text-adapter-transformer-heads "
            "must contain the same number of values"
        )
    if len(args.text_adapter_transformer_heads) != len(
        args.text_adapter_transformer_kv_heads
    ):
        raise ValueError(
            "--text-adapter-transformer-heads and "
            "--text-adapter-transformer-kv-heads must contain the same number "
            "of values"
        )
    for dim, heads in zip(
        args.text_adapter_transformer_dims, args.text_adapter_transformer_heads,
    ):
        if dim % heads:
            raise ValueError(
                "each text transformer dim must be divisible by its head count"
            )
    for heads, kv_heads in zip(
        args.text_adapter_transformer_heads,
        args.text_adapter_transformer_kv_heads,
    ):
        validate_attention_head_counts(heads, kv_heads, "text transformer")
    if args.text_adapter_dim % args.context_heads:
        raise ValueError(
            "--text-adapter-dim must be divisible by --context-heads"
        )
    if args.text_adapter_transformer_ff_mult <= 0:
        raise ValueError("--text-adapter-transformer-ff-mult must be positive")
    if args.text_adapter_rope_theta <= 0:
        raise ValueError("--text-adapter-rope-theta must be positive")
    if not 0.0 <= args.null_conditioning_prob <= 1.0:
        raise ValueError("--null-conditioning-prob must be between 0 and 1")
    if args.resume_epoch < 0:
        raise ValueError("--resume-epoch must be >= 0")
    init_path = resolve_resume_path(args.init_checkpoint, args.output_dir) if args.init_checkpoint else None
    return resume_path, resume_config, init_path
