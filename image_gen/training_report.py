"""Training configuration summary output."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TrainingReportContext:
    """Runtime values that accompany the CLI settings in the startup report."""

    dataset: Any
    loader: Any
    scheduler: Any
    device: Any
    dtype: Any
    trainable_dtype: Any
    vae_dtype: Any
    compile_enabled: bool
    warmup_steps: int
    steps_per_epoch: int
    total_optimizer_steps: int
    latent_channels: int
    text_encoder_dim: int
    dit: Any
    linear_parameter_count: int
    conv_parameter_count: int
    main_parameter_count: int
    sample_prompts: list[str]
    performance_jsonl_path: str
    run_output_dir: str
    tensorboard_dir: str
    resume_path: str | None
    start_epoch: int
    global_step: int


def print_training_configuration(args, context):
    """Print the resolved dataset, model, optimizer, and runtime settings."""
    print("training configuration:")
    selected_split = getattr(
        context.dataset, "selected_split",
        "local" if args.records else args.dataset_split,
    )
    print(
        f"  dataset={args.dataset_name if not args.records else args.records} "
        f"split={selected_split}"
    )
    print(f"  samples={len(context.dataset)} batches_per_epoch={len(context.loader)}")
    print(
        f"  dataloader_workers={args.num_workers} "
        f"persistent_workers={args.num_workers > 0} "
        f"pin_memory={context.device.type == 'cuda'}"
    )
    print(
        f"  lr_scheduler={args.lr_scheduler} warmup_steps={context.warmup_steps} "
        f"warmup_ratio={args.warmup_ratio} min_lr_ratio={args.min_lr_ratio} "
        f"steps_per_epoch={context.steps_per_epoch} "
        f"total_optimizer_steps={context.total_optimizer_steps} "
        f"scheduled_lr={context.scheduler.format_scheduled_lrs()} "
        f"effective_lr={context.scheduler.format_effective_lrs()}"
    )
    auto_schedule_state = context.scheduler.format_auto_schedule_state()
    if auto_schedule_state:
        print(f"  auto_schedule_state={auto_schedule_state}")
    print(f"  vae={args.vae_model} vae_dtype={context.vae_dtype} text_encoder={args.text_model}")
    print(
        f"  device={context.device} amp={args.amp} compute_dtype={context.dtype} "
        f"trainable_dtype={context.trainable_dtype}"
    )
    print("  optimizer_state_dtype=fp32")
    print(
        f"  frozen_encoders=inference_mode amp="
        f"{'enabled' if context.device.type == 'cuda' and context.dtype != torch.float32 else 'disabled'}"
    )
    print(
        f"  compile={'enabled' if context.compile_enabled else 'disabled'} "
        f"mode={args.compile_mode}"
    )
    print(
        f"  gradient_checkpointing={'enabled' if args.gradient_checkpointing else 'disabled'}"
        " target=main_dit_blocks"
    )
    print(
        f"  optimizer={args.optimizer} linear_optimizer={args.linear_optimizer} "
        f"conv_optimizer={args.conv_optimizer} "
        f"lr={args.lr} weight_decay={args.weight_decay} "
        f"linear_lr={args.linear_lr if args.linear_lr is not None else 'auto'} "
        f"linear_weight_decay={args.linear_weight_decay if args.linear_weight_decay is not None else 'main'} "
        f"conv_lr={args.conv_lr if args.conv_lr is not None else 'main'} "
        f"conv_weight_decay={args.conv_weight_decay if args.conv_weight_decay is not None else 'main'} "
        "optimizer_state_estimate=enabled"
    )
    if any(
        optimizer_name in {
            "APOLLO", "APOLLO-AutoSchedule", "APOLLO-Mini",
            "APOLLO-CAME", "APOLLO-CAME-AutoSchedule",
        }
        for optimizer_name in (
            args.optimizer, args.linear_optimizer, args.conv_optimizer,
        )
    ):
        print(
            f"  apollo_rank={args.apollo_rank} apollo_scale={args.apollo_scale} "
            f"apollo_mini_scale={args.apollo_mini_scale} "
            f"apollo_update_proj_gap={args.apollo_update_proj_gap} "
            f"apollo_projection_refresh_mode={args.apollo_projection_refresh_mode} "
            f"apollo_projection_refresh_window={args.apollo_projection_refresh_window} "
            f"apollo_projection_refresh_mix={args.apollo_projection_refresh_mix} "
            f"apollo_projection_refresh_state={args.apollo_projection_refresh_state} "
            f"apollo_orthogonal_refresh_rate={args.apollo_orthogonal_refresh_rate} "
            f"apollo_fallback_1d={args.apollo_fallback} "
            f"apollo_matrix_fallback={args.apollo_matrix_fallback} "
            f"apollo_fallback_state_margin={args.apollo_fallback_state_margin} "
            f"apollo_norm_growth_limiter={not args.apollo_disable_norm_growth_limiter} "
            f"apollo_norm_growth_rate={args.apollo_norm_growth_rate}"
        )
    if args.auto_schedule or any(
        "AutoSchedule" in optimizer_name
        for optimizer_name in (
            args.optimizer, args.linear_optimizer, args.conv_optimizer,
        )
    ):
        print(
            f"  auto_schedule_target_update_ratio={args.auto_schedule_target_update_ratio} "
            f"ema_beta={args.auto_schedule_ema_beta} "
            f"trust_alpha={args.auto_schedule_trust_alpha} "
            f"factor_range=[{args.auto_schedule_min_factor}, {args.auto_schedule_max_factor}] "
            f"warmup_steps={args.auto_schedule_warmup_steps}"
        )
    if args.linear_optimizer != "same" or args.conv_optimizer != "same":
        print(
            f"  linear_weight_parameters={context.linear_parameter_count:,} "
            f"conv_weight_parameters={context.conv_parameter_count:,} "
            f"main_parameters={context.main_parameter_count:,}"
        )
    effective_batch_size = args.batch_size * args.grad_accumulation
    print(
        f"  epochs={args.epochs} batch_size={args.batch_size} "
        f"grad_accumulation={args.grad_accumulation} "
        f"timestep_repeats={args.timestep_repeats} "
        f"effective_batch_size={effective_batch_size} "
        f"effective_timestep_samples={effective_batch_size * args.timestep_repeats}"
    )
    print(
        f"  image_size={args.image_size} bucket_step={args.bucket_step} "
        f"buckets={context.dataset.bucket_shapes}"
    )
    print(
        f"  latent_channels={context.latent_channels} latent_scale={args.latent_scale} "
        f"text_encoder_dim={context.text_encoder_dim} "
        f"text_adapter_dim={args.text_adapter_dim}"
    )
    print(
        f"  text_adapter_transformer_dims={args.text_adapter_transformer_dims} "
        f"heads={args.text_adapter_transformer_heads} "
        f"kv_heads={args.text_adapter_transformer_kv_heads} "
        f"ff_mult={args.text_adapter_transformer_ff_mult} "
        f"rope_theta={args.text_adapter_rope_theta:g} mask=bidirectional"
    )
    print(
        f"  latent_input_path=GatedConv2d+GroupNorm+GatedResidualConvFFNBlock"
        f" -> image:{context.dit.image_input_dim}ch"
    )
    print("  latent_output_path=RMSNorm2d+ConvTranspose2d+ResidualConvFFNBlock")
    print(
        f"  image_context_embedder=Conv2d({context.dit.image_input_dim}->"
        f"{args.text_adapter_dim}) x3 stride=2 + GroupNorm + SiLU"
    )
    print(
        f"  context_transformer=depth={args.context_depth} "
        f"heads={args.context_heads} kv_heads={args.context_kv_heads} "
        "image_digest+text, AdaRMS=timestep+resolution"
    )
    print(
        f"  dit=MMDiT dim={args.model_dim} depth={args.depth} "
        f"heads={args.heads} kv_heads={args.kv_heads} "
        f"attention_pattern={args.attention_pattern} "
        f"attention_gate={args.attention_gate} patch_size={args.patch_size} "
        f"mhla_blocks=({args.mhla_latent_blocks},{args.mhla_image_blocks},"
        f"{args.mhla_text_blocks}) mhla_backend={args.mhla_backend} "
        f"mhla_recompute_output={args.mhla_recompute_output} "
        "streams=latent(2D)+image_digest(2D)+text(1D)"
    )
    print(
        f"  prediction={args.prediction_type} time_scale={args.time_scale} "
        f"reconstruction_loss_weight={args.reconstruction_loss_weight} "
        f"max_reconstruction_contribution={args.max_reconstruction_contribution:.2f}"
    )
    print(
        f"  null_conditioning_prob={args.null_conditioning_prob} "
        "(empty caption for CFG training)"
    )
    print("  AdaRMSNorm conditioning=timestep + latent resolution/aspect-ratio features")
    print("  augmentation=color_jitter(0.05), rotation(±2°), blur(p=0.15), gaussian_noise(std=0.01)")
    print(
        f"  observe_interval={args.observe_interval} sample_steps={args.sample_steps} "
        f"sample_prompts={len(context.sample_prompts)} gc_interval={args.gc_interval} "
        f"empty_cache_interval={args.empty_cache_interval}"
    )
    print(f"  artifacts={'disabled' if args.no_artifacts else 'enabled'}")
    print(f"  performance={'enabled' if args.performance else 'disabled'}")
    print(
        f"  backward_breakdown="
        f"{'enabled' if args.perf_backward_breakdown and not context.compile_enabled else 'disabled'}"
    )
    print(
        f"  optimizer_breakdown="
        f"{'enabled' if args.perf_optimizer_breakdown else 'disabled'}"
    )
    for index, prompt in enumerate(context.sample_prompts, start=1):
        print(f"    sample_prompt[{index}]={prompt}")
    print(f"  output={context.run_output_dir} tensorboard={context.tensorboard_dir}")
    if args.performance:
        print(f"  performance_detail={context.performance_jsonl_path}")
    if context.resume_path:
        print(
            f"  resume={context.resume_path} start_epoch={context.start_epoch} "
            f"start_step={context.global_step}"
        )
