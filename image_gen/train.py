"""Train a caption-conditioned image-latent DiT.

The VAE and text encoder are frozen. Aspect-ratio buckets make every batch
rectangular while preserving more of the original image. The VAE image latent
is transported from noise to data with Rectified Flow velocity prediction.
Self-attention uses 2D RoPE.

Example::

        # --text-model を省略すると Qwen/Qwen3.5-0.8B をText Encoderに使用

    python image_gen/train.py \
        --dataset-name lmms-lab-encoder/flickr30k \
        --vae-model Qwen/Qwen-Image \
        --output-dir image_gen/output

For a local dataset, ``--records`` accepts JSONL or CSV.  Each record needs an
image path (``image`` or ``path``) and a caption (``caption`` or ``text``).
"""

import os
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader as DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image as Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not __package__ and PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from image_gen.layers import (  # noqa: E402
    HEAD_GATE_SCALE as HEAD_GATE_SCALE,
    RMSNorm as RMSNorm,
    ResolutionEmbedding as ResolutionEmbedding,
    TimestepEmbedding as TimestepEmbedding,
    _ensure_autograd_tensor as _ensure_autograd_tensor,
    apply_rope_pairs as apply_rope_pairs,
    scaled_dot_product_attention_gqa as scaled_dot_product_attention_gqa,
    validate_attention_head_counts as validate_attention_head_counts,
    validate_attention_mask_shape as validate_attention_mask_shape,
    validate_spatial_token_grid as validate_spatial_token_grid,
)
from image_gen.dit import (  # noqa: E402
    ContextSelfAttention as ContextSelfAttention,
    ContextTransformer as ContextTransformer,
    ContextTransformerBlock as ContextTransformerBlock,
    DiT as DiT,
    GatedResidualConvFFNBlock as GatedResidualConvFFNBlock,
    ImageContextEmbedder as ImageContextEmbedder,
    JointMHLA as JointMHLA,
    LatentDownsample as LatentDownsample,
    LatentUpsample as LatentUpsample,
    MMDiTBlock as MMDiTBlock,
    ResidualConvFFNBlock as ResidualConvFFNBlock,
    SpatialRMSNorm as SpatialRMSNorm,
    context_group_norm as context_group_norm,
    group_norm as group_norm,
)
from image_gen.performance import (  # noqa: E402
    OptimizerBundle as OptimizerBundle,
    PERFORMANCE_SUMMARY_KEYS as PERFORMANCE_SUMMARY_KEYS,
    PerformanceAccumulator as PerformanceAccumulator,
    TIMING_SUMMARY_KEYS as TIMING_SUMMARY_KEYS,
    cast_trainable_modules_dtype as cast_trainable_modules_dtype,
    estimate_optimizer_state_bytes as estimate_optimizer_state_bytes,
    format_memory_bytes as format_memory_bytes,
    format_performance_summary as format_performance_summary,
    format_timing_summary as format_timing_summary,
    install_backward_timing_hooks as install_backward_timing_hooks,
    iter_optimizers as iter_optimizers,
    materialized_optimizer_state_bytes as materialized_optimizer_state_bytes,
    materialized_optimizer_state_dtype_summary as materialized_optimizer_state_dtype_summary,
    module_parameter_dtype_summary as module_parameter_dtype_summary,
    module_storage_bytes as module_storage_bytes,
    optimizer_step_with_timing as optimizer_step_with_timing,
    query_gpu_telemetry as query_gpu_telemetry,
    report_memory_estimate as report_memory_estimate,
    report_runtime_cuda_memory as report_runtime_cuda_memory,
    summarize_gpu_telemetry as summarize_gpu_telemetry,
    tensor_storage_bytes as tensor_storage_bytes,
    unique_optimizer_parameters as unique_optimizer_parameters,
    write_performance_jsonl as write_performance_jsonl,
    write_timing_jsonl as write_timing_jsonl,
)
from image_gen.inference import (  # noqa: E402
    amp_context_for as amp_context_for,
    decode_images as decode_images,
    encode_images as encode_images,
    encode_text as encode_text,
    flow_velocity_target as flow_velocity_target,
    resolve_vae_dtype as resolve_vae_dtype,
    resolve_vae_latent_scale as resolve_vae_latent_scale,
    save_flow_samples as save_flow_samples,
    save_labeled_sample_grid as save_labeled_sample_grid,
    validate_bucket_shapes_with_vae as validate_bucket_shapes_with_vae,
)
from image_gen.checkpoint import (  # noqa: E402
    NETWORK_CONFIG_VERSION as NETWORK_CONFIG_VERSION,
    RESUME_CONFIG_KEYS as RESUME_CONFIG_KEYS,
    RESUME_CONFIG_OPTION_NAMES as RESUME_CONFIG_OPTION_NAMES,
    checkpoint_epoch as checkpoint_epoch,
    checkpoint_info as checkpoint_info,
    checkpoint_metadata as checkpoint_metadata,
    initialize_matching_weights as initialize_matching_weights,
    load_resume_weights as load_resume_weights,
    resolve_resume_path as resolve_resume_path,
    save_checkpoint as save_checkpoint,
)
from image_gen.cli import (  # noqa: E402
    DEFAULT_DATASET_NAME as DEFAULT_DATASET_NAME,
    DEFAULT_TEXT_MODEL as DEFAULT_TEXT_MODEL,
    parse_args as parse_args,
    parse_int_tuple as parse_int_tuple,
)
from image_gen.training_config import (  # noqa: E402
    prepare_training_config as prepare_training_config,
)
from image_gen.training_data import (  # noqa: E402
    build_training_data,
    probe_training_latents,
)
from image_gen.training_model import build_training_models  # noqa: E402
from image_gen.training_optimizer import (  # noqa: E402
    CONV_MODULE_TYPES as CONV_MODULE_TYPES,
    build_training_optimizer,
    split_weight_parameters as split_weight_parameters,
)
from image_gen.training_resume import (  # noqa: E402
    optimizer_evaluation_mode as optimizer_evaluation_mode,
    restore_resume_optimizer_state,
    restore_training_checkpoint,
    save_training_checkpoint as save_training_checkpoint_state,
)
from image_gen.training_runtime import resolve_training_runtime  # noqa: E402
from image_gen.training_report import (  # noqa: E402
    TrainingReportContext,
    print_training_configuration,
)
from image_gen.training_components import (  # noqa: E402
    load_frozen_training_components,
)
from runtime.memory import collect_memory as collect_memory  # noqa: E402
from image_gen.attention import (  # noqa: E402
    TwoDRoPECache as TwoDRoPECache,
    TwoDRoPESelfAttention as TwoDRoPESelfAttention,
    MMDiTJointAttention as MMDiTJointAttention,
)
from image_gen.joint_attention import (  # noqa: E402
    JointMHLAAttentionMetadata as JointMHLAAttentionMetadata,
    JointMHLALayoutCache as JointMHLALayoutCache,
    joint_mhla_attention as joint_mhla_attention,
    _JointMHLATritonFunction as _JointMHLATritonFunction,
    _JointMHLATritonVectorizedBackwardFunction as _JointMHLATritonVectorizedBackwardFunction,
    _joint_mhla_block_mix as _joint_mhla_block_mix,
    _joint_mhla_naive as _joint_mhla_naive,
    _joint_mhla_vectorized as _joint_mhla_vectorized,
    _joint_mhla_block_statistics as _joint_mhla_block_statistics,
    _joint_mhla_prepare_padded_layout as _joint_mhla_prepare_padded_layout,
    _joint_mhla_triton_available as _joint_mhla_triton_available,
    _joint_mhla_triton_backward as _joint_mhla_triton_backward,
    _joint_mhla_triton_forward as _joint_mhla_triton_forward,
    _joint_mhla_triton_unavailable_reason as _joint_mhla_triton_unavailable_reason,
    triton as triton,
)
if triton is not None:
    from image_gen.joint_attention import (  # noqa: E402
        _joint_mhla_block_means_kernel as _joint_mhla_block_means_kernel,
        _joint_mhla_block_statistics_kernel as _joint_mhla_block_statistics_kernel,
        _joint_mhla_output_kernel as _joint_mhla_output_kernel,
        _joint_mhla_backward_mix_kernel as _joint_mhla_backward_mix_kernel,
        _joint_mhla_backward_summary_kernel as _joint_mhla_backward_summary_kernel,
        _joint_mhla_backward_query_kernel as _joint_mhla_backward_query_kernel,
        _joint_mhla_backward_kv_kernel as _joint_mhla_backward_kv_kernel,
    )
from image_gen.text_conditioning import (  # noqa: E402
    BidirectionalTextTransformerBlock as BidirectionalTextTransformerBlock,
    GroupedQueryProjection as GroupedQueryProjection,
    OneDRoPECache as OneDRoPECache,
    TextConditioningAdapter as TextConditioningAdapter,
)
from image_gen.data import (  # noqa: E402
    AddGaussianNoise as AddGaussianNoise,
    AspectRatioBatchSampler as AspectRatioBatchSampler,
    Flickr30KDataset as Flickr30KDataset,
    HFDataset as HFDataset,
    RecordsDataset as RecordsDataset,
    assign_bucket as assign_bucket,
    collate as collate,
    image_transform as image_transform,
    make_bucket_shapes as make_bucket_shapes,
    row_caption as row_caption,
)
from runtime.progress import RichProgress  # noqa: E402
from runtime.data import build_dataloader_options as build_dataloader_options  # noqa: E402
from runtime.metrics import write_standard_training_metrics  # noqa: E402
from runtime.preflight import build_training_preflight  # noqa: E402
from runtime.validation import ValidationTimer, build_validation_report  # noqa: E402
from runtime.run import RunRecorder  # noqa: E402
from runtime.signal import GracefulStop  # noqa: E402
from runtime.checkpoint import (  # noqa: E402
    make_training_state as make_training_state,
)
from runtime.device import resolve_device as resolve_device  # noqa: E402
from optimizers.lr_scheduler import (  # noqa: E402
    build_lr_scheduler,
)


def iterate_with_progress(iterable, progress, performance=None):
    """Iterate with a Rich progress display while timing DataLoader waits."""
    progress.__enter__()
    try:
        iterator = iter(iterable)
        while True:
            try:
                if performance is None:
                    batch = next(iterator)
                else:
                    with performance.measure_host("data_wait"):
                        batch = next(iterator)
            except StopIteration:
                return
            yield batch
    finally:
        progress.close()


# Compatibility name for callers that imported the old timing-only helper.
TimingAccumulator = PerformanceAccumulator


def apply_reconstruction_loss_cap(
    diffusion_loss,
    weighted_reconstruction_loss,
    max_contribution,
):
    """Cap reconstruction loss while keeping the cap scale gradient-free.

    For positive losses, the capped reconstruction share satisfies
    ``R / (D + R) <= max_contribution``.  The scale is derived from detached
    losses so the cap controls the objective without adding a gradient path.
    """
    if not 0.0 < max_contribution <= 1.0:
        raise ValueError("max_contribution must be in (0, 1]")
    if max_contribution < 1.0:
        scale = (
            max_contribution
            * diffusion_loss.detach()
            / (
                (1.0 - max_contribution)
                * weighted_reconstruction_loss.detach()
                + 1e-12
            )
        ).clamp(max=1.0)
    else:
        scale = torch.ones_like(weighted_reconstruction_loss)
    return weighted_reconstruction_loss * scale, scale


def main():
    args = parse_args()
    resume_path, resume_config, init_path = prepare_training_config(
        args, argv=sys.argv[1:],
    )
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    training_runtime = resolve_training_runtime(args)
    device = training_runtime.device
    dtype = training_runtime.dtype
    trainable_dtype = training_runtime.trainable_dtype
    vae_dtype = training_runtime.vae_dtype
    compile_enabled = training_runtime.compile_enabled
    run_recorder = RunRecorder(
        args.output_dir,
        script="image_gen.train",
        config=vars(args),
        run_name=args.run_name,
    )
    run_recorder.install_exception_hook()
    preflight = build_training_preflight(
        script="image_gen.train",
        device=device,
        dtype=dtype,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        resume=args.resume,
        extra={"optimizer": args.optimizer, "amp": args.amp},
    )
    run_recorder.record("preflight", **preflight)
    if args.dry_run:
        run_recorder.finish(status="dry_run")
        print("Dry run completed; no dataset or model was loaded.")
        return
    validation_timer = ValidationTimer(device) if args.validate_only else None

    frozen_components = load_frozen_training_components(args, device, vae_dtype)
    tokenizer = frozen_components.tokenizer
    text_encoder = frozen_components.text_encoder
    vae = frozen_components.vae
    text_encoder_dim = frozen_components.text_encoder_dim
    args.text_max_length = frozen_components.text_max_length
    args.latent_scale = frozen_components.latent_scale

    training_data = build_training_data(args, vae, device)
    dataset = training_data.dataset
    loader = training_data.loader
    train_batch_sampler = training_data.batch_sampler
    vae_stride = training_data.vae_stride
    steps_per_epoch = training_data.steps_per_epoch
    total_optimizer_steps = training_data.total_optimizer_steps
    warmup_steps = training_data.warmup_steps
    latent_probe = probe_training_latents(args, dataset, vae, device, dtype)
    probe = latent_probe.probe
    latent = latent_probe.latent
    sample_probe = latent_probe.sample_probe
    sample_latent = latent_probe.sample_latent
    latent_channels = latent_probe.latent_channels
    sample_latent_height = latent_probe.sample_latent_height
    sample_latent_width = latent_probe.sample_latent_width
    training_models = build_training_models(
        args,
        text_encoder_dim=text_encoder_dim,
        latent_channels=latent_channels,
        sample_latent_height=sample_latent_height,
        sample_latent_width=sample_latent_width,
        device=device,
        trainable_dtype=trainable_dtype,
        init_path=init_path,
    )
    dit = training_models.dit
    text_adapter = training_models.text_adapter
    init_frozen_parameter_names = training_models.init_frozen_parameter_names
    if args.validate_only:
        assert validation_timer is not None
        model_parameters = sum(
            parameter.numel()
            for module in (dit, text_adapter)
            for parameter in module.parameters()
        )
        trainable_parameters = sum(
            parameter.numel()
            for module in (dit, text_adapter)
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        validation = build_validation_report(
            script="image_gen.train",
            device=device,
            dtype=dtype,
            train_examples=len(dataset),
            eval_examples=None,
            model_parameters=model_parameters,
            trainable_parameters=trainable_parameters,
            steps_per_epoch=steps_per_epoch,
            measurements=validation_timer.finish(),
            extra={
                "bucket_shapes": dataset.bucket_shapes,
                "vae_stride": vae_stride,
                "latent_shape": list(latent.shape),
                "sample_latent_shape": list(sample_latent.shape),
                "latent_channels": latent_channels,
                "text_encoder_dim": text_encoder_dim,
                "optimizer": args.optimizer,
                "linear_optimizer": args.linear_optimizer,
                "conv_optimizer": args.conv_optimizer,
            },
        )
        run_recorder.record("validation", **validation)
        run_recorder.finish(status="validate_only")
        print("Validation completed; training was not started.")
        return
    training_optimizer = build_training_optimizer((dit, text_adapter), args)
    optimizer = training_optimizer.optimizer
    linear_parameter_count = training_optimizer.linear_parameter_count
    conv_parameter_count = training_optimizer.conv_parameter_count
    main_parameter_count = training_optimizer.main_parameter_count
    del probe, latent, sample_probe, sample_latent
    collect_memory(python_gc=True, empty_cache=True)
    resume_training = restore_training_checkpoint(
        args, resume_path, resume_config, dit, text_adapter, train_batch_sampler,
    )
    start_epoch = resume_training.start_epoch
    resume_global_step = resume_training.global_step
    resume_state = resume_training.state

    dit_forward = dit
    text_adapter_forward = text_adapter
    if compile_enabled:
        dit_forward = torch.compile(dit, mode=args.compile_mode, fullgraph=False)
        text_adapter_forward = torch.compile(
            text_adapter, mode=args.compile_mode, fullgraph=False,
        )
        print(
            f"torch.compile enabled: modules=dit,text_adapter mode={args.compile_mode}"
        )

    report_memory_estimate(
        device, dit, text_adapter, vae, text_encoder, optimizer,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = str(run_recorder.run_dir)
    checkpoint_dir = str(run_recorder.checkpoints_dir)
    artifact_dir = str(run_recorder.artifacts_dir)
    run_tensorboard_dir = (
        os.path.join(args.tensorboard_dir, timestamp)
        if args.tensorboard_dir is not None
        else str(run_recorder.tensorboard_dir)
    )
    os.makedirs(run_tensorboard_dir, exist_ok=True)
    performance_jsonl_path = os.path.join(artifact_dir, "performance.jsonl")
    writer = SummaryWriter(run_tensorboard_dir)
    amp_context = (
        torch.autocast("cuda", dtype=dtype)
        if device.type == "cuda" and dtype != torch.float32
        else nullcontext()
    )
    frozen_amp_context = (
        torch.autocast("cuda", dtype=dtype)
        if device.type == "cuda" and dtype != torch.float32
        else nullcontext()
    )
    stop_controller = GracefulStop(
        "SIGINT received; finishing the current batch"
        + ("; artifact saving is disabled." if args.no_artifacts else " and saving a checkpoint.")
    )
    stop_controller.install()
    memory_reported_after_first_step = False
    interrupted_checkpoint_path = None
    global_step = (
        int(resume_global_step)
        if resume_global_step is not None
        else start_epoch * steps_per_epoch
    )
    lr_scheduler = build_lr_scheduler(
        optimizer, args, total_optimizer_steps,
    )
    restore_resume_optimizer_state(
        resume_state, optimizer, lr_scheduler, global_step,
    )

    def save_training_checkpoint(path, epoch, step):
        save_training_checkpoint_state(
            path,
            epoch,
            step,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            batch_sampler=train_batch_sampler,
            dit=dit,
            text_adapter=text_adapter,
            args=args,
        )
    performance = PerformanceAccumulator(
        device,
        enabled=args.performance,
        optimizer_breakdown=args.perf_optimizer_breakdown,
    )
    backward_timing_handles = (
        install_backward_timing_hooks(dit, performance)
        if args.perf_backward_breakdown and not compile_enabled
        else []
    )
    # Keep optional stage instrumentation outside torch.compile.  Passing a
    # mutable timing object through a compiled graph would create graph breaks
    # and would make the reported numbers less representative.
    dit_timing = performance if performance.enabled and not compile_enabled else None
    sample_prompts = args.sample_prompt or [
        "a photograph of a child playing with a dog outdoors",
        "a photograph of a person riding a bicycle on a city street",
        "a photograph of a horse running across a grassy field",
        "a photograph of people and dogs relaxing in a park",
    ]
    print_training_configuration(
        args,
        TrainingReportContext(
            dataset=dataset,
            loader=loader,
            scheduler=lr_scheduler,
            device=device,
            dtype=dtype,
            trainable_dtype=trainable_dtype,
            vae_dtype=vae_dtype,
            compile_enabled=compile_enabled,
            warmup_steps=warmup_steps,
            steps_per_epoch=steps_per_epoch,
            total_optimizer_steps=total_optimizer_steps,
            latent_channels=latent_channels,
            text_encoder_dim=text_encoder_dim,
            dit=dit,
            linear_parameter_count=linear_parameter_count,
            conv_parameter_count=conv_parameter_count,
            main_parameter_count=main_parameter_count,
            sample_prompts=sample_prompts,
            performance_jsonl_path=performance_jsonl_path,
            run_output_dir=run_output_dir,
            tensorboard_dir=run_tensorboard_dir,
            resume_path=resume_path,
            start_epoch=start_epoch,
            global_step=global_step,
        ),
    )
    for epoch in range(start_epoch, args.epochs):
        epoch_started_at = time.perf_counter()
        epoch_start_global_step = global_step
        if train_batch_sampler.epoch != epoch:
            train_batch_sampler.set_epoch(epoch)
        steps_in_epoch = len(loader)
        if init_frozen_parameter_names and global_step >= args.init_freeze_steps:
            for prefix, module in (
                ("dit.", dit),
                ("text_adapter.", text_adapter),
            ):
                for name, parameter in module.named_parameters():
                    if prefix + name in init_frozen_parameter_names:
                        parameter.requires_grad_(True)
            init_frozen_parameter_names.clear()
            print(f"unfroze transferred parameters at optimizer step {global_step}")
        dit.train()
        text_adapter.train()
        if hasattr(optimizer, "train"):
            optimizer.train()
        total_loss = 0.0
        total_diffusion_loss = 0.0
        total_image_reconstruction_loss = 0.0
        total_reconstruction_loss = 0.0
        total_pre_cap_weighted_reconstruction_loss = 0.0
        total_weighted_reconstruction_loss = 0.0
        total_image_nrmse = 0.0
        total_image_x0_nrmse = 0.0
        optimizer.zero_grad(set_to_none=True)
        progress = RichProgress(
            total=steps_in_epoch,
            description=f"epoch {epoch + 1}/{args.epochs}",
        )
        initial_status = {
            "loss": "waiting for first batch",
            "lr scheduled": lr_scheduler.format_scheduled_lrs(),
            "lr effective": lr_scheduler.format_effective_lrs(),
        }
        auto_schedule_state = lr_scheduler.format_auto_schedule_state()
        if auto_schedule_state:
            initial_status["controller"] = auto_schedule_state
        initial_status.update({"mse": "v=- x0=-", "nrmse": "v=- x0=-", "observe": "-"})
        progress.set_status(initial_status)
        optimizer_step_started_at = None
        last_step_time = "-"
        for batch_index, (images, captions) in enumerate(
            iterate_with_progress(loader, progress, performance),
        ):
            if batch_index % args.grad_accumulation == 0:
                optimizer_step_started_at = time.perf_counter()
            completed_optimizer_step = False
            with performance.measure_host("host_to_device"):
                images = images.to(device, non_blocking=True)
            with performance.measure_host("batch_prepare"):
                null_conditioning_mask = torch.rand(images.shape[0], device=device) < args.null_conditioning_prob
                conditioning_captions = [
                    "" if null_conditioning_mask[index].item() else caption
                    for index, caption in enumerate(captions)
                ]
            with torch.inference_mode():
                with frozen_amp_context:
                    with performance.measure("vae_encode"):
                        latents = encode_images(vae, images, args.latent_scale)
                    with performance.measure("text_encoder"):
                        text_hidden_states, text_condition_mask = encode_text(
                            tokenizer, text_encoder, conditioning_captions, device, args.text_max_length,
                        )
            # inference_mode tensors cannot be saved for backward.  Make normal
            # tensors before feeding frozen outputs into trainable modules/losses.
            latents = latents.detach().clone()
            text_hidden_states = text_hidden_states.detach().clone()
            text_condition_mask = text_condition_mask.detach().clone()
            clean = latents
            batch_loss = 0.0
            batch_diffusion_loss = 0.0
            batch_image_reconstruction_loss = 0.0
            batch_reconstruction_loss = 0.0
            batch_pre_cap_weighted_reconstruction_loss = 0.0
            batch_weighted_reconstruction_loss = 0.0
            batch_image_velocity_target_mse = 0.0
            # Text conditioning is independent of the sampled timestep.  Run
            # the adapter once, then accumulate gradients at its output while
            # the DiT handles each timestep repeat.  The accumulated output
            # gradient is propagated through the adapter once after all
            # repeats.  This avoids both repeated adapter forwards and
            # repeated adapter backwards without dropping adapter gradients.
            with amp_context:
                with performance.measure("text_adapter"):
                    text_condition_tokens = text_adapter_forward(
                        text_hidden_states, text_condition_mask,
                    )
            text_condition_tokens_for_dit = text_condition_tokens.detach()
            if text_condition_tokens.requires_grad:
                text_condition_tokens_for_dit.requires_grad_(True)
            for _ in range(args.timestep_repeats):
                timestep = torch.rand(images.shape[0], device=device)
                noise = torch.randn_like(clean)
                velocity_target = flow_velocity_target(
                    clean, noise, args.prediction_type,
                )
                batch_image_velocity_target_mse += (
                    velocity_target.float().square().mean().detach()
                )
                time_value = timestep[:, None, None, None]
                noisy = (1 - time_value) * clean + time_value * noise
                with amp_context:
                    with performance.measure("dit_forward_and_loss"):
                        prediction = dit_forward(
                            noisy, timestep * args.time_scale,
                            text_condition_tokens_for_dit, text_condition_mask,
                            timing=dit_timing,
                        )
                        diffusion_loss = F.mse_loss(prediction, velocity_target)
                        x0_prediction = noisy - time_value * prediction
                        reconstruction_loss = F.mse_loss(
                            x0_prediction.float(), latents.float(),
                        )
                        weighted_reconstruction_loss = (
                            args.reconstruction_loss_weight * reconstruction_loss
                        )
                        capped_weighted_reconstruction_loss, _ = apply_reconstruction_loss_cap(
                            diffusion_loss,
                            weighted_reconstruction_loss,
                            args.max_reconstruction_contribution,
                        )
                        loss = (
                            diffusion_loss + capped_weighted_reconstruction_loss
                        ) / (args.grad_accumulation * args.timestep_repeats)
                with performance.measure("backward"):
                    loss.backward()
                # Keep the scalar on-device until all repeats finish; calling
                # item() here would synchronize the GPU once per timestep.
                batch_loss += loss.detach()
                batch_diffusion_loss += diffusion_loss.detach()
                batch_image_reconstruction_loss += reconstruction_loss.detach()
                batch_reconstruction_loss += reconstruction_loss.detach()
                batch_pre_cap_weighted_reconstruction_loss += (
                    weighted_reconstruction_loss.detach()
                )
                batch_weighted_reconstruction_loss += capped_weighted_reconstruction_loss.detach()
            if text_condition_tokens.requires_grad:
                output_gradient = text_condition_tokens_for_dit.grad
                if output_gradient is not None:
                    with performance.measure("backward"):
                        text_condition_tokens.backward(output_gradient)
            if (batch_index + 1) % args.grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(dit.parameters()) + list(text_adapter.parameters()),
                    1.0,
                )
                lr_scheduler.step(global_step + 1)
                optimizer_step_with_timing(optimizer, performance)
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                performance.step_completed()
                completed_optimizer_step = True
                if not memory_reported_after_first_step:
                    report_runtime_cuda_memory(
                        device, optimizer, label="after first optimizer step",
                    )
                    memory_reported_after_first_step = True
                if init_frozen_parameter_names and global_step >= args.init_freeze_steps:
                    for prefix, module in (
                        ("dit.", dit),
                        ("text_adapter.", text_adapter),
                    ):
                        for name, parameter in module.named_parameters():
                            if prefix + name in init_frozen_parameter_names:
                                parameter.requires_grad_(True)
                    init_frozen_parameter_names.clear()
                    print(f"unfroze transferred parameters at optimizer step {global_step}")
            batch_loss_value = batch_loss.item()
            batch_diffusion_loss_value = batch_diffusion_loss.item() / args.timestep_repeats
            batch_image_reconstruction_loss_value = (
                batch_image_reconstruction_loss.item() / args.timestep_repeats
            )
            batch_reconstruction_loss_value = (
                batch_reconstruction_loss.item() / args.timestep_repeats
            )
            batch_pre_cap_weighted_reconstruction_loss_value = (
                batch_pre_cap_weighted_reconstruction_loss.item()
                / args.timestep_repeats
            )
            batch_weighted_reconstruction_loss_value = (
                batch_weighted_reconstruction_loss.item() / args.timestep_repeats
            )
            batch_image_target_mse_value = (
                batch_image_velocity_target_mse.item() / args.timestep_repeats
            )
            image_x0_target_mse_value = latents.float().square().mean().item()
            batch_image_nrmse_value = (
                batch_diffusion_loss_value
                / max(batch_image_target_mse_value, 1e-12)
            ) ** 0.5
            batch_image_x0_nrmse_value = (
                batch_image_reconstruction_loss_value
                / max(image_x0_target_mse_value, 1e-12)
            ) ** 0.5
            if completed_optimizer_step and optimizer_step_started_at is not None:
                last_step_time = f"{(time.perf_counter() - optimizer_step_started_at) * 1000.0:.1f}ms"
            total_loss += batch_loss_value * args.grad_accumulation
            total_diffusion_loss += batch_diffusion_loss_value * args.grad_accumulation
            total_image_reconstruction_loss += (
                batch_image_reconstruction_loss_value * args.grad_accumulation
            )
            total_reconstruction_loss += batch_reconstruction_loss_value * args.grad_accumulation
            total_pre_cap_weighted_reconstruction_loss += (
                batch_pre_cap_weighted_reconstruction_loss_value
                * args.grad_accumulation
            )
            total_weighted_reconstruction_loss += (
                batch_weighted_reconstruction_loss_value * args.grad_accumulation
            )
            total_image_nrmse += batch_image_nrmse_value * args.grad_accumulation
            total_image_x0_nrmse += batch_image_x0_nrmse_value * args.grad_accumulation
            batch_objective_value = batch_loss_value * args.grad_accumulation
            batch_pre_cap_objective_value = (
                batch_diffusion_loss_value
                + batch_pre_cap_weighted_reconstruction_loss_value
            )
            batch_reconstruction_component_value = batch_weighted_reconstruction_loss_value
            diffusion_contribution = (
                100.0 * batch_diffusion_loss_value
                / max(abs(batch_objective_value), 1e-12)
            )
            pre_cap_reconstruction_contribution = (
                100.0 * batch_pre_cap_weighted_reconstruction_loss_value
                / max(abs(batch_pre_cap_objective_value), 1e-12)
            )
            reconstruction_contribution = (
                100.0 * batch_reconstruction_component_value
                / max(abs(batch_objective_value), 1e-12)
            )
            next_observe = (
                (global_step // args.observe_interval) + 1
            ) * args.observe_interval
            progress.update(advance=1, step_time=last_step_time)
            status = {
                "loss": (
                    f"{batch_objective_value:.4f}  "
                    f"d={diffusion_contribution:.1f}% "
                    f"r={reconstruction_contribution:.1f}%"
                ),
                "lr scheduled": lr_scheduler.format_scheduled_lrs(),
                "lr effective": lr_scheduler.format_effective_lrs(),
            }
            auto_schedule_state = lr_scheduler.format_auto_schedule_state()
            if auto_schedule_state:
                status["controller"] = auto_schedule_state
            status.update(
                {
                    "contribution": (
                        f"d={diffusion_contribution:.1f}% "
                        f"r_pre={pre_cap_reconstruction_contribution:.1f}% "
                        f"r_post={reconstruction_contribution:.1f}%"
                    ),
                    "reconstruction": (
                        f"weighted={batch_pre_cap_weighted_reconstruction_loss_value:.4f}"
                        f" -> {batch_weighted_reconstruction_loss_value:.4f}"
                    ),
                    "mse": (
                        f"v={batch_diffusion_loss_value:.4f} "
                        f"x0={batch_image_reconstruction_loss_value:.4f}"
                    ),
                    "nrmse": (
                        f"v={batch_image_nrmse_value:.3f} "
                        f"x0={batch_image_x0_nrmse_value:.3f}"
                    ),
                    "observe": f"in {max(next_observe - global_step, 0)} optimizer steps",
                }
            )
            progress.set_status(status)
            train_batch_sampler.set_position(train_batch_sampler.position + 1)
            if stop_controller.requested:
                if args.no_artifacts:
                    print(
                        f"interrupted safely after batch {batch_index + 1}; "
                        "artifact saving disabled"
                    )
                else:
                    interrupted_path = os.path.join(
                        checkpoint_dir, "checkpoint_latest.safetensors",
                    )
                    save_training_checkpoint(interrupted_path, epoch, global_step)
                    interrupted_checkpoint_path = interrupted_path
                    print(
                        f"interrupted safely after batch {batch_index + 1}; "
                        f"saved={interrupted_path}"
                    )
                break
            should_gc = global_step > 0 and global_step % args.gc_interval == 0
            should_empty_cache = (
                args.empty_cache_interval > 0
                and global_step > 0
                and global_step % args.empty_cache_interval == 0
            )
            should_observe = (
                global_step > 0
                and global_step % args.observe_interval == 0
            )
            if should_observe:
                auto_schedule_state = lr_scheduler.format_auto_schedule_state()
                print(
                    f"learning rates at optimizer step {global_step}: "
                    f"scheduled={lr_scheduler.format_scheduled_lrs()} "
                    f"effective={lr_scheduler.format_effective_lrs()} "
                    + (f"controller={auto_schedule_state}" if auto_schedule_state else "")
                )
            observe_cleanup = should_observe and not args.no_artifacts
            if should_gc or should_empty_cache or observe_cleanup:
                del prediction, diffusion_loss, reconstruction_loss, x0_prediction
                del (
                    noisy, noise, velocity_target, clean,
                    text_hidden_states, text_condition_mask,
                    text_condition_tokens, latents, images,
                    timestep, time_value,
                    batch_loss, batch_diffusion_loss,
                    batch_image_reconstruction_loss, batch_reconstruction_loss,
                    batch_pre_cap_weighted_reconstruction_loss,
                    batch_weighted_reconstruction_loss,
                    batch_image_velocity_target_mse,
                    null_conditioning_mask,
                )
                collect_memory(
                    python_gc=should_gc or observe_cleanup,
                    empty_cache=should_empty_cache or observe_cleanup,
                )
            if should_observe:
                if args.performance:
                    performance_report = performance.report_and_reset()
                    write_performance_jsonl(
                        performance_jsonl_path, performance_report,
                        epoch=epoch, global_step=global_step,
                    )
                    print(
                        "performance per optimizer step: "
                        + format_performance_summary(performance_report)
                    )
                if not args.no_artifacts:
                    step_checkpoint_path = os.path.join(
                        checkpoint_dir, "checkpoint_latest.safetensors",
                    )
                    sample_path = os.path.join(artifact_dir, "sample_latest.png")
                    with optimizer_evaluation_mode(optimizer):
                        save_flow_samples(
                            dit, text_adapter, vae, tokenizer, text_encoder, device, dtype,
                            latent_channels,
                            sample_latent_height, sample_latent_width,
                            args.latent_scale, sample_prompts, args.text_max_length, args.sample_steps,
                            args.time_scale, sample_path,
                        )
                    save_training_checkpoint(step_checkpoint_path, epoch, global_step)
                    # The sampler/VAE decode can use different CUDA workspaces than
                    # the training batch.  Release those cached blocks before the
                    # next training batch starts.
                    collect_memory(python_gc=True, empty_cache=True)
                    performance.reset_memory_stats()
                    print(
                        f"saved step artifacts: step={global_step} "
                        f"checkpoint={step_checkpoint_path} sample={sample_path}"
                    )
        progress.close()
        if not stop_controller.requested:
            remainder = steps_in_epoch % args.grad_accumulation
            if remainder:
                # Each microbatch loss was divided by the full accumulation
                # count.  Rescale the remaining gradients so this final
                # optimizer step has the same mean-gradient convention as a
                # complete accumulation window instead of dropping them.
                correction = args.grad_accumulation / remainder
                for parameter in (
                    list(dit.parameters())
                    + list(text_adapter.parameters())
                ):
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
                torch.nn.utils.clip_grad_norm_(
                    list(dit.parameters()) + list(text_adapter.parameters()),
                    1.0,
                )
                lr_scheduler.step(global_step + 1)
                optimizer_step_with_timing(optimizer, performance)
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                performance.step_completed()
                if optimizer_step_started_at is not None:
                    last_step_time = f"{(time.perf_counter() - optimizer_step_started_at) * 1000.0:.1f}ms"
                if not memory_reported_after_first_step:
                    report_runtime_cuda_memory(
                        device, optimizer, label="after first optimizer step",
                    )
                    memory_reported_after_first_step = True
                if init_frozen_parameter_names and global_step >= args.init_freeze_steps:
                    for prefix, module in (
                        ("dit.", dit),
                        ("text_adapter.", text_adapter),
                    ):
                        for name, parameter in module.named_parameters():
                            if prefix + name in init_frozen_parameter_names:
                                parameter.requires_grad_(True)
                    init_frozen_parameter_names.clear()
                    print(f"unfroze transferred parameters at optimizer step {global_step}")
                if (
                    global_step > 0
                    and global_step % args.observe_interval == 0
                ):
                    auto_schedule_state = lr_scheduler.format_auto_schedule_state()
                    print(
                        f"learning rates at optimizer step {global_step}: "
                        f"scheduled={lr_scheduler.format_scheduled_lrs()} "
                        f"effective={lr_scheduler.format_effective_lrs()} "
                        + (f"controller={auto_schedule_state}" if auto_schedule_state else "")
                    )
                    if args.performance:
                        performance_report = performance.report_and_reset()
                        write_performance_jsonl(
                            performance_jsonl_path, performance_report,
                            epoch=epoch, global_step=global_step,
                        )
                        print(
                            "performance per optimizer step: "
                            + format_performance_summary(performance_report)
                        )
                    del prediction, diffusion_loss, reconstruction_loss, x0_prediction
                    del (
                        noisy, noise, velocity_target, clean,
                        text_hidden_states, text_condition_mask,
                        text_condition_tokens, latents, images,
                        timestep, time_value,
                        batch_loss, batch_diffusion_loss,
                        batch_image_reconstruction_loss, batch_reconstruction_loss,
                        batch_pre_cap_weighted_reconstruction_loss,
                        batch_weighted_reconstruction_loss,
                        batch_image_velocity_target_mse,
                        null_conditioning_mask,
                    )
                    if not args.no_artifacts:
                        collect_memory(python_gc=True, empty_cache=True)
                        step_checkpoint_path = os.path.join(
                            checkpoint_dir, "checkpoint_latest.safetensors",
                        )
                        sample_path = os.path.join(artifact_dir, "sample_latest.png")
                        with optimizer_evaluation_mode(optimizer):
                            save_flow_samples(
                                dit, text_adapter, vae, tokenizer, text_encoder, device, dtype,
                                latent_channels,
                                sample_latent_height, sample_latent_width,
                                args.latent_scale, sample_prompts, args.text_max_length, args.sample_steps,
                                args.time_scale, sample_path,
                            )
                        save_training_checkpoint(step_checkpoint_path, epoch, global_step)
                        collect_memory(python_gc=True, empty_cache=True)
                        performance.reset_memory_stats()
                        print(
                            f"saved step artifacts: step={global_step} "
                            f"checkpoint={step_checkpoint_path} sample={sample_path}"
                        )
                print(
                    f"applied final partial gradient accumulation: "
                    f"batches={remainder} global_step={global_step}"
                )
        if stop_controller.requested:
            for handle in backward_timing_handles:
                handle.remove()
            writer.close()
            print(f"training stopped; resume from={run_output_dir}")
            stop_controller.restore()
            run_recorder.finish(
                status="interrupted",
                checkpoints=(
                    [interrupted_checkpoint_path]
                    if interrupted_checkpoint_path is not None
                    else []
                ),
            )
            return
        mean_loss = total_loss / max(steps_in_epoch, 1)
        mean_diffusion_loss = total_diffusion_loss / max(steps_in_epoch, 1)
        mean_image_reconstruction_loss = (
            total_image_reconstruction_loss / max(steps_in_epoch, 1)
        )
        mean_reconstruction_loss = total_reconstruction_loss / max(steps_in_epoch, 1)
        mean_pre_cap_weighted_reconstruction_loss = (
            total_pre_cap_weighted_reconstruction_loss / max(steps_in_epoch, 1)
        )
        mean_weighted_reconstruction_loss = (
            total_weighted_reconstruction_loss / max(steps_in_epoch, 1)
        )
        mean_image_nrmse = total_image_nrmse / max(steps_in_epoch, 1)
        mean_image_x0_nrmse = total_image_x0_nrmse / max(steps_in_epoch, 1)
        weighted_reconstruction_loss = mean_weighted_reconstruction_loss
        loss_components = {
            "diffusion": mean_diffusion_loss,
            "reconstruction": weighted_reconstruction_loss,
        }
        writer.add_scalar("train/loss", mean_loss, epoch + 1)
        writer.add_scalar("train/loss/diffusion", mean_diffusion_loss, epoch + 1)
        writer.add_scalar("train/loss/diffusion_image", mean_diffusion_loss, epoch + 1)
        writer.add_scalar(
            "train/loss/reconstruction", mean_reconstruction_loss, epoch + 1,
        )
        writer.add_scalar(
            "train/loss/reconstruction_image", mean_image_reconstruction_loss, epoch + 1,
        )
        writer.add_scalar(
            "train/loss/reconstruction_weighted", weighted_reconstruction_loss, epoch + 1,
        )
        writer.add_scalar(
            "train/loss/reconstruction_weighted_pre_cap",
            mean_pre_cap_weighted_reconstruction_loss,
            epoch + 1,
        )
        writer.add_scalar(
            "train/loss/reconstruction_weighted_post_cap",
            weighted_reconstruction_loss,
            epoch + 1,
        )
        writer.add_scalar("train/nrmse/image_velocity", mean_image_nrmse, epoch + 1)
        writer.add_scalar("train/nrmse/image_x0", mean_image_x0_nrmse, epoch + 1)
        for name, value in loss_components.items():
            writer.add_scalar(
                f"contribution/train/loss/{name}",
                100.0 * value / max(abs(mean_loss), 1e-12),
                epoch + 1,
            )
        writer.add_scalar(
            "contribution/train/loss/reconstruction_image",
            100.0 * weighted_reconstruction_loss
            / max(abs(mean_loss), 1e-12),
            epoch + 1,
        )
        pre_cap_objective = (
            mean_diffusion_loss + mean_pre_cap_weighted_reconstruction_loss
        )
        pre_cap_reconstruction_contribution = (
            100.0 * mean_pre_cap_weighted_reconstruction_loss
            / max(abs(pre_cap_objective), 1e-12)
        )
        writer.add_scalar(
            "contribution/train/loss/reconstruction_pre_cap",
            pre_cap_reconstruction_contribution,
            epoch + 1,
        )
        writer.add_scalar(
            "contribution/train/loss/reconstruction_post_cap",
            reconstruction_contribution,
            epoch + 1,
        )
        effective_lr = float(optimizer.param_groups[0]["lr"])
        scheduled_lr = float(
            optimizer.param_groups[0].get("scheduled_lr", effective_lr)
        )
        write_standard_training_metrics(
            writer,
            step=epoch + 1,
            train_loss=mean_loss,
            learning_rate=effective_lr,
            scheduled_learning_rate=scheduled_lr,
        )
        epoch_elapsed = time.perf_counter() - epoch_started_at
        optimizer_steps = max(1, global_step - epoch_start_global_step)
        run_recorder.record_training_step(
            global_step=global_step,
            epoch=epoch + 1,
            train_loss=mean_loss,
            effective_lr=effective_lr,
            scheduled_lr=scheduled_lr,
            step_time_sec=epoch_elapsed / optimizer_steps,
            steps_per_second=optimizer_steps / max(epoch_elapsed, 1e-6),
            metrics={
                "diffusion_loss": mean_diffusion_loss,
                "reconstruction_loss": mean_reconstruction_loss,
                "image_velocity_nrmse": mean_image_nrmse,
                "image_x0_nrmse": mean_image_x0_nrmse,
            },
        )
        if not args.no_artifacts:
            train_batch_sampler.set_epoch(epoch + 1)
            latest = os.path.join(checkpoint_dir, "checkpoint_latest.safetensors")
            interrupted_checkpoint_path = latest
            save_training_checkpoint(latest, epoch + 1, global_step)
        diffusion_contribution = 100.0 * mean_diffusion_loss / max(abs(mean_loss), 1e-12)
        reconstruction_contribution = (
            100.0 * weighted_reconstruction_loss / max(abs(mean_loss), 1e-12)
        )
        print(
            f"epoch {epoch + 1:03d}/{args.epochs}: "
            f"loss={mean_loss:.6f} "
            f"diffusion={mean_diffusion_loss:.6f} ({diffusion_contribution:.1f}%) "
            f"reconstruction={mean_reconstruction_loss:.6f} "
            f"weighted={mean_pre_cap_weighted_reconstruction_loss:.6f}"
            f" -> {weighted_reconstruction_loss:.6f} "
            f"(pre={pre_cap_reconstruction_contribution:.1f}% "
            f"post={reconstruction_contribution:.1f}%) "
            f"nrmse=(v={mean_image_nrmse:.3f} x0={mean_image_x0_nrmse:.3f}) "
            + (f"saved={latest}" if not args.no_artifacts else "artifacts=disabled")
        )
        if stop_controller.requested:
            break
    for handle in backward_timing_handles:
        handle.remove()
    writer.close()
    stop_controller.restore()
    if stop_controller.requested:
        run_recorder.finish(
            status="interrupted",
            checkpoints=(
                [interrupted_checkpoint_path]
                if interrupted_checkpoint_path is not None
                else []
            ),
        )
        return
    run_recorder.finish(
        checkpoints=[]
        if args.no_artifacts
        else [os.path.join(checkpoint_dir, "checkpoint_latest.safetensors")],
    )
    print(f"training finished; output={run_output_dir} tensorboard={run_tensorboard_dir}")

if __name__ == "__main__":
    main()
