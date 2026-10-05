"""Model construction and initialization for image generation training."""

from dataclasses import dataclass

from image_gen.checkpoint import initialize_matching_weights
from image_gen.dit import DiT
from image_gen.performance import (
    cast_trainable_modules_dtype,
    module_parameter_dtype_summary,
)
from image_gen.text_conditioning import TextConditioningAdapter


@dataclass
class TrainingModels:
    """Trainable models and parameters frozen during initialization."""

    dit: DiT
    text_adapter: TextConditioningAdapter
    init_frozen_parameter_names: set[str]


def build_training_models(
    args,
    *,
    text_encoder_dim,
    latent_channels,
    sample_latent_height,
    sample_latent_width,
    device,
    trainable_dtype,
    init_path=None,
):
    """Build trainable modules, optionally transferring matching weights."""
    text_adapter = TextConditioningAdapter(
        text_encoder_dim,
        args.text_adapter_dim,
        args.text_adapter_transformer_dims,
        args.text_adapter_transformer_heads,
        args.text_adapter_transformer_kv_heads,
        args.text_adapter_transformer_ff_mult,
        args.text_adapter_rope_theta,
        args.attention_gate == "head",
    ).to(device)
    dit = DiT(
        latent_channels, args.text_adapter_dim,
        args.model_dim, args.depth, args.heads, args.patch_size,
        int(sample_latent_height), int(sample_latent_width),
        args.context_depth, args.context_heads,
        args.gradient_checkpointing,
        args.kv_heads, args.context_kv_heads,
        args.attention_gate == "head",
        args.attention_pattern,
        args.mhla_latent_blocks,
        args.mhla_image_blocks,
        args.mhla_text_blocks,
        args.mhla_backend,
        args.mhla_recompute_output,
    ).to(device)

    init_frozen_parameter_names = set()
    if init_path:
        transferred = initialize_matching_weights(init_path, dit, text_adapter)
        trainable_modules = (
            ("dit.", dit),
            ("text_adapter.", text_adapter),
        )
        all_trainable_names = [
            prefix + name
            for prefix, module in trainable_modules
            for name, _ in module.named_parameters()
        ]
        transferred_names = set(transferred)
        not_transferred_names = [
            name for name in all_trainable_names if name not in transferred_names
        ]
        not_transferred_numel = sum(
            parameter.numel()
            for prefix, module in trainable_modules
            for name, parameter in module.named_parameters()
            if prefix + name in not_transferred_names
        )
        print(
            f"initialized {len(transferred)} matching tensors from {init_path}; "
            f"not transferred={len(not_transferred_names)} tensors "
            f"({not_transferred_numel:,} parameters)"
        )
        if args.init_freeze_steps > 0:
            transferred_set = set(transferred)
            for prefix, module in trainable_modules:
                for name, parameter in module.named_parameters():
                    if prefix + name in transferred_set:
                        parameter.requires_grad_(False)
                        init_frozen_parameter_names.add(prefix + name)
            print(
                f"freezing {len(init_frozen_parameter_names)} transferred parameters "
                f"for {args.init_freeze_steps} optimizer steps"
            )
            trainable_count = sum(
                parameter.requires_grad
                for module in (dit, text_adapter)
                for parameter in module.parameters()
            )
            if trainable_count == 0:
                for prefix, module in trainable_modules:
                    for name, parameter in module.named_parameters():
                        if prefix + name in init_frozen_parameter_names:
                            parameter.requires_grad_(True)
                init_frozen_parameter_names.clear()
                print("all trainable parameters came from init-checkpoint; disabled init freeze")

    cast_trainable_modules_dtype((dit, text_adapter), trainable_dtype)
    print(
        f"trainable parameter dtype={trainable_dtype} "
        f"({module_parameter_dtype_summary(dit, text_adapter)})"
    )
    return TrainingModels(
        dit=dit,
        text_adapter=text_adapter,
        init_frozen_parameter_names=init_frozen_parameter_names,
    )
