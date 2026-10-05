"""Parameter grouping and optimizer construction for image generation training."""

from dataclasses import dataclass

from torch import nn

from image_gen.performance import OptimizerBundle
from optimizers.factory import build_optimizer


CONV_MODULE_TYPES = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
)


@dataclass
class TrainingOptimizer:
    """Optimizer and parameter counts used by training reports."""

    optimizer: object
    linear_parameter_count: int
    conv_parameter_count: int
    main_parameter_count: int


def split_weight_parameters(modules, parameters):
    """Classify trainable weights that can use separate optimizers."""
    parameter_ids = {
        "linear": {
            id(module.weight)
            for root in modules
            for module in root.modules()
            if isinstance(module, nn.Linear) and module.weight is not None
        },
        "conv": {
            id(module.weight)
            for root in modules
            for module in root.modules()
            if isinstance(module, CONV_MODULE_TYPES) and module.weight is not None
        },
    }
    groups = {"linear": [], "conv": [], "other": []}
    for parameter in parameters:
        if id(parameter) in parameter_ids["linear"]:
            groups["linear"].append(parameter)
        elif id(parameter) in parameter_ids["conv"]:
            groups["conv"].append(parameter)
        else:
            groups["other"].append(parameter)
    return groups


def build_training_optimizer(modules, args):
    """Build configured optimizers for shared, linear, and convolution weights."""
    parameters = [parameter for module in modules for parameter in module.parameters()]
    parameter_groups = split_weight_parameters(modules, parameters)
    selected_roles = {
        "linear": args.linear_optimizer,
        "conv": args.conv_optimizer,
    }
    separated_parameters = {
        role: parameter_groups[role]
        for role, optimizer_name in selected_roles.items()
        if optimizer_name != "same"
    }
    for role, role_parameters in separated_parameters.items():
        if not role_parameters:
            raise ValueError(
                f"--{role}-optimizer was selected, but no matching "
                f"weight parameters were found"
            )

    main_parameters = list(parameter_groups["other"])
    for role, role_parameters in parameter_groups.items():
        if role in selected_roles and selected_roles[role] == "same":
            main_parameters.extend(role_parameters)

    optimizers = {}
    if main_parameters:
        optimizers["main"] = build_optimizer(
            args.optimizer, main_parameters, args,
            adamw_betas=(0.9, 0.95),
        )
    for role, role_parameters in separated_parameters.items():
        optimizers[role] = build_optimizer(
            selected_roles[role], role_parameters, args, role=role,
            adamw_betas=(0.9, 0.95),
        )
    if not optimizers:
        raise ValueError("No trainable parameters were assigned to an optimizer")
    optimizer = (
        next(iter(optimizers.values()))
        if len(optimizers) == 1
        else OptimizerBundle(optimizers)
    )

    return TrainingOptimizer(
        optimizer=optimizer,
        linear_parameter_count=sum(
            parameter.numel() for parameter in parameter_groups["linear"]
        ),
        conv_parameter_count=sum(
            parameter.numel() for parameter in parameter_groups["conv"]
        ),
        main_parameter_count=sum(parameter.numel() for parameter in main_parameters),
    )
