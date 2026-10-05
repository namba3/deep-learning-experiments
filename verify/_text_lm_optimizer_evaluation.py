"""Evaluation helpers for the Text-LM optimizer convergence probe."""

from __future__ import annotations

import torch
from optimizers.factory import is_schedule_free_optimizer


def _evaluate_validation(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    eval_loader,
    device: torch.device,
    name: str,
    loss_fn,
    *,
    restore_training: bool = False,
) -> float:
    saved_parameters = None
    saved_train_modes = None
    if restore_training and is_schedule_free_optimizer(name):
        # eval()/train() is algebraically inverse but can accumulate BF16
        # round-off. Preserve the exact train-mode trajectory for the
        # diagnostic-only periodic validation path.
        saved_parameters = [
            parameter.detach().cpu().clone()
            for parameter in model.parameters()
        ]
        saved_train_modes = [
            bool(
                group["train_mode"]
                if "train_mode" in group else group["sf_train_mode"]
            )
            for group in optimizer.param_groups
        ]
    if is_schedule_free_optimizer(name):
        optimizer.eval()
    model.eval()
    eval_loss = 0.0
    eval_steps = 0
    with torch.no_grad():
        for batch in eval_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            eval_loss += float(loss_fn(model, batch))
            eval_steps += 1
    value = eval_loss / max(1, eval_steps)
    if saved_parameters is not None:
        with torch.no_grad():
            for parameter, saved in zip(model.parameters(), saved_parameters):
                parameter.copy_(
                    saved.to(device=parameter.device, dtype=parameter.dtype)
                )
            assert saved_train_modes is not None
            for group, train_mode in zip(optimizer.param_groups, saved_train_modes):
                if "train_mode" in group:
                    group["train_mode"] = train_mode
                else:
                    group["sf_train_mode"] = train_mode
        model.train()
    return value
