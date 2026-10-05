"""SOAP: Shampoo preconditioning with Adam in the eigenbasis."""

from __future__ import annotations

import torch

from ._matrix_common import (
    AdamFallback,
    apply_decoupled_weight_decay,
    loss_from_closure,
)


class SOAP(torch.optim.Optimizer):
    """Single-device SOAP with parameter-dtype full-size statistics."""

    def __init__(self, params, lr=1e-3, betas=(0.95, 0.99), eps=1e-8,
                 weight_decay=0.0, precondition_frequency=10,
                 max_preconditioner_dim=4096):
        if lr < 0 or not 0 < betas[0] < 1 or not 0 < betas[1] < 1:
            raise ValueError("invalid SOAP hyperparameters")
        if precondition_frequency < 1:
            raise ValueError("precondition_frequency must be positive")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        precondition_frequency=precondition_frequency,
                        max_preconditioner_dim=max_preconditioner_dim)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = loss_from_closure(closure)
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad.float()
                if grad.is_sparse:
                    raise RuntimeError("SOAP does not support sparse gradients")
                state = self.state[parameter]
                if parameter.ndim < 2:
                    AdamFallback.step(parameter, grad, state, group)
                    continue
                matrix = grad.reshape(parameter.shape[0], -1)
                rows, cols = matrix.shape
                if max(rows, cols) > group["max_preconditioner_dim"]:
                    raise ValueError(
                        f"SOAP preconditioner for {tuple(parameter.shape)} exceeds "
                        f"max_preconditioner_dim={group['max_preconditioner_dim']}"
                    )
                if "exp_avg" not in state:
                    state.update({
                        "step": 0,
                        "exp_avg": torch.zeros_like(
                            matrix, dtype=parameter.dtype,
                        ),
                        "exp_avg_sq": torch.zeros_like(
                            matrix, dtype=parameter.dtype,
                        ),
                        "precond_left": torch.eye(rows, device=matrix.device),
                        "precond_right": torch.eye(cols, device=matrix.device),
                        "basis_left": torch.eye(rows, device=matrix.device),
                        "basis_right": torch.eye(cols, device=matrix.device),
                        "basis_step": 0,
                    })
                else:
                    for key in ("exp_avg", "exp_avg_sq"):
                        if state[key].dtype != parameter.dtype:
                            state[key] = state[key].to(dtype=parameter.dtype)
                state["step"] += 1
                state["exp_avg"].mul_(beta1).add_(matrix, alpha=1 - beta1)
                state["precond_left"].mul_(beta2).addmm_(matrix, matrix.t(), beta=1 - beta2)
                state["precond_right"].mul_(beta2).addmm_(matrix.t(), matrix, beta=1 - beta2)
                frequency = group["precondition_frequency"]
                if state["step"] == 1 or state["step"] % frequency == 0:
                    state["basis_left"] = torch.linalg.eigh(state["precond_left"])[1]
                    state["basis_right"] = torch.linalg.eigh(state["precond_right"])[1]
                    state["basis_step"] = state["step"]
                left, right = state["basis_left"], state["basis_right"]
                # Bases remain FP32 for eigendecomposition; promote the
                # parameter-dtype moment only for this temporary GEMM.
                rotated = left.t().mm(state["exp_avg"].float()).mm(right)
                rotated_grad = left.t().mm(matrix).mm(right)
                state["exp_avg_sq"].mul_(beta2).addcmul_(rotated_grad, rotated_grad, value=1 - beta2)
                bias1 = 1 - beta1 ** state["step"]
                bias2 = 1 - beta2 ** state["step"]
                update = rotated / bias1 / (
                    state["exp_avg_sq"].float() / bias2
                ).sqrt().add_(group["eps"])
                update = left.mm(update).mm(right.t()).reshape_as(parameter)
                apply_decoupled_weight_decay(parameter, group["lr"], group["weight_decay"])
                parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
        return loss
