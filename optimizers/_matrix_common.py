"""Shared helpers for single-device matrix optimizers."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .muon_triton import apply as _triton_apply


def loss_from_closure(closure: Optional[Callable[[], float]]) -> Optional[float]:
    if closure is None:
        return None
    with torch.enable_grad():
        return closure()


def apply_decoupled_weight_decay(
    parameter: torch.Tensor, lr: float, weight_decay: float,
) -> None:
    if weight_decay:
        parameter.mul_(1.0 - lr * weight_decay)


def muon_matrix(parameter: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Flatten a Muon matrix while preserving its leading neuron dimension."""

    return grad.reshape(parameter.shape[0], -1)


def ensure_momentum_buffer(
    state, matrix: torch.Tensor, *, dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Initialize or repair Muon momentum in the parameter storage dtype."""

    target_dtype = matrix.dtype if dtype is None else dtype
    momentum = state.get("momentum_buffer")
    if momentum is None:
        momentum = state["momentum_buffer"] = torch.zeros_like(
            matrix, dtype=target_dtype,
        )
    elif momentum.dtype != target_dtype:
        momentum = state["momentum_buffer"] = momentum.to(dtype=target_dtype)
    return momentum


def apply_muon_update(
    parameter: torch.Tensor,
    update: torch.Tensor,
    *,
    lr: float,
    weight_decay: float,
    backend: str = "torch",
) -> None:
    """Apply the common Muon parameter update without extra full-size buffers."""

    update = update.reshape_as(parameter)
    if backend in {"auto", "triton"} and _triton_apply(
        parameter, update, lr=lr, weight_decay=weight_decay,
    ):
        return
    apply_decoupled_weight_decay(parameter, lr, weight_decay)
    parameter.add_(update.to(parameter.dtype), alpha=-lr)


class AdamFallback:
    """AdamW update for vectors and scalar parameters."""

    @staticmethod
    def step(parameter, grad, state, group):
        if "exp_avg" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq"] = torch.zeros_like(parameter)
        else:
            for key in ("exp_avg", "exp_avg_sq"):
                if state[key].dtype != parameter.dtype:
                    state[key] = state[key].to(dtype=parameter.dtype)
        state["step"] += 1
        beta1, beta2 = group["betas"]
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        bias1 = 1.0 - beta1 ** state["step"]
        bias2 = 1.0 - beta2 ** state["step"]
        update = (exp_avg / bias1) / (exp_avg_sq / bias2).sqrt().add_(group["eps"])
        apply_decoupled_weight_decay(parameter, group["lr"], group["weight_decay"])
        parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
