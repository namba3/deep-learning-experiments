"""Schedule-Free optimizers with parameter-dtype full-size state.

The upstream ``schedulefree`` implementations allocate ``z`` and
``exp_avg_sq`` with the parameter dtype.  These local variants preserve that
storage contract and cast only temporary arithmetic to FP32 when needed.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from schedulefree import (
    AdamWScheduleFree as _AdamWScheduleFree,
    RAdamScheduleFree as _RAdamScheduleFree,
)

from .schedulefree_triton import apply as _triton_apply
from .schedulefree_triton import apply_radam as _triton_apply_radam


class _ParameterDtypeStateScheduleFreeMixin:
    """Common parameter-dtype state initialization for Schedule-Free."""

    @staticmethod
    def _ensure_parameter_dtype_state(parameter, state):
        if "z" not in state:
            state["z"] = parameter.detach().clone(
                memory_format=torch.preserve_format,
            )
        elif state["z"].dtype != parameter.dtype:
            state["z"] = state["z"].to(dtype=parameter.dtype)

        if "exp_avg_sq" not in state:
            state["exp_avg_sq"] = torch.zeros_like(
                parameter, memory_format=torch.preserve_format,
            )
        elif state["exp_avg_sq"].dtype != parameter.dtype:
            state["exp_avg_sq"] = state["exp_avg_sq"].to(dtype=parameter.dtype)

    @staticmethod
    def _copy_updated_parameter(parameter, value):
        parameter.copy_(value.to(dtype=parameter.dtype))


class AdamWScheduleFree(_ParameterDtypeStateScheduleFreeMixin, _AdamWScheduleFree):
    """Schedule-Free AdamW with parameter-dtype full-size state."""

    def __init__(
        self, params, lr=0.0025, betas=(0.9, 0.999), eps=1e-8,
        weight_decay: float = 0.0, warmup_steps=0, r=0.0, weight_lr_power=2.0,
        foreach=True, *, backend="auto",
    ):
        if backend not in {"auto", "torch", "triton"}:
            raise ValueError("backend must be 'auto', 'torch', or 'triton'")
        super().__init__(
            params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, warmup_steps=warmup_steps,
            r=r, weight_lr_power=weight_lr_power, foreach=foreach,
        )
        self.backend = backend

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        if not self.param_groups[0]["train_mode"]:
            raise RuntimeError(
                "Optimizer was not in train mode when step is called. "
                "Call optimizer.train() before training."
            )

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            decay = group["weight_decay"]
            k = group["k"]
            r = group["r"]
            warmup_steps = group["warmup_steps"]
            weight_lr_power = group["weight_lr_power"]

            sched = (k + 1) / warmup_steps if k < warmup_steps else 1.0
            bias_correction2 = 1 - beta2 ** (k + 1)
            lr = group["lr"] * sched
            group["scheduled_lr"] = lr
            lr_max = group["lr_max"] = max(lr, group["lr_max"])
            weight = ((k + 1) ** r) * (lr_max ** weight_lr_power)
            weight_sum = group["weight_sum"] = group["weight_sum"] + weight
            ckp1 = weight / weight_sum if weight_sum != 0 else 0.0

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError(
                        "AdamWScheduleFree does not support sparse gradients"
                    )
                state = self.state[parameter]
                self._ensure_parameter_dtype_state(parameter, state)
                z = state["z"]
                exp_avg_sq = state["exp_avg_sq"]

                if self.backend in {"auto", "triton"} and _triton_apply(
                    parameter, parameter.grad, exp_avg_sq, z,
                    beta2=beta2, bias_correction2=bias_correction2,
                    eps=eps, decay=decay, ckp1=ckp1,
                    gradient_scale=lr * (beta1 * (1 - ckp1) - 1),
                    z_scale=lr,
                ):
                    continue

                grad = parameter.grad.float()
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad, grad, value=1.0 - beta2,
                )
                denom = exp_avg_sq.float().div(
                    bias_correction2,
                ).sqrt_().add_(eps)
                grad_normalized = grad.div(denom)

                y = parameter.float()
                if decay != 0:
                    grad_normalized.add_(y, alpha=decay)
                y.lerp_(z.float(), weight=ckp1)
                y.add_(grad_normalized, alpha=lr * (beta1 * (1 - ckp1) - 1))
                z.sub_(grad_normalized, alpha=lr)
                self._copy_updated_parameter(parameter, y)

            group["k"] = k + 1
        return loss


class RAdamScheduleFree(_ParameterDtypeStateScheduleFreeMixin, _RAdamScheduleFree):
    """Schedule-Free RAdam with parameter-dtype full-size state."""

    def __init__(
        self, params, lr=0.0025, betas=(0.9, 0.999), eps=1e-8,
        weight_decay: float = 0.0, r=0.0, weight_lr_power=2.0, foreach=True,
        silent_sgd_phase=True, *, backend="auto",
    ):
        if backend not in {"auto", "torch", "triton"}:
            raise ValueError("backend must be 'auto', 'torch', or 'triton'")
        super().__init__(
            params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, r=r, weight_lr_power=weight_lr_power,
            foreach=foreach, silent_sgd_phase=silent_sgd_phase,
        )
        self.backend = backend

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        if not self.param_groups[0]["train_mode"]:
            raise RuntimeError(
                "Optimizer was not in train mode when step is called. "
                "Call optimizer.train() before training."
            )

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            decay = group["weight_decay"]
            k = group["k"]
            step = k + 1
            r = group["r"]
            weight_lr_power = group["weight_lr_power"]

            beta2_t = beta2 ** step
            bias_correction2 = 1 - beta2_t
            rho_inf = 2 / (1 - beta2) - 1
            rho_t = rho_inf - 2 * step * beta2_t / bias_correction2
            rect = (
                (
                    (rho_t - 4) * (rho_t - 2) * rho_inf
                    / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                ) ** 0.5
                if rho_t > 4.0
                else float(not group["silent_sgd_phase"])
            )

            lr = group["lr"] * rect
            group["scheduled_lr"] = lr
            lr_max = group["lr_max"] = max(lr, group["lr_max"])
            weight = (step ** r) * (lr_max ** weight_lr_power)
            weight_sum = group["weight_sum"] = group["weight_sum"] + weight
            ckp1 = weight / weight_sum if weight_sum != 0 else 0.0
            adaptive_y_lr = lr * (beta1 * (1 - ckp1) - 1)

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError(
                        "RAdamScheduleFree does not support sparse gradients"
                    )
                state = self.state[parameter]
                self._ensure_parameter_dtype_state(parameter, state)
                z = state["z"]
                exp_avg_sq = state["exp_avg_sq"]

                if self.backend in {"auto", "triton"} and _triton_apply_radam(
                    parameter, parameter.grad, exp_avg_sq, z,
                    beta2=beta2, bias_correction2=bias_correction2,
                    eps=eps, decay=decay, ckp1=ckp1,
                    adaptive_y_lr=adaptive_y_lr, lr=lr,
                    normalize=rho_t > 4.0,
                ):
                    continue

                grad = parameter.grad.float()
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad, grad, value=1.0 - beta2,
                )
                if rho_t > 4.0:
                    denom = exp_avg_sq.float().div(
                        bias_correction2,
                    ).sqrt_().add_(eps)
                    grad = grad.div_(denom)

                y = parameter.float()
                if decay != 0:
                    grad.add_(y, alpha=decay)
                y.lerp_(z.float(), weight=ckp1)
                y.add_(grad, alpha=adaptive_y_lr)
                z.sub_(grad, alpha=lr)
                self._copy_updated_parameter(parameter, y)

            group["k"] = k + 1
        return loss
