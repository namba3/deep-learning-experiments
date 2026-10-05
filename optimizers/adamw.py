"""AdamW with the project low-memory AutoSchedule controller."""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch

from .auto_schedule import AutoScheduleMixin
from .adamw_triton import apply as _triton_apply
from .adamw_triton import apply_parameter_update as _triton_apply_parameter_update


class AdamWFP32State(torch.optim.Optimizer):
    """AdamW with parameter-dtype moments.

    The class name is retained for checkpoint/API compatibility; state
    tensors with the same shape as the parameter follow its storage dtype.
    """

    def __init__(
        self,
        params,
        *,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        backend="auto",
    ):
        if lr < 0.0:
            raise ValueError("lr must be non-negative")
        if len(betas) != 2 or not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if eps < 0.0:
            raise ValueError("eps must be non-negative")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        defaults = dict(
            lr=float(lr),
            betas=tuple(betas),
            eps=float(eps),
            weight_decay=float(weight_decay),
            amsgrad=bool(amsgrad),
        )
        super().__init__(params, defaults)
        if backend not in {"auto", "torch", "triton"}:
            raise ValueError("backend must be 'auto', 'torch', or 'triton'")
        self.backend = backend

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError(
                        "AdamWFP32State does not support sparse gradients"
                    )
                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(parameter)

                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in state and state[key].dtype != parameter.dtype:
                        state[key] = state[key].to(dtype=parameter.dtype)

                state["step"] += 1
                step = int(state["step"])
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                max_exp_avg_sq = (
                    state["max_exp_avg_sq"] if group["amsgrad"] else None
                )
                if self.backend in {"auto", "triton"} and _triton_apply(
                    parameter,
                    parameter.grad,
                    exp_avg,
                    exp_avg_sq,
                    max_exp_avg_sq=max_exp_avg_sq,
                    beta1=beta1,
                    beta2=beta2,
                    bias_correction1=bias_correction1,
                    bias_correction2=bias_correction2,
                    eps=eps,
                    lr=lr,
                    weight_decay=group["weight_decay"],
                ):
                    continue
                state_grad = parameter.grad
                if state_grad.dtype != parameter.dtype:
                    state_grad = state_grad.to(dtype=parameter.dtype)
                exp_avg.lerp_(state_grad, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    state_grad, state_grad, value=1.0 - beta2,
                )
                if group["amsgrad"]:
                    torch.maximum(exp_avg_sq, max_exp_avg_sq, out=max_exp_avg_sq)
                    second_moment = max_exp_avg_sq
                else:
                    second_moment = exp_avg_sq
                denominator = (
                    second_moment.float().sqrt() / math.sqrt(bias_correction2)
                ).add_(eps)
                if group["weight_decay"] != 0.0:
                    parameter.mul_(1.0 - lr * group["weight_decay"])
                # addcdiv_ applies the FP32 update directly to the parameter,
                # avoiding separate full-size update and dtype-conversion
                # buffers for BF16/FP16 parameters.
                parameter.addcdiv_(
                    exp_avg.float(), denominator, value=-lr / bias_correction1,
                )
        return loss


class AdamWAutoSchedule(AutoScheduleMixin, torch.optim.Optimizer):
    """AdamW with bounded per-parameter-group LR adaptation.

    The Adam moments remain parameter-tensor state, while AutoSchedule keeps
    only scalar controller state in each optimizer parameter group.
    """

    def __init__(
        self,
        params,
        *,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=False,
        backend="auto",
        auto_schedule_target_update_ratio=1e-3,
        auto_schedule_ema_beta=0.99,
        auto_schedule_trust_alpha=0.1,
        auto_schedule_min_factor=0.5,
        auto_schedule_max_factor=4.0,
        auto_schedule_max_increase=1.05,
        auto_schedule_max_decrease=0.95,
        auto_schedule_confidence_floor=0.25,
        auto_schedule_stability_gain=2.0,
        auto_schedule_limiter_gain=2.0,
        auto_schedule_cooldown_steps=4,
        auto_schedule_warmup_steps=200,
    ):
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if len(betas) != 2 or not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if eps < 0.0:
            raise ValueError("eps must be non-negative")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")

        defaults = dict(
            lr=float(lr),
            betas=tuple(betas),
            eps=float(eps),
            weight_decay=float(weight_decay),
            amsgrad=bool(amsgrad),
        )
        super().__init__(params, defaults)
        if backend not in {"auto", "torch", "triton"}:
            raise ValueError("backend must be 'auto', 'torch', or 'triton'")
        self.backend = backend
        self._enable_auto_schedule(
            kind="adamw",
            target_update_ratio=auto_schedule_target_update_ratio,
            ema_beta=auto_schedule_ema_beta,
            trust_alpha=auto_schedule_trust_alpha,
            min_factor=auto_schedule_min_factor,
            max_factor=auto_schedule_max_factor,
            max_increase=auto_schedule_max_increase,
            max_decrease=auto_schedule_max_decrease,
            confidence_floor=auto_schedule_confidence_floor,
            stability_gain=auto_schedule_stability_gain,
            limiter_gain=auto_schedule_limiter_gain,
            cooldown_steps=auto_schedule_cooldown_steps,
            warmup_steps=auto_schedule_warmup_steps,
        )

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = self._auto_schedule_begin_group(group)
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            stats = None

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("AdamWAutoSchedule does not support sparse gradients")
                if stats is None:
                    stats = self._auto_schedule_new_stats(parameter)
                self._auto_schedule_add_norm(
                    stats, "parameter_norm_sq", parameter
                )

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(parameter)

                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in state and state[key].dtype != parameter.dtype:
                        state[key] = state[key].to(dtype=parameter.dtype)

                state["step"] += 1
                step = int(state["step"])
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state_grad = parameter.grad
                if state_grad.dtype != parameter.dtype:
                    state_grad = state_grad.to(dtype=parameter.dtype)
                exp_avg.lerp_(state_grad, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    state_grad, state_grad, value=1.0 - beta2
                )

                if group["amsgrad"]:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.maximum(
                        max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq
                    )
                    second_moment = max_exp_avg_sq
                else:
                    second_moment = exp_avg_sq

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denominator = (
                    second_moment.float().sqrt() / math.sqrt(bias_correction2)
                ).add_(eps)
                adaptive_update = (
                    exp_avg.float() / denominator
                ).mul_(lr / bias_correction1)
                self._auto_schedule_add_norm(
                    stats, "update_norm_sq", adaptive_update
                )
                self._auto_schedule_add_norm(
                    stats, "moment_norm_sq", exp_avg
                )

                if group["weight_decay"] != 0.0:
                    parameter.mul_(1.0 - lr * group["weight_decay"])
                used_triton = False
                if self.backend in {"auto", "triton"}:
                    used_triton = _triton_apply_parameter_update(
                        parameter,
                        adaptive_update,
                        lr=1.0,
                        weight_decay=0.0,
                    )
                if not used_triton:
                    parameter.add_(-adaptive_update.to(dtype=parameter.dtype))

            self._auto_schedule_finish_group(group, stats)

        return loss
