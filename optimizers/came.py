"""Optimizer-specific AutoSchedule variants."""

from __future__ import annotations

from typing import Callable, Optional

import torch
from came_pytorch import CAME as _ReferenceCAME

from .auto_schedule import AutoScheduleMixin
from .came_triton import apply_update as _triton_apply_update


class CAME(_ReferenceCAME):
    """CAME with parameter-dtype full-size moments."""

    @staticmethod
    def _ensure_full_state_dtype(
        parameter: torch.Tensor, state: dict,
    ) -> None:
        """Keep full-size CAME moments in the parameter storage dtype."""
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            if value is not None and value.dtype != parameter.dtype:
                state[key] = value.to(dtype=parameter.dtype)

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-16),
        clip_threshold=1.0,
        betas=(0.9, 0.999, 0.9999),
        weight_decay=0.0,
        *,
        backend="auto",
    ):
        if backend not in {"auto", "torch", "triton"}:
            raise ValueError("backend must be 'auto', 'torch', or 'triton'")
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
        )
        self.backend = backend

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
            beta1, beta2, beta3 = group["betas"]
            eps_square, eps_instability = group["eps"]
            lr = group["lr"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients.")

                state = self.state[parameter]
                grad_shape = grad.shape
                factored = self._get_options(grad_shape)
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(
                            grad_shape[:-1], device=grad.device, dtype=grad.dtype
                        )
                        state["exp_avg_sq_col"] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:],
                            device=grad.device, dtype=grad.dtype,
                        )
                        state["exp_avg_res_row"] = torch.zeros(
                            grad_shape[:-1], device=grad.device, dtype=grad.dtype
                        )
                        state["exp_avg_res_col"] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:],
                            device=grad.device, dtype=grad.dtype,
                        )
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                    state["RMS"] = 0
                self._ensure_full_state_dtype(parameter, state)

                state["step"] += 1
                # Keep the reference implementation's checkpoint-visible state.
                state["RMS"] = self._rms(parameter.data)
                update = grad.square().add_(eps_square)
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    exp_avg_sq_row.mul_(beta2).add_(
                        update.mean(dim=-1), alpha=1.0 - beta2
                    )
                    exp_avg_sq_col.mul_(beta2).add_(
                        update.mean(dim=-2), alpha=1.0 - beta2
                    )
                    update = self._approx_sq_grad(
                        exp_avg_sq_row, exp_avg_sq_col
                    ).mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_(
                        update, alpha=1.0 - beta2
                    )
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_(
                    (self._rms(update) / group["clip_threshold"]).clamp_(min=1.0)
                )
                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta1).add_(update, alpha=1.0 - beta1)

                # update is no longer needed after exp_avg has been updated.
                # Reusing it avoids a full-size subtraction/square temporary.
                update.sub_(exp_avg).square_().add_(eps_instability)
                if factored:
                    exp_avg_res_row = state["exp_avg_res_row"]
                    exp_avg_res_col = state["exp_avg_res_col"]
                    exp_avg_res_row.mul_(beta3).add_(
                        update.mean(dim=-1), alpha=1.0 - beta3
                    )
                    exp_avg_res_col.mul_(beta3).add_(
                        update.mean(dim=-2), alpha=1.0 - beta3
                    )
                    update = self._approx_sq_grad(
                        exp_avg_res_row, exp_avg_res_col
                    ).mul_(exp_avg)
                else:
                    update = exp_avg.clone()

                used_triton = False
                if self.backend in {"auto", "triton"}:
                    used_triton = _triton_apply_update(
                        parameter,
                        update,
                        lr=lr,
                        weight_decay=group["weight_decay"],
                    )
                if not used_triton:
                    if group["weight_decay"] != 0:
                        parameter.add_(
                            parameter, alpha=-group["weight_decay"] * lr
                        )
                    update.mul_(lr)
                    parameter.add_(-update)

        return loss


class CAMEAutoSchedule(AutoScheduleMixin, CAME):
    """CAME with a bounded, per-parameter-group LR controller."""

    def __init__(
        self,
        params,
        *,
        lr=0.001,
        eps=(1e-30, 1e-16),
        clip_threshold=1.0,
        betas=(0.9, 0.999, 0.9999),
        weight_decay=0.0,
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
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
            backend=backend,
        )
        self._enable_auto_schedule(
            kind="came",
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
            stats = None
            beta1, beta2, beta3 = group["betas"]
            eps_square, eps_instability = group["eps"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients.")

                if stats is None:
                    stats = self._auto_schedule_new_stats(parameter)
                self._auto_schedule_add_norm(
                    stats, "parameter_norm_sq", parameter.data
                )

                state = self.state[parameter]
                grad_shape = grad.shape
                factored = len(grad_shape) >= 2
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(
                            grad_shape[:-1], device=grad.device, dtype=grad.dtype
                        )
                        state["exp_avg_sq_col"] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:],
                            device=grad.device,
                            dtype=grad.dtype,
                        )
                        state["exp_avg_res_row"] = torch.zeros(
                            grad_shape[:-1], device=grad.device, dtype=grad.dtype
                        )
                        state["exp_avg_res_col"] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:],
                            device=grad.device,
                            dtype=grad.dtype,
                        )
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                    state["RMS"] = 0
                self._ensure_full_state_dtype(parameter, state)

                state["step"] += 1
                state["RMS"] = self._rms(parameter.data)

                update = grad.square().add_(eps_square)
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    exp_avg_sq_row.mul_(beta2).add_(
                        update.mean(dim=-1), alpha=1.0 - beta2
                    )
                    exp_avg_sq_col.mul_(beta2).add_(
                        update.mean(dim=-2), alpha=1.0 - beta2
                    )
                    update = self._approx_sq_grad(
                        exp_avg_sq_row, exp_avg_sq_col
                    ).mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_(
                        update, alpha=1.0 - beta2
                    )
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_(
                    (self._rms(update) / group["clip_threshold"]).clamp_(min=1.0)
                )

                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta1).add_(update, alpha=1.0 - beta1)
                residual = (update - exp_avg).square().add_(eps_instability)
                self._auto_schedule_add_norm(stats, "noise_norm_sq", update - exp_avg)
                self._auto_schedule_add_norm(stats, "moment_norm_sq", exp_avg)

                if factored:
                    exp_avg_res_row = state["exp_avg_res_row"]
                    exp_avg_res_col = state["exp_avg_res_col"]
                    exp_avg_res_row.mul_(beta3).add_(
                        residual.mean(dim=-1), alpha=1.0 - beta3
                    )
                    exp_avg_res_col.mul_(beta3).add_(
                        residual.mean(dim=-2), alpha=1.0 - beta3
                    )
                    update = self._approx_sq_grad(
                        exp_avg_res_row, exp_avg_res_col
                    ).mul_(exp_avg)
                else:
                    update = exp_avg.clone()

                self._auto_schedule_add_norm(
                    stats, "update_norm_sq", update * lr
                )

                used_triton = False
                if self.backend in {"auto", "triton"}:
                    used_triton = _triton_apply_update(
                        parameter,
                        update,
                        lr=lr,
                        weight_decay=group["weight_decay"],
                    )
                if not used_triton:
                    if group["weight_decay"] != 0:
                        parameter.data.add_(
                            parameter.data,
                            alpha=-group["weight_decay"] * lr,
                        )
                    update.mul_(lr)
                    parameter.data.add_(-update)

            self._auto_schedule_finish_group(group, stats)

        return loss
