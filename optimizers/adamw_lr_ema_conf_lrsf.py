"""Schedule-Free LRSF with a low-rank projected-gradient confidence state."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .adamw_lrsf import AdamWLRSF
from .projection_refresh import (
    OrthogonalRefreshPolicy,
    ProjectionRefreshPolicy,
    add_mixed_delta,
    advance_refresh,
    maybe_start_refresh,
    prepare_stochastic_refresh,
    rotate_projection_state,
)


class AdamWLRSEMAConfLRSF(AdamWLRSF):
    """Combine low-rank confidence normalization with LRSF drift.

    ``m`` and ``c`` are maintained in a fixed gradient projection, while the
    inherited LRSF state maintains the Schedule-Free drift in an independent
    projection.  The separation is intentional: confidence models gradient
    innovation and LRSF models train/eval trajectory drift.

    Non-matrix and rank-saturated matrix parameters use the inherited full
    Schedule-Free AdamW fallback.  Confidence projection refresh is not part
    of this first prototype; only the LRSF/delta projection follows the
    configured refresh policy.
    """

    def __init__(
        self,
        params,
        *,
        lr=0.0025,
        rank=8,
        sf_beta1=0.9,
        beta2=0.999,
        warmup_steps=0,
        r=0.0,
        weight_lr_power=2.0,
        ema_beta=0.9,
        confidence_beta=0.99,
        confidence_alpha=1e-3,
        seed=0,
        eps=1e-8,
        weight_decay=0.0,
        projection_refresh=None,
        orthogonal_refresh=None,
        backend="auto",
    ):
        if not 0.0 <= confidence_beta < 1.0:
            raise ValueError("confidence_beta must be between 0 and 1")
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError("ema_beta must be between 0 and 1")
        if confidence_alpha < 0.0:
            raise ValueError("confidence_alpha must be non-negative")
        super().__init__(
            params,
            lr=lr,
            rank=rank,
            sf_beta1=sf_beta1,
            beta2=beta2,
            warmup_steps=warmup_steps,
            r=r,
            weight_lr_power=weight_lr_power,
            seed=seed,
            eps=eps,
            weight_decay=weight_decay,
            projection_refresh=projection_refresh,
            orthogonal_refresh=orthogonal_refresh,
            backend=backend,
        )
        for group in self.param_groups:
            group.update(
                lr_ema_confidence_beta=float(confidence_beta),
                lr_ema_confidence_alpha=float(confidence_alpha),
                lr_ema_beta=float(ema_beta),
                # Keep the two coordinate systems independent even when the
                # public optimizer seed is the same.
                lr_ema_confidence_seed=int(seed) + 1_000_003,
            )

    @classmethod
    def _ensure_confidence_state(
        cls, parameter: torch.Tensor, state: dict, group: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        matrix = cls._matrix_view(parameter)
        rank = min(int(group["lrsf_rank"]), min(matrix.shape))
        if not cls._use_lrsf(parameter, rank):
            return None
        if "lr_ema_confidence_projection" not in state:
            projection = cls._make_projection(
                parameter, rank, group["lr_ema_confidence_seed"],
            )
            latent_shape = (
                (matrix.shape[0], rank)
                if matrix.shape[0] >= matrix.shape[1]
                else (rank, matrix.shape[1])
            )
            state["lr_ema_confidence_projection"] = projection
            state["lr_ema_grad"] = torch.zeros(
                latent_shape, device=parameter.device, dtype=torch.float32,
            )
            state["lr_ema_residual_sq"] = torch.zeros(
                latent_shape, device=parameter.device, dtype=torch.float32,
            )
            state["lr_ema_confidence_rank"] = rank
        projection = state["lr_ema_confidence_projection"]
        mean = state["lr_ema_grad"]
        residual_sq = state["lr_ema_residual_sq"]
        if (
            projection.device != parameter.device
            or mean.device != parameter.device
            or residual_sq.device != parameter.device
        ):
            raise RuntimeError(
                "AdamW-LR-EMA-Conf-LRSF state and parameter must share a device"
            )
        return projection, mean, residual_sq

    @staticmethod
    def _confidence_update(
        gradient: torch.Tensor,
        projection: torch.Tensor,
        mean: torch.Tensor,
        residual_sq: torch.Tensor,
        *,
        step: int,
        beta: float,
        confidence_beta: float,
        alpha: float,
        eps: float,
    ) -> torch.Tensor:
        matrix = gradient.reshape(gradient.shape[0], -1)
        tall = matrix.shape[0] >= matrix.shape[1]
        projected = (
            matrix.matmul(projection)
            if tall else projection.matmul(matrix)
        )
        residual = projected - mean
        residual_sq.mul_(confidence_beta).addcmul_(
            residual, residual, value=1.0 - confidence_beta,
        )
        mean.mul_(beta).add_(projected, alpha=1.0 - beta)
        corrected_mean = mean.float().div(1.0 - beta**step)
        corrected_variance = residual_sq.float().div(
            1.0 - confidence_beta**step
        )
        normalized = corrected_mean.div(
            corrected_variance.addcmul(
                corrected_mean, corrected_mean, value=alpha,
            ).sqrt_().add_(eps)
        )
        input_dim = projection.shape[0] if tall else projection.shape[1]
        rank_dim = projection.shape[1] if tall else projection.shape[0]
        normalized.mul_((input_dim / rank_dim) ** 0.5)
        decoded = torch.empty_like(gradient)
        decoded_matrix = decoded.reshape(gradient.shape[0], -1)
        if tall:
            decoded_matrix.copy_(normalized.matmul(projection.transpose(0, 1)))
        else:
            decoded_matrix.copy_(
                projection.transpose(0, 1).matmul(normalized)
            )
        return decoded

    @torch.no_grad()
    def step(
        self, closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        if any(not group["train_mode"] for group in self.param_groups):
            raise RuntimeError(
                "AdamW-LR-EMA-Conf-LRSF requires optimizer.train() before step()."
            )
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta_sf, beta2 = group["betas"]
            k = group["k"]
            warmup_steps = group["warmup_steps"]
            sched = (k + 1) / warmup_steps if k < warmup_steps else 1.0
            lr = group["lr"] * sched
            group["scheduled_lr"] = lr
            group["lr_max"] = max(lr, group["lr_max"])
            weight = ((k + 1) ** group["r"]) * (
                group["lr_max"] ** group["weight_lr_power"]
            )
            group["weight_sum"] += weight
            ckp1 = weight / group["weight_sum"] if group["weight_sum"] else 0.0
            bias_correction2 = 1.0 - beta2 ** (k + 1)

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError(
                        "AdamW-LR-EMA-Conf-LRSF does not support sparse gradients"
                    )
                state = self.state[parameter]
                _projection, _delta, use_lrsf = self._ensure_lrsf_state(
                    parameter, state, group,
                )
                if not use_lrsf:
                    exp_avg_sq = self._ensure_second_moment(parameter, state)
                    if "z" not in state:
                        state["z"] = parameter.detach().clone(
                            memory_format=torch.preserve_format,
                        )
                    elif state["z"].dtype != parameter.dtype:
                        state["z"] = state["z"].to(dtype=parameter.dtype)
                    grad = parameter.grad.float()
                    exp_avg_sq.mul_(beta2).addcmul_(
                        grad, grad, value=1.0 - beta2,
                    )
                    denom = exp_avg_sq.div(
                        bias_correction2,
                    ).sqrt_().add_(group["eps"])
                    effective_update = grad.div(denom).add(
                        parameter.float(), alpha=group["weight_decay"],
                    )
                    y = parameter.float()
                    # The fallback state follows the parameter dtype, while
                    # the update is accumulated in FP32.  Cast only the
                    # temporary interpolation operand to keep lerp_ valid
                    # for BF16 parameters without changing persistent state.
                    y.lerp_(state["z"].float(), weight=ckp1)
                    y.add_(
                        effective_update,
                        alpha=lr * (beta_sf * (1.0 - ckp1) - 1.0),
                    )
                    state["z"].sub_(effective_update, alpha=lr)
                    parameter.copy_(y.to(dtype=parameter.dtype))
                    continue

                confidence_state = self._ensure_confidence_state(
                    parameter, state, group,
                )
                assert confidence_state is not None
                confidence_projection, mean, residual_sq = confidence_state
                state["lr_ema_step"] = int(state.get("lr_ema_step", 0)) + 1
                effective_update = self._confidence_update(
                    parameter.grad.float(),
                    confidence_projection,
                    mean,
                    residual_sq,
                    step=state["lr_ema_step"],
                    beta=group["lr_ema_beta"],
                    confidence_beta=group["lr_ema_confidence_beta"],
                    alpha=group["lr_ema_confidence_alpha"],
                    eps=group["eps"],
                )
                if group["weight_decay"] != 0.0:
                    effective_update.add_(
                        parameter.float(), alpha=group["weight_decay"],
                    )
                policy = ProjectionRefreshPolicy.from_value(
                    group.get("projection_refresh")
                )
                orthogonal_policy = OrthogonalRefreshPolicy.from_value(
                    group.get("orthogonal_refresh")
                )
                rotation_signal = (
                    effective_update
                    if orthogonal_policy.signal == "effective_update"
                    else parameter.grad
                )
                rotate_projection_state(
                    state, orthogonal_policy, gradient=rotation_signal,
                )
                maybe_start_refresh(
                    parameter, state, policy, k, group["lrsf_seed"],
                    self._make_projection,
                )
                prepare_stochastic_refresh(state, policy, group["lrsf_seed"])
                update_scale = lr * (beta_sf * (1.0 - ckp1) - 1.0)
                delta_scale = -(1.0 - ckp1) * lr * beta_sf

                def apply_update(matrix):
                    add_mixed_delta(
                        matrix, state, ckp1, self._add_low_rank, policy,
                    )
                    matrix.add_(
                        effective_update.reshape(matrix.shape),
                        alpha=update_scale,
                    )

                self._apply_to_parameter(parameter, apply_update)
                self._update_projected_delta(
                    effective_update, state, 1.0 - ckp1, delta_scale,
                )
                advance_refresh(state, policy)
            group["k"] = k + 1
        return loss


__all__ = ["AdamWLRSEMAConfLRSF"]
