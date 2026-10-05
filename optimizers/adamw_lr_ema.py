"""AdamW-style low-rank projected-gradient EMA optimizer."""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch


class AdamWLowRankGradientEMA(torch.optim.Optimizer):
    """Decoupled weight decay with a low-rank gradient EMA.

    This is an ablation for Schedule-Free experiments.  It does not keep a
    Schedule-Free hidden parameter or an Adam second moment.  For matrix
    parameters, the gradient is projected onto a fixed orthonormal basis,
    accumulated as a fast EMA in that latent matrix, bias-corrected, decoded,
    and used directly as an SGD-like update.  Vectors and matrices for which
    the requested rank is not smaller than the effective matrix rank use a
    full gradient EMA instead.

    ``projection_scale="norm"`` restores the expected update norm after a
    random rank-r projection, matching the norm-preserving convention used by
    APOLLO.  It is a scale correction, not an attempt to reconstruct the
    discarded gradient components.
    """

    _backend_name = "lr_ema"

    def __init__(
        self,
        params,
        *,
        lr=1e-3,
        rank=8,
        ema_beta=0.9,
        seed=0,
        weight_decay=0.0,
        projection_scale="norm",
    ):
        if lr < 0.0:
            raise ValueError("lr must be non-negative")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError("ema_beta must be between 0 and 1")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if projection_scale not in {"norm", "none"}:
            raise ValueError("projection_scale must be 'norm' or 'none'")
        defaults = dict(
            lr=float(lr),
            lr_ema_rank=int(rank),
            lr_ema_beta=float(ema_beta),
            lr_ema_seed=int(seed),
            weight_decay=float(weight_decay),
            lr_ema_projection_scale=projection_scale,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _matrix_view(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1)

    @classmethod
    def _make_projection(
        cls, parameter: torch.Tensor, rank: int, seed: int,
    ) -> torch.Tensor:
        matrix = cls._matrix_view(parameter)
        rows, cols = matrix.shape
        generator = torch.Generator(device=parameter.device).manual_seed(seed)
        if rows >= cols:
            random = torch.randn(
                cols,
                rank,
                generator=generator,
                device=parameter.device,
                dtype=torch.float32,
            )
            return torch.linalg.qr(random, mode="reduced").Q.contiguous()
        random = torch.randn(
            rows,
            rank,
            generator=generator,
            device=parameter.device,
            dtype=torch.float32,
        )
        return torch.linalg.qr(random, mode="reduced").Q.transpose(
            0, 1,
        ).contiguous()

    @classmethod
    def _ensure_state(
        cls, parameter: torch.Tensor, state: dict, group: dict,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor, bool]:
        matrix = cls._matrix_view(parameter)
        if parameter.ndim < 2 or min(matrix.shape) <= 0:
            state.setdefault("backend", "full")
            if state["backend"] != "full":
                raise RuntimeError(
                    "AdamW-LR-EMA checkpoint contains incompatible state"
                )
            if "lr_ema_grad" not in state:
                state["lr_ema_grad"] = torch.zeros_like(
                    parameter, memory_format=torch.preserve_format,
                )
            elif state["lr_ema_grad"].dtype != parameter.dtype:
                state["lr_ema_grad"] = state["lr_ema_grad"].to(
                    dtype=parameter.dtype,
                )
            return None, state["lr_ema_grad"], False

        rank = min(int(group["lr_ema_rank"]), min(matrix.shape))
        if rank >= min(matrix.shape):
            state.setdefault("backend", "full")
            if state["backend"] != "full":
                raise RuntimeError(
                    "AdamW-LR-EMA checkpoint contains incompatible state"
                )
            if "lr_ema_grad" not in state:
                state["lr_ema_grad"] = torch.zeros_like(
                    parameter, memory_format=torch.preserve_format,
                )
            elif state["lr_ema_grad"].dtype != parameter.dtype:
                state["lr_ema_grad"] = state["lr_ema_grad"].to(
                    dtype=parameter.dtype,
                )
            return None, state["lr_ema_grad"], False

        if state.get("backend") not in {None, cls._backend_name}:
            raise RuntimeError(
                "AdamW-LR-EMA checkpoint contains incompatible state"
            )
        if "lr_ema_projection" not in state:
            projection = cls._make_projection(
                parameter, rank, int(group["lr_ema_seed"]),
            )
            state["lr_ema_projection"] = projection
            latent_shape = (
                (matrix.shape[0], rank)
                if matrix.shape[0] >= matrix.shape[1]
                else (rank, matrix.shape[1])
            )
            state["lr_ema_grad"] = torch.zeros(
                latent_shape, device=parameter.device, dtype=torch.float32,
            )
            state["lr_ema_rank"] = rank
            state["backend"] = cls._backend_name
        projection = state["lr_ema_projection"]
        ema_gradient = state["lr_ema_grad"]
        if (
            projection.device != parameter.device
            or ema_gradient.device != parameter.device
        ):
            raise RuntimeError(
                "AdamW-LR-EMA state and parameter must share a device"
            )
        return projection, ema_gradient, True

    @staticmethod
    def _copy_parameter(parameter: torch.Tensor, value: torch.Tensor) -> None:
        parameter.copy_(value.to(dtype=parameter.dtype).reshape(parameter.shape))

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
            lr = group["lr"]
            beta = group["lr_ema_beta"]
            decay = group["weight_decay"]
            scale_mode = group["lr_ema_projection_scale"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError(
                        "AdamW-LR-EMA does not support sparse gradients"
                    )
                state = self.state[parameter]
                state["step"] = int(state.get("step", 0)) + 1
                step = state["step"]
                projection, ema_gradient, low_rank = self._ensure_state(
                    parameter, state, group,
                )
                gradient = parameter.grad.float()
                bias_correction = 1.0 - beta**step
                value = parameter.float()
                if decay != 0.0:
                    value.mul_(1.0 - lr * decay)

                if low_rank:
                    matrix = gradient.reshape(gradient.shape[0], -1)
                    tall = matrix.shape[0] >= matrix.shape[1]
                    if tall:
                        projected = matrix.matmul(projection)
                    else:
                        projected = projection.matmul(matrix)
                    ema_gradient.mul_(beta).add_(
                        projected, alpha=1.0 - beta,
                    )
                    corrected = ema_gradient.float().div(bias_correction)
                    if scale_mode == "norm":
                        input_dim = (
                            projection.shape[0] if tall else projection.shape[1]
                        )
                        rank_dim = (
                            projection.shape[1] if tall else projection.shape[0]
                        )
                        corrected = corrected.mul_(
                            math.sqrt(input_dim / rank_dim)
                        )
                    if tall:
                        value.reshape(value.shape[0], -1).addmm_(
                            corrected,
                            projection.transpose(0, 1),
                            alpha=-lr,
                        )
                    else:
                        value.reshape(value.shape[0], -1).addmm_(
                            projection.transpose(0, 1),
                            corrected,
                            alpha=-lr,
                        )
                else:
                    state_gradient = gradient.to(dtype=parameter.dtype)
                    ema_gradient.lerp_(state_gradient, 1.0 - beta)
                    value.add_(
                        ema_gradient.float(), alpha=-lr / bias_correction,
                    )
                self._copy_parameter(parameter, value)
        return loss


class AdamWLowRankGradientEMAConfidence(AdamWLowRankGradientEMA):
    """Low-rank gradient EMA with innovation-variance normalization.

    In addition to the latent gradient EMA ``m``, this variant keeps
    ``c = EMA((g - m_previous)^2)`` in the same latent space and updates with
    ``m_hat / sqrt(c_hat + alpha * m_hat**2)``.  The ``alpha`` floor prevents
    nearly deterministic gradient directions from producing an unbounded
    update when the innovation variance collapses.
    """

    _backend_name = "lr_ema_confidence"

    def __init__(
        self,
        params,
        *,
        lr=1e-3,
        rank=8,
        ema_beta=0.9,
        confidence_beta=0.99,
        confidence_alpha=1e-3,
        seed=0,
        weight_decay=0.0,
        projection_scale="norm",
        eps=1e-8,
    ):
        if not 0.0 <= confidence_beta < 1.0:
            raise ValueError("confidence_beta must be between 0 and 1")
        if confidence_alpha < 0.0:
            raise ValueError("confidence_alpha must be non-negative")
        if eps < 0.0:
            raise ValueError("eps must be non-negative")
        super().__init__(
            params,
            lr=lr,
            rank=rank,
            ema_beta=ema_beta,
            seed=seed,
            weight_decay=weight_decay,
            projection_scale=projection_scale,
        )
        for group in self.param_groups:
            group.update(
                lr_ema_confidence_beta=float(confidence_beta),
                lr_ema_confidence_alpha=float(confidence_alpha),
                lr_ema_eps=float(eps),
            )

    @classmethod
    def _ensure_state(
        cls, parameter: torch.Tensor, state: dict, group: dict,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor, bool]:
        projection, ema_gradient, low_rank = super()._ensure_state(
            parameter, state, group,
        )
        if "lr_ema_residual_sq" not in state:
            state["lr_ema_residual_sq"] = torch.zeros_like(ema_gradient)
        elif state["lr_ema_residual_sq"].dtype != ema_gradient.dtype:
            state["lr_ema_residual_sq"] = state["lr_ema_residual_sq"].to(
                dtype=ema_gradient.dtype,
            )
        return projection, ema_gradient, low_rank

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
            lr = group["lr"]
            beta = group["lr_ema_beta"]
            confidence_beta = group["lr_ema_confidence_beta"]
            alpha = group["lr_ema_confidence_alpha"]
            eps = group["lr_ema_eps"]
            decay = group["weight_decay"]
            scale_mode = group["lr_ema_projection_scale"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError(
                        "AdamW-LR-EMA-Conf does not support sparse gradients"
                    )
                state = self.state[parameter]
                state["step"] = int(state.get("step", 0)) + 1
                step = state["step"]
                projection, ema_gradient, low_rank = self._ensure_state(
                    parameter, state, group,
                )
                residual_sq = state["lr_ema_residual_sq"]
                gradient = parameter.grad.float()
                value = parameter.float()
                if decay != 0.0:
                    value.mul_(1.0 - lr * decay)

                if low_rank:
                    matrix = gradient.reshape(gradient.shape[0], -1)
                    tall = matrix.shape[0] >= matrix.shape[1]
                    if tall:
                        projected = matrix.matmul(projection)
                    else:
                        projected = projection.matmul(matrix)
                    residual = projected - ema_gradient
                    residual_sq.mul_(confidence_beta).addcmul_(
                        residual, residual, value=1.0 - confidence_beta,
                    )
                    ema_gradient.mul_(beta).add_(
                        projected, alpha=1.0 - beta,
                    )
                    mean = ema_gradient.float().div(1.0 - beta**step)
                    variance = residual_sq.float().div(
                        1.0 - confidence_beta**step
                    )
                    normalized = mean.div(
                        variance.addcmul(mean, mean, value=alpha)
                        .sqrt_()
                        .add_(eps)
                    )
                    if scale_mode == "norm":
                        input_dim = (
                            projection.shape[0] if tall else projection.shape[1]
                        )
                        rank_dim = (
                            projection.shape[1] if tall else projection.shape[0]
                        )
                        normalized.mul_(math.sqrt(input_dim / rank_dim))
                    if tall:
                        value.reshape(value.shape[0], -1).addmm_(
                            normalized,
                            projection.transpose(0, 1),
                            alpha=-lr,
                        )
                    else:
                        value.reshape(value.shape[0], -1).addmm_(
                            projection.transpose(0, 1),
                            normalized,
                            alpha=-lr,
                        )
                else:
                    state_gradient = gradient.to(dtype=parameter.dtype)
                    residual = gradient - ema_gradient.float()
                    residual_sq.mul_(confidence_beta).addcmul_(
                        residual.to(dtype=parameter.dtype),
                        residual.to(dtype=parameter.dtype),
                        value=1.0 - confidence_beta,
                    )
                    ema_gradient.lerp_(state_gradient, 1.0 - beta)
                    mean = ema_gradient.float().div(1.0 - beta**step)
                    variance = residual_sq.float().div(
                        1.0 - confidence_beta**step
                    )
                    value.add_(
                        mean.div(
                            variance.addcmul(mean, mean, value=alpha)
                            .sqrt_()
                            .add_(eps)
                        ),
                        alpha=-lr,
                    )
                self._copy_parameter(parameter, value)
        return loss
