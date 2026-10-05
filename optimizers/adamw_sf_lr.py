"""Schedule-Free AdamW with a low-rank latent preconditioner."""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch

from .schedulefree import AdamWScheduleFree
from .schedulefree_triton import apply as _triton_apply


class AdamWSFLowRankPreconditioner(AdamWScheduleFree):
    """Schedule-Free AdamW with only ``exp_avg_sq`` low-rank compressed.

    The full Schedule-Free hidden parameter ``z`` is intentionally retained.
    For a matrix-shaped parameter, the gradient is projected onto a fixed
    orthonormal basis, an elementwise second moment is maintained in that
    latent matrix, and the normalized latent update is decoded before the
    Schedule-Free interpolation.  Vectors and matrices whose requested rank
    is not smaller than their effective matrix rank use the exact full-state
    Schedule-Free path.

    This is an experimental ablation: it isolates preconditioner compression
    from the low-rank drift used by :class:`AdamWLRSF`.
    """

    def __init__(
        self,
        params,
        *,
        lr=0.0025,
        rank=4,
        sf_beta1=0.9,
        beta2=0.999,
        warmup_steps=0,
        r=0.0,
        weight_lr_power=2.0,
        seed=0,
        eps=1e-8,
        weight_decay=0.0,
        backend="auto",
    ):
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not 0.0 < sf_beta1 < 1.0:
            raise ValueError("sf_beta1 must be between 0 and 1")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be between 0 and 1")
        super().__init__(
            params,
            lr=lr,
            betas=(sf_beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            r=r,
            weight_lr_power=weight_lr_power,
            backend=backend,
        )
        for group in self.param_groups:
            group.update(sf_lr_rank=int(rank), sf_lr_seed=int(seed))

    @staticmethod
    def _matrix_view(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1)

    @classmethod
    def _use_low_rank(cls, parameter: torch.Tensor, rank: int) -> bool:
        if parameter.ndim < 2:
            return False
        matrix = cls._matrix_view(parameter)
        return rank < min(matrix.shape)

    @classmethod
    def _make_projection(
        cls, parameter: torch.Tensor, rank: int, seed: int,
    ) -> torch.Tensor:
        matrix = cls._matrix_view(parameter)
        rows, cols = matrix.shape
        generator = torch.Generator(device=parameter.device).manual_seed(seed)
        if rows >= cols:
            random = torch.randn(
                cols, rank, generator=generator,
                device=parameter.device, dtype=torch.float32,
            )
            return torch.linalg.qr(random, mode="reduced").Q.contiguous()
        random = torch.randn(
            rows, rank, generator=generator,
            device=parameter.device, dtype=torch.float32,
        )
        return torch.linalg.qr(random, mode="reduced").Q.transpose(0, 1).contiguous()

    @classmethod
    def _ensure_low_rank_state(
        cls, parameter: torch.Tensor, state: dict, group: dict,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]:
        matrix = cls._matrix_view(parameter)
        rank = min(int(group["sf_lr_rank"]), min(matrix.shape))
        if not cls._use_low_rank(parameter, rank):
            return None, None, False
        if state.get("backend") not in {None, "sf_lr"}:
            raise RuntimeError(
                "AdamW-SF-LR checkpoint contains an incompatible state backend"
            )
        if "z" not in state:
            state["z"] = parameter.detach().clone(
                memory_format=torch.preserve_format,
            )
        elif state["z"].dtype != parameter.dtype:
            state["z"] = state["z"].to(dtype=parameter.dtype)
        if "sf_lr_projection" not in state:
            state["sf_lr_projection"] = cls._make_projection(
                parameter, rank, int(group["sf_lr_seed"]),
            )
            state["sf_lr_rank"] = rank
            if matrix.shape[0] >= matrix.shape[1]:
                latent_shape = (matrix.shape[0], rank)
            else:
                latent_shape = (rank, matrix.shape[1])
            state["sf_lr_exp_avg_sq"] = torch.zeros(
                latent_shape, device=parameter.device, dtype=torch.float32,
            )
            state["backend"] = "sf_lr"
        projection = state["sf_lr_projection"]
        latent_second_moment = state["sf_lr_exp_avg_sq"]
        if projection.device != parameter.device or latent_second_moment.device != parameter.device:
            raise RuntimeError("AdamW-SF-LR state and parameter must share a device")
        return projection, latent_second_moment, True

    @staticmethod
    def _full_state(parameter: torch.Tensor, state: dict) -> tuple[torch.Tensor, torch.Tensor]:
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
        return state["z"], state["exp_avg_sq"]

    @staticmethod
    def _decode_normalized_gradient(
        gradient: torch.Tensor,
        projection: torch.Tensor,
        latent_second_moment: torch.Tensor,
        beta2: float,
        bias_correction2: float,
        eps: float,
    ) -> torch.Tensor:
        matrix = gradient.reshape(gradient.shape[0], -1)
        tall = matrix.shape[0] >= matrix.shape[1]
        if tall:
            projected = matrix.matmul(projection)
            latent_second_moment.mul_(beta2).addcmul_(
                projected, projected, value=1.0 - beta2,
            )
            projected.div_(
                latent_second_moment.div(bias_correction2).sqrt_().add_(eps)
            )
            # An orthonormal rank-r projection retains about r/n of the
            # isotropic update energy. Restore the expected full-space norm
            # before decoding, as APOLLO does for its projected update.
            projected.mul_(math.sqrt(projection.shape[0] / projection.shape[1]))
            return projected.matmul(projection.transpose(0, 1)).reshape_as(gradient)
        projected = projection.matmul(matrix)
        latent_second_moment.mul_(beta2).addcmul_(
            projected, projected, value=1.0 - beta2,
        )
        projected.div_(
            latent_second_moment.div(bias_correction2).sqrt_().add_(eps)
        )
        projected.mul_(math.sqrt(projection.shape[1] / projection.shape[0]))
        return projection.transpose(0, 1).matmul(projected).reshape_as(gradient)

    @torch.no_grad()
    def step(
        self, closure: Optional[Callable[[], float]] = None,
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
            beta1, beta2 = group["betas"]
            decay = group["weight_decay"]
            k = group["k"]
            warmup_steps = group["warmup_steps"]
            sched = (k + 1) / warmup_steps if k < warmup_steps else 1.0
            bias_correction2 = 1.0 - beta2 ** (k + 1)
            lr = group["lr"] * sched
            group["scheduled_lr"] = lr
            group["lr_max"] = max(lr, group["lr_max"])
            weight = ((k + 1) ** group["r"]) * (
                group["lr_max"] ** group["weight_lr_power"]
            )
            group["weight_sum"] += weight
            ckp1 = weight / group["weight_sum"] if group["weight_sum"] else 0.0

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError(
                        "AdamW-SF-LR does not support sparse gradients"
                    )
                state = self.state[parameter]
                projection, latent_second_moment, use_low_rank = (
                    self._ensure_low_rank_state(parameter, state, group)
                )
                if use_low_rank:
                    gradient = parameter.grad.float()
                    normalized = self._decode_normalized_gradient(
                        gradient, projection, latent_second_moment,
                        beta2, bias_correction2, group["eps"],
                    )
                    y = parameter.float()
                    if decay:
                        normalized.add_(y, alpha=decay)
                    y.lerp_(state["z"].float(), weight=ckp1)
                    y.add_(
                        normalized,
                        alpha=lr * (beta1 * (1.0 - ckp1) - 1.0),
                    )
                    state["z"].sub_(normalized, alpha=lr)
                    parameter.copy_(y.to(dtype=parameter.dtype))
                    continue

                state["backend"] = "sf_full"
                z, exp_avg_sq = self._full_state(parameter, state)
                if self.backend in {"auto", "triton"} and _triton_apply(
                    parameter, parameter.grad, exp_avg_sq, z,
                    beta2=beta2,
                    bias_correction2=bias_correction2,
                    eps=group["eps"],
                    decay=decay,
                    ckp1=ckp1,
                    gradient_scale=lr * (beta1 * (1.0 - ckp1) - 1.0),
                    z_scale=lr,
                ):
                    continue
                gradient = parameter.grad.float()
                exp_avg_sq.mul_(beta2).addcmul_(
                    gradient, gradient, value=1.0 - beta2,
                )
                normalized = gradient.div(
                    exp_avg_sq.float().div(bias_correction2).sqrt_().add_(
                        group["eps"]
                    )
                )
                y = parameter.float()
                if decay:
                    normalized.add_(y, alpha=decay)
                y.lerp_(z.float(), weight=ckp1)
                y.add_(
                    normalized,
                    alpha=lr * (beta1 * (1.0 - ckp1) - 1.0),
                )
                z.sub_(normalized, alpha=lr)
                parameter.copy_(y.to(dtype=parameter.dtype))

            group["k"] = k + 1
        return loss


__all__ = ["AdamWSFLowRankPreconditioner"]
