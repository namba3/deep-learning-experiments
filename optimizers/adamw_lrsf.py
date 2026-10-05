"""AdamW with a low-rank Schedule-Free hidden state."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .projection_refresh import (
    OrthogonalRefreshPolicy,
    ProjectionRefreshPolicy,
    add_mixed_delta,
    advance_refresh,
    maybe_start_refresh,
    prepare_stochastic_refresh,
    rotate_projection_state,
    ensure_shadow_state,
)
from .schedulefree import AdamWScheduleFree
from .schedulefree_triton import apply as _triton_schedulefree_apply
from .schedulefree_triton import (
    apply_lrsf_preconditioner as _triton_lrsf_preconditioner,
)


class AdamWLRSF(AdamWScheduleFree):
    """Schedule-Free AdamW with a low-rank hidden delta for matrix weights.

    The AdamW second-moment state remains full-size and follows the parameter
    storage dtype.  Only the hidden Schedule-Free delta is projected to a
    fixed low-rank subspace.  Vectors and matrices for which the requested
    rank is not smaller than the effective matrix rank use the full
    Schedule-Free representation.
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
        projection_refresh=None,
        orthogonal_refresh=None,
        backend="auto",
    ):
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not 0.0 < sf_beta1 < 1.0:
            raise ValueError("sf_beta1 must be between 0 and 1")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be between 0 and 1")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if r < 0.0 or weight_lr_power < 0.0:
            raise ValueError("Schedule-Free weighting values must be non-negative")
        refresh_policy = ProjectionRefreshPolicy.from_value(projection_refresh)
        orthogonal_policy = OrthogonalRefreshPolicy.from_value(
            orthogonal_refresh,
        )
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
            group.update(
                lrsf_rank=int(rank),
                lrsf_seed=int(seed),
                projection_refresh=refresh_policy.as_dict(),
                orthogonal_refresh=orthogonal_policy.as_dict(),
            )

    @staticmethod
    def _matrix_view(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1)

    @classmethod
    def _effective_rank(cls, parameter: torch.Tensor, requested_rank: int) -> int:
        matrix = cls._matrix_view(parameter)
        return min(int(requested_rank), min(matrix.shape))

    @classmethod
    def _use_lrsf(cls, parameter: torch.Tensor, rank: int) -> bool:
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
        generator = torch.Generator(device=parameter.device).manual_seed(int(seed))
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
    def _ensure_lrsf_state(
        cls, parameter: torch.Tensor, state: dict, group: dict,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]:
        rank = cls._effective_rank(parameter, group["lrsf_rank"])
        if not cls._use_lrsf(parameter, rank):
            state.setdefault("backend", "sf_full")
            return None, None, False
        if state.get("backend") not in {None, "lrsf"}:
            return None, None, False
        matrix = cls._matrix_view(parameter)
        rows, cols = matrix.shape
        if "lrsf_projection" not in state:
            state["lrsf_projection"] = cls._make_projection(
                parameter, rank, group["lrsf_seed"],
            )
            state["lrsf_projection_rank"] = rank
            delta_shape = (rows, rank) if rows >= cols else (rank, cols)
            state["lrsf_delta"] = torch.zeros(
                delta_shape, device=parameter.device, dtype=torch.float32,
            )
            state["backend"] = "lrsf"
        state.setdefault("refresh_count", 0)
        state.setdefault("refresh_progress", 0)
        state.setdefault("refresh_active", False)
        policy = ProjectionRefreshPolicy.from_value(
            group.get("projection_refresh")
        )
        ensure_shadow_state(
            parameter,
            state,
            policy,
            group["lrsf_seed"],
            cls._make_projection,
        )
        projection = state["lrsf_projection"]
        delta = state["lrsf_delta"]
        if projection.device != parameter.device or delta.device != parameter.device:
            raise RuntimeError("AdamW-LRSF state and parameter must share a device")
        if state.get("shadow_active", False) and (
            state["lrsf_shadow_projection"].device != parameter.device
            or state["lrsf_shadow_delta"].device != parameter.device
        ):
            raise RuntimeError("AdamW-LRSF shadow state and parameter must share a device")
        return projection, delta, True

    @staticmethod
    def _add_low_rank(
        matrix: torch.Tensor,
        delta: torch.Tensor,
        projection: torch.Tensor,
        alpha: float,
    ) -> None:
        if matrix.shape[0] >= matrix.shape[1]:
            matrix.addmm_(delta, projection.transpose(0, 1), alpha=alpha)
        else:
            matrix.addmm_(projection.transpose(0, 1), delta, alpha=alpha)

    @staticmethod
    def _update_projected_delta(
        effective_update: torch.Tensor,
        state: dict,
        decay: float,
        update_scale: float,
    ) -> None:
        """Project and accumulate without materializing a temporary result."""
        matrix = effective_update.reshape(effective_update.shape[0], -1)
        tall = matrix.shape[0] >= matrix.shape[1]

        def update_delta(delta: torch.Tensor, projection: torch.Tensor) -> None:
            delta.mul_(decay)
            if tall:
                delta.addmm_(matrix, projection, alpha=update_scale)
            else:
                delta.addmm_(projection, matrix, alpha=update_scale)

        update_delta(state["lrsf_delta"], state["lrsf_projection"])
        if state.get("shadow_active", False):
            update_delta(
                state["lrsf_shadow_delta"], state["lrsf_shadow_projection"],
            )
            return
        if state.get("refresh_active", False):
            update_delta(
                state["refresh_next_delta"],
                state["refresh_next_projection"],
            )

    @staticmethod
    def _apply_to_parameter(
        parameter: torch.Tensor,
        callback: Callable[[torch.Tensor], None],
    ) -> None:
        if parameter.dtype == torch.float32 and parameter.is_contiguous():
            callback(parameter.reshape(parameter.shape[0], -1))
            return
        value = parameter.float()
        callback(value.reshape(value.shape[0], -1))
        parameter.copy_(value.to(dtype=parameter.dtype).reshape(parameter.shape))

    @staticmethod
    def _ensure_second_moment(parameter: torch.Tensor, state: dict) -> torch.Tensor:
        if "exp_avg_sq" not in state:
            state["exp_avg_sq"] = torch.zeros_like(parameter)
        elif state["exp_avg_sq"].dtype != parameter.dtype:
            state["exp_avg_sq"] = state["exp_avg_sq"].to(dtype=parameter.dtype)
        return state["exp_avg_sq"]

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            beta = group["betas"][0]
            restore_scale = 1.0 / beta - 1.0
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if not state:
                    continue
                if state.get("backend") == "lrsf":
                    self._apply_to_parameter(
                        parameter,
                        lambda matrix, state=state, group=group: add_mixed_delta(
                            matrix, state, restore_scale, self._add_low_rank,
                            ProjectionRefreshPolicy.from_value(
                                group.get("projection_refresh")
                            ),
                        ),
                    )
                elif "z" in state:
                    parameter.lerp_(state["z"].to(parameter.device), weight=1.0 - beta)
            group["train_mode"] = True

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            if not group["train_mode"]:
                continue
            beta = group["betas"][0]
            eval_scale = 1.0 - 1.0 / beta
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if not state:
                    continue
                if state.get("backend") == "lrsf":
                    self._apply_to_parameter(
                        parameter,
                        lambda matrix, state=state, group=group: add_mixed_delta(
                            matrix, state, eval_scale, self._add_low_rank,
                            ProjectionRefreshPolicy.from_value(
                                group.get("projection_refresh")
                            ),
                        ),
                    )
                elif "z" in state:
                    parameter.lerp_(
                        state["z"].to(parameter.device),
                        weight=1.0 - 1.0 / beta,
                    )
            group["train_mode"] = False

    @torch.no_grad()
    def step(
        self, closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        if any(not group["train_mode"] for group in self.param_groups):
            raise RuntimeError("AdamW-LRSF requires optimizer.train() before step().")
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
                    raise RuntimeError("AdamW-LRSF does not support sparse gradients")
                state = self.state[parameter]
                projection, _delta, use_lrsf = self._ensure_lrsf_state(
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
                    if self.backend in {"auto", "triton"} and _triton_schedulefree_apply(
                        parameter,
                        parameter.grad,
                        exp_avg_sq,
                        state["z"],
                        beta2=beta2,
                        bias_correction2=bias_correction2,
                        eps=group["eps"],
                        decay=group["weight_decay"],
                        ckp1=ckp1,
                        gradient_scale=lr * (beta_sf * (1.0 - ckp1) - 1.0),
                        z_scale=lr,
                    ):
                        continue
                    grad = parameter.grad.float()
                    exp_avg_sq.mul_(beta2).addcmul_(
                        grad, grad, value=1.0 - beta2,
                    )
                    denom = exp_avg_sq.div(
                        bias_correction2,
                    ).sqrt_().add_(group["eps"])
                    effective_update = grad.div(denom)
                    y = parameter.float()
                    if group["weight_decay"] != 0.0:
                        effective_update = effective_update.add(
                            y, alpha=group["weight_decay"],
                        )
                    y.lerp_(state["z"], weight=ckp1)
                    y.add_(
                        effective_update,
                        alpha=lr * (beta_sf * (1.0 - ckp1) - 1.0),
                    )
                    state["z"].sub_(effective_update, alpha=lr)
                    parameter.copy_(y.to(dtype=parameter.dtype))
                    continue

                exp_avg_sq = self._ensure_second_moment(parameter, state)
                effective_update = None
                if self.backend in {"auto", "triton"}:
                    effective_update = _triton_lrsf_preconditioner(
                        parameter,
                        parameter.grad,
                        exp_avg_sq,
                        beta2=beta2,
                        bias_correction2=bias_correction2,
                        eps=group["eps"],
                        decay=group["weight_decay"],
                    )
                if effective_update is None:
                    grad = parameter.grad.float()
                    exp_avg_sq.mul_(beta2).addcmul_(
                        grad, grad, value=1.0 - beta2,
                    )
                    denom = exp_avg_sq.div(
                        bias_correction2,
                    ).sqrt_().add_(group["eps"])
                    effective_update = grad.div(denom)
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


__all__ = ["AdamWLRSF"]
