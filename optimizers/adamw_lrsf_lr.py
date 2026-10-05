"""AdamW with low-rank Schedule-Free drift and preconditioner states."""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch

from .adamw_lrsf import AdamWLRSF
from .projection_refresh import (
    ProjectionRefreshPolicy,
    ensure_shadow_state,
    maybe_start_refresh,
)
from .schedulefree_triton import apply as _triton_apply


class AdamWLRSLowRankPreconditioner(AdamWLRSF):
    """Schedule-Free AdamW with both matrix states in one low-rank space.

    Matrix parameters keep a low-rank ``lrsf_delta`` for the Schedule-Free
    hidden drift and a low-rank ``lrsf_exp_avg_sq`` for a latent diagonal
    preconditioner.  The two states share ``lrsf_projection``.  Vectors and
    rank-saturated small matrices use the exact full Schedule-Free fallback.

    This is the first integrated LRTDO prototype.  Hard refresh supports two
    explicit latent-moment policies: approximate ``transport`` or local-step
    ``reset``.  ``shadow`` refresh additionally keeps a second projection,
    drift, and latent-moment branch warm in the background and promotes it at
    the refresh interval.
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
        projection_refresh_state="transport",
        backend="auto",
    ):
        refresh_policy = ProjectionRefreshPolicy.from_value(projection_refresh)
        if refresh_policy.mode not in {"none", "hard", "shadow"}:
            raise ValueError(
                "AdamW-LRSF-LR currently supports only none, hard, or shadow refresh"
            )
        if projection_refresh_state not in {"reset", "transport"}:
            raise ValueError(
                "projection_refresh_state must be 'reset' or 'transport'"
            )
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
            projection_refresh=refresh_policy.as_dict(),
            orthogonal_refresh=None,
            backend=backend,
        )
        for group in self.param_groups:
            group["projection_refresh_state"] = projection_refresh_state

    @staticmethod
    def _transport_latent_second_moment(
        value: torch.Tensor,
        old_projection: torch.Tensor,
        new_projection: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate diagonal-variance transport into a new basis."""
        if old_projection.shape[0] >= old_projection.shape[1]:
            overlap = old_projection.transpose(0, 1).matmul(new_projection)
            return value.matmul(overlap.square()).clamp_min_(0.0)
        overlap = new_projection.matmul(old_projection.transpose(0, 1))
        return overlap.square().matmul(value).clamp_min_(0.0)

    @staticmethod
    def _transport_low_rank_delta(
        value: torch.Tensor,
        old_projection: torch.Tensor,
        new_projection: torch.Tensor,
    ) -> torch.Tensor:
        """Transport a low-rank coefficient matrix into a new basis."""
        if old_projection.shape[0] >= old_projection.shape[1]:
            return value.matmul(old_projection.transpose(0, 1)).matmul(new_projection)
        return new_projection.matmul(old_projection.transpose(0, 1)).matmul(value)

    @staticmethod
    def _blend_shadow_projection(
        old_projection: torch.Tensor,
        new_projection: torch.Tensor,
        overlap: float | None,
    ) -> torch.Tensor:
        """Keep a new shadow basis close when overlap blending is requested."""
        if overlap is None or overlap == 0.0:
            return new_projection
        if overlap == 1.0:
            return old_projection.detach().clone()
        candidate = old_projection.float().mul(overlap).add(
            new_projection.float(), alpha=1.0 - overlap,
        )
        if old_projection.shape[0] >= old_projection.shape[1]:
            return torch.linalg.qr(candidate, mode="reduced").Q.to(
                dtype=old_projection.dtype,
            ).contiguous()
        return torch.linalg.qr(candidate.transpose(0, 1), mode="reduced").Q.transpose(
            0, 1,
        ).to(dtype=old_projection.dtype).contiguous()

    @classmethod
    def _ensure_shadow_integrated_state(
        cls, parameter: torch.Tensor, state: dict, group: dict,
    ) -> None:
        policy = ProjectionRefreshPolicy.from_value(group.get("projection_refresh"))
        if policy.mode != "shadow":
            state["shadow_active"] = False
            return
        ensure_shadow_state(
            parameter,
            state,
            policy,
            group["lrsf_seed"],
            cls._make_projection,
        )
        if "lrsf_shadow_exp_avg_sq" not in state:
            state["lrsf_shadow_exp_avg_sq"] = cls._transport_latent_second_moment(
                state["lrsf_exp_avg_sq"],
                state["lrsf_projection"],
                state["lrsf_shadow_projection"],
            )
            state["lrsf_shadow_moment_step"] = int(
                state.get("lrsf_moment_step", 0)
            )

    @classmethod
    def _promote_shadow_integrated_state(
        cls,
        parameter: torch.Tensor,
        state: dict,
        group: dict,
        step: int,
    ) -> bool:
        """Promote the warm branch and start a newly seeded shadow branch."""
        policy = ProjectionRefreshPolicy.from_value(group.get("projection_refresh"))
        if (
            policy.mode != "shadow"
            or step <= 0
            or step % policy.interval != 0
            or not state.get("shadow_active", False)
        ):
            return False

        shadow_projection = state["lrsf_shadow_projection"]
        shadow_delta = state["lrsf_shadow_delta"]
        shadow_moment = state["lrsf_shadow_exp_avg_sq"]

        state["lrsf_projection"] = shadow_projection
        state["lrsf_delta"] = shadow_delta
        state["lrsf_exp_avg_sq"] = shadow_moment
        state["lrsf_moment_step"] = int(
            state.get("lrsf_shadow_moment_step", state.get("lrsf_moment_step", 0))
        )

        refresh_count = int(state.get("refresh_count", 0)) + 1
        rank = (
            shadow_projection.shape[-1]
            if shadow_projection.shape[0] >= shadow_projection.shape[1]
            else shadow_projection.shape[0]
        )
        new_shadow_projection = cls._make_projection(
            parameter, rank, int(group["lrsf_seed"]) + refresh_count + 1,
        )
        new_shadow_projection = cls._blend_shadow_projection(
            shadow_projection,
            new_shadow_projection,
            policy.transport_overlap,
        )
        new_shadow_delta = cls._transport_low_rank_delta(
            state["lrsf_delta"], shadow_projection, new_shadow_projection,
        )
        new_shadow_moment = cls._transport_latent_second_moment(
            state["lrsf_exp_avg_sq"], shadow_projection, new_shadow_projection,
        )
        state["lrsf_shadow_projection"] = new_shadow_projection
        state["lrsf_shadow_delta"] = new_shadow_delta
        state["lrsf_shadow_exp_avg_sq"] = new_shadow_moment
        state["lrsf_shadow_moment_step"] = state["lrsf_moment_step"]
        state["refresh_count"] = refresh_count
        state["shadow_generation"] = refresh_count
        return True

    @classmethod
    def _ensure_integrated_state(
        cls, parameter: torch.Tensor, state: dict, group: dict,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]:
        matrix = cls._matrix_view(parameter)
        rank = min(int(group["lrsf_rank"]), min(matrix.shape))
        if not cls._use_lrsf(parameter, rank):
            state.setdefault("backend", "sf_full")
            return None, None, False
        if state.get("backend") not in {None, "lrsf"}:
            raise RuntimeError(
                "AdamW-LRSF-LR checkpoint contains an incompatible state backend"
            )
        if "lrsf_projection" not in state:
            projection = cls._make_projection(
                parameter, rank, int(group["lrsf_seed"]),
            )
            rows, cols = matrix.shape
            latent_shape = (rows, rank) if rows >= cols else (rank, cols)
            state["lrsf_projection"] = projection
            state["lrsf_delta"] = torch.zeros(
                latent_shape, device=parameter.device, dtype=torch.float32,
            )
            state["lrsf_exp_avg_sq"] = torch.zeros(
                latent_shape, device=parameter.device, dtype=torch.float32,
            )
            state["lrsf_rank"] = rank
            state["backend"] = "lrsf"
        projection = state["lrsf_projection"]
        latent_second_moment = state["lrsf_exp_avg_sq"]
        if projection.device != parameter.device or latent_second_moment.device != parameter.device:
            raise RuntimeError(
                "AdamW-LRSF-LR state and parameter must share a device"
            )
        cls._ensure_shadow_integrated_state(parameter, state, group)
        if state.get("shadow_active", False) and (
            state["lrsf_shadow_projection"].device != parameter.device
            or state["lrsf_shadow_delta"].device != parameter.device
            or state["lrsf_shadow_exp_avg_sq"].device != parameter.device
        ):
            raise RuntimeError(
                "AdamW-LRSF-LR shadow state and parameter must share a device"
            )
        return projection, latent_second_moment, True

    @staticmethod
    def _latent_effective_update(
        gradient: torch.Tensor,
        parameter: torch.Tensor,
        projection: torch.Tensor,
        latent_second_moment: torch.Tensor,
        beta2: float,
        bias_correction2: float,
        eps: float,
        decay: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return decoded effective update and its projected coefficient."""
        matrix = gradient.reshape(gradient.shape[0], -1)
        parameter_matrix = parameter.float().reshape(parameter.shape[0], -1)
        tall = matrix.shape[0] >= matrix.shape[1]
        if tall:
            projected = matrix.matmul(projection)
            latent_second_moment.mul_(beta2).addcmul_(
                projected, projected, value=1.0 - beta2,
            )
            projected.div_(
                latent_second_moment.div(bias_correction2).sqrt_().add_(eps)
            )
            projected.mul_(math.sqrt(projection.shape[0] / projection.shape[1]))
            projected_effective = projected
            if decay:
                projected_effective = projected_effective.add(
                    parameter_matrix.matmul(projection), alpha=decay,
                )
            decoded = projected_effective.matmul(projection.transpose(0, 1))
            return decoded.reshape_as(gradient), projected_effective

        projected = projection.matmul(matrix)
        latent_second_moment.mul_(beta2).addcmul_(
            projected, projected, value=1.0 - beta2,
        )
        projected.div_(
            latent_second_moment.div(bias_correction2).sqrt_().add_(eps)
        )
        projected.mul_(math.sqrt(projection.shape[1] / projection.shape[0]))
        projected_effective = projected
        if decay:
            projected_effective = projected_effective.add(
                projection.matmul(parameter_matrix), alpha=decay,
            )
        decoded = projection.transpose(0, 1).matmul(projected_effective)
        return decoded.reshape_as(gradient), projected_effective

    @staticmethod
    def _full_state(
        parameter: torch.Tensor, state: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    @torch.no_grad()
    def step(
        self, closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        if any(not group["train_mode"] for group in self.param_groups):
            raise RuntimeError("AdamW-LRSF-LR requires optimizer.train() before step().")
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
            update_scale = lr * (beta_sf * (1.0 - ckp1) - 1.0)
            delta_scale = -(1.0 - ckp1) * lr * beta_sf

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError(
                        "AdamW-LRSF-LR does not support sparse gradients"
                    )
                state = self.state[parameter]
                projection, latent_second_moment, use_lrsf = (
                    self._ensure_integrated_state(parameter, state, group)
                )
                if not use_lrsf:
                    z, exp_avg_sq = self._full_state(parameter, state)
                    if self.backend in {"auto", "triton"} and _triton_apply(
                        parameter,
                        parameter.grad,
                        exp_avg_sq,
                        z,
                        beta2=beta2,
                        bias_correction2=bias_correction2,
                        eps=group["eps"],
                        decay=group["weight_decay"],
                        ckp1=ckp1,
                        gradient_scale=update_scale,
                        z_scale=lr,
                    ):
                        continue
                    gradient = parameter.grad.float()
                    exp_avg_sq.mul_(beta2).addcmul_(
                        gradient, gradient, value=1.0 - beta2,
                    )
                    effective_update = gradient.div(
                        exp_avg_sq.float().div(bias_correction2).sqrt_().add_(
                            group["eps"]
                        )
                    )
                    y = parameter.float()
                    if group["weight_decay"]:
                        effective_update.add_(y, alpha=group["weight_decay"])
                    y.lerp_(z.float(), weight=ckp1)
                    y.add_(effective_update, alpha=update_scale)
                    z.sub_(effective_update, alpha=lr)
                    parameter.copy_(y.to(dtype=parameter.dtype))
                    continue

                refresh_policy = ProjectionRefreshPolicy.from_value(
                    group.get("projection_refresh")
                )
                if refresh_policy.mode == "shadow":
                    self._promote_shadow_integrated_state(
                        parameter, state, group, k,
                    )
                    projection = state["lrsf_projection"]
                    latent_second_moment = state["lrsf_exp_avg_sq"]
                elif refresh_policy.mode == "hard":
                    old_projection = projection
                    if maybe_start_refresh(
                        parameter,
                        state,
                        refresh_policy,
                        k,
                        group["lrsf_seed"],
                        self._make_projection,
                    ):
                        projection = state["lrsf_projection"]
                        if group["projection_refresh_state"] == "reset":
                            state["lrsf_exp_avg_sq"].zero_()
                            state["lrsf_moment_step"] = 0
                        else:
                            state["lrsf_exp_avg_sq"] = (
                                self._transport_latent_second_moment(
                                    latent_second_moment,
                                    old_projection,
                                    projection,
                                )
                            )
                        latent_second_moment = state["lrsf_exp_avg_sq"]

                moment_step = int(state.get("lrsf_moment_step", k)) + 1
                state["lrsf_moment_step"] = moment_step
                moment_bias_correction2 = 1.0 - beta2 ** moment_step
                effective_update, projected_effective = self._latent_effective_update(
                    parameter.grad.float(),
                    parameter,
                    projection,
                    latent_second_moment,
                    beta2,
                    moment_bias_correction2,
                    group["eps"],
                    group["weight_decay"],
                )
                matrix_delta = state["lrsf_delta"]

                def apply_update(matrix):
                    self._add_low_rank(matrix, matrix_delta, projection, ckp1)
                    matrix.add_(
                        effective_update.reshape(matrix.shape),
                        alpha=update_scale,
                    )

                shadow_projected_effective = None
                if state.get("shadow_active", False):
                    shadow_step = int(
                        state.get("lrsf_shadow_moment_step", k)
                    ) + 1
                    state["lrsf_shadow_moment_step"] = shadow_step
                    shadow_bias_correction2 = 1.0 - beta2 ** shadow_step
                    _, shadow_projected_effective = self._latent_effective_update(
                        parameter.grad.float(),
                        parameter,
                        state["lrsf_shadow_projection"],
                        state["lrsf_shadow_exp_avg_sq"],
                        beta2,
                        shadow_bias_correction2,
                        group["eps"],
                        group["weight_decay"],
                    )

                self._apply_to_parameter(parameter, apply_update)
                matrix_delta.mul_(1.0 - ckp1).add_(
                    projected_effective, alpha=delta_scale,
                )

                if shadow_projected_effective is not None:
                    state["lrsf_shadow_delta"].mul_(1.0 - ckp1).add_(
                        shadow_projected_effective, alpha=delta_scale,
                    )

            group["k"] = k + 1
        return loss


__all__ = ["AdamWLRSLowRankPreconditioner"]
