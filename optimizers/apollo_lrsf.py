"""APOLLO-CAME with a low-rank Schedule-Free delta state.

Optional delta-projection refreshes are independent of APOLLO's update
projection refresh path.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Callable, Optional

import torch

from .apollo import APOLLOFallbackPolicy, APOLLOCAME
from .came_lrsf import CAMELRSF
from .projection_refresh import (
    OrthogonalRefreshPolicy,
    ProjectionRefreshPolicy,
    add_mixed_delta,
    advance_refresh,
    ensure_shadow_state,
    maybe_start_refresh,
    prepare_stochastic_refresh,
    project_mixed_update,
    rotate_projection_state,
    update_mixed_delta,
)


class APOLLOCAMELRSF(APOLLOCAME):
    """APOLLO-CAME plus a separate fixed-projection Schedule-Free delta.

    APOLLO-CAME's ``projection`` and low-rank CAME state are the update path.
    ``lrsf_projection`` and ``lrsf_delta`` are an independent path for the
    Schedule-Free hidden parameter.  Unsupported vectors and CAME-selected
    small matrices remain ordinary APOLLO-CAME fallbacks.
    """

    def __init__(
        self,
        params,
        *,
        lr=1e-3,
        rank=4,
        lrsf_rank=None,
        sf_beta1=0.9,
        warmup_steps=0,
        r=0.0,
        weight_lr_power=2.0,
        seed=0,
        eps=(1e-30, 1e-16),
        clip_threshold=1.0,
        betas=(0.9, 0.999, 0.9999),
        weight_decay=0.0,
        scale=1.0,
        update_proj_gap=200,
        scale_front=False,
        norm_growth_limiter=False,
        norm_growth_rate=1.01,
        fallback: APOLLOFallbackPolicy | Mapping[str, object] | str = "came",
        came_backend="torch",
        delta_refresh=None,
        orthogonal_refresh=None,
        apollo_orthogonal_refresh=None,
    ):
        if rank <= 0:
            raise ValueError("rank must be positive")
        if lrsf_rank is not None and lrsf_rank <= 0:
            raise ValueError("lrsf_rank must be positive")
        if not 0.0 < sf_beta1 < 1.0:
            raise ValueError("sf_beta1 must be between 0 and 1")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if r < 0.0 or weight_lr_power < 0.0:
            raise ValueError("Schedule-Free weighting values must be non-negative")
        refresh_policy = ProjectionRefreshPolicy.from_value(delta_refresh)
        orthogonal_policy = OrthogonalRefreshPolicy.from_value(
            orthogonal_refresh, default_seed=int(seed),
        )
        super().__init__(
            params,
            lr=lr,
            rank=rank,
            scale=scale,
            betas=betas,
            eps=eps,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
            update_proj_gap=update_proj_gap,
            seed=seed,
            scale_front=scale_front,
            norm_growth_limiter=norm_growth_limiter,
            norm_growth_rate=norm_growth_rate,
            fallback=fallback,
            came_backend=came_backend,
            orthogonal_refresh=apollo_orthogonal_refresh,
        )
        for group in self.param_groups:
            group.update(
                lrsf_rank=int(rank if lrsf_rank is None else lrsf_rank),
                sf_beta1=float(sf_beta1),
                warmup_steps=int(warmup_steps),
                lrsf_r=float(r),
                lrsf_weight_lr_power=float(weight_lr_power),
                lrsf_seed=int(seed),
                k=0,
                train_mode=False,
                weight_sum=0.0,
                lr_max=-1.0,
                scheduled_lr=0.0,
                delta_refresh=refresh_policy.as_dict(),
                orthogonal_refresh=orthogonal_policy.as_dict(),
            )

    @staticmethod
    def _matrix_view(tensor: torch.Tensor) -> torch.Tensor:
        return CAMELRSF._matrix_view(tensor)

    @staticmethod
    def _use_lrsf(parameter: torch.Tensor, rank: int) -> bool:
        return CAMELRSF._use_lrsf(parameter, rank)

    @staticmethod
    def _make_lrsf_projection(
        parameter: torch.Tensor, rank: int, seed: int,
    ) -> torch.Tensor:
        return CAMELRSF._make_projection(parameter, rank, seed)

    @staticmethod
    def _add_low_rank(
        matrix: torch.Tensor,
        delta: torch.Tensor,
        projection: torch.Tensor,
        alpha: float,
    ) -> None:
        CAMELRSF._add_low_rank(matrix, delta, projection, alpha)

    @staticmethod
    def _project_update(
        update: torch.Tensor, projection: torch.Tensor,
    ) -> torch.Tensor:
        return CAMELRSF._project_update(update, projection)

    @staticmethod
    def _apply_to_parameter(
        parameter: torch.Tensor,
        callback: Callable[[torch.Tensor], None],
    ) -> None:
        CAMELRSF._apply_to_parameter(parameter, callback)

    def _ensure_lrsf_state(
        self, parameter: torch.Tensor, state: dict, group: dict,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]:
        rank = min(
            int(group["lrsf_rank"]),
            min(self._matrix_view(parameter).shape),
        )
        if not self._use_lrsf(parameter, rank):
            return None, None, False
        matrix = self._matrix_view(parameter)
        if "lrsf_projection" not in state:
            state["lrsf_projection"] = self._make_lrsf_projection(
                parameter, rank, group["lrsf_seed"],
            )
            state["lrsf_projection_rank"] = rank
            state["lrsf_delta"] = torch.zeros(
                (matrix.shape[0], rank) if matrix.shape[0] >= matrix.shape[1]
                else (rank, matrix.shape[1]),
                device=parameter.device,
                dtype=torch.float32,
            )
        state.setdefault("refresh_count", 0)
        state.setdefault("refresh_progress", 0)
        state.setdefault("refresh_active", False)
        policy = ProjectionRefreshPolicy.from_value(group.get("delta_refresh"))
        ensure_shadow_state(
            parameter,
            state,
            policy,
            group["lrsf_seed"],
            self._make_lrsf_projection,
        )
        projection = state["lrsf_projection"]
        delta = state["lrsf_delta"]
        if projection.device != parameter.device or delta.device != parameter.device:
            raise RuntimeError("APOLLO-CAME-LRSF state and parameter must share a device")
        if state.get("shadow_active", False) and (
            state["lrsf_shadow_projection"].device != parameter.device
            or state["lrsf_shadow_delta"].device != parameter.device
        ):
            raise RuntimeError(
                "APOLLO-CAME-LRSF shadow state and parameter must share a device"
            )
        return projection, delta, True

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            restore_scale = 1.0 / group["sf_beta1"] - 1.0
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if not state or "lrsf_delta" not in state:
                    continue
                self._apply_to_parameter(
                    parameter,
                    lambda matrix, state=state, group=group: add_mixed_delta(
                        matrix, state, restore_scale, self._add_low_rank,
                        ProjectionRefreshPolicy.from_value(group.get("delta_refresh")),
                    ),
                )
            group["train_mode"] = True

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            if not group["train_mode"]:
                continue
            eval_scale = 1.0 - 1.0 / group["sf_beta1"]
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if not state or "lrsf_delta" not in state:
                    continue
                self._apply_to_parameter(
                    parameter,
                    lambda matrix, state=state, group=group: add_mixed_delta(
                        matrix, state, eval_scale, self._add_low_rank,
                        ProjectionRefreshPolicy.from_value(group.get("delta_refresh")),
                    ),
                )
            group["train_mode"] = False

    @torch.no_grad()
    def step(
        self, closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        if any(not group["train_mode"] for group in self.param_groups):
            raise RuntimeError(
                "APOLLO-CAME-LRSF requires optimizer.train() before step().",
            )
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            k = int(group["k"])
            warmup_steps = group["warmup_steps"]
            sched = (k + 1) / warmup_steps if k < warmup_steps else 1.0
            lr = group["lr"] * sched
            group["scheduled_lr"] = lr
            group["lr_max"] = max(lr, group["lr_max"])
            weight = ((k + 1) ** group["lrsf_r"]) * (
                group["lr_max"] ** group["lrsf_weight_lr_power"]
            )
            group["weight_sum"] += weight
            ckp1 = weight / group["weight_sum"] if group["weight_sum"] else 0.0
            scale_factor = group["scale"] ** 0.5

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "APOLLO-CAME-LRSF does not support sparse gradients",
                    )
                state = self.state[parameter]
                backend = self._select_backend(parameter, state, group)
                if backend == "came":
                    if parameter.ndim >= 2:
                        update = self._full_came_matrix_update(
                            parameter, grad, state, group,
                        )
                    else:
                        update = self._fallback_scale(parameter, grad, state, group)
                    if group["weight_decay"] != 0.0:
                        parameter.mul_(1.0 - lr * group["weight_decay"])
                    parameter.add_(update, alpha=-lr)
                    continue

                self._ensure_state(parameter, state, group)
                state["step"] = int(state.get("step", 0)) + 1
                self._refresh_projection(parameter, state, group)
                grad_matrix = self._matrix_view(grad.float())
                _, scaling = self._low_rank_update(
                    parameter, grad, state, group, grad_matrix=grad_matrix,
                )
                update = self._apply_scaling(
                    grad_matrix, scaling, self.scale_type, grad.shape,
                )
                if group["scale_front"]:
                    update = update.mul(scale_factor)
                current_norm = update.norm()
                limiter_ratio = None
                if group["norm_growth_limiter"]:
                    previous_norm = state.get("scaled_grad_norm")
                    if previous_norm is not None:
                        max_norm = previous_norm * group["norm_growth_rate"]
                        limiter_ratio = torch.minimum(
                            current_norm.new_ones(()),
                            max_norm / current_norm.clamp_min(group["eps"]),
                        )
                        current_norm = torch.minimum(current_norm, max_norm)
                    state["scaled_grad_norm"] = current_norm.detach()
                if not group["scale_front"]:
                    update = update.mul(scale_factor)
                if limiter_ratio is not None:
                    update = update * limiter_ratio

                _projection, _delta, use_lrsf = self._ensure_lrsf_state(
                    parameter, state, group,
                )
                if not use_lrsf:
                    parameter.add_(update, alpha=-lr)
                    continue
                policy = ProjectionRefreshPolicy.from_value(group.get("delta_refresh"))
                orthogonal_policy = OrthogonalRefreshPolicy.from_value(
                    group.get("orthogonal_refresh"),
                )
                if group["weight_decay"] != 0.0:
                    effective_update = update.reshape(update.shape).add(
                        parameter.float(), alpha=group["weight_decay"],
                    )
                else:
                    effective_update = update
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
                    self._make_lrsf_projection,
                )
                prepare_stochastic_refresh(state, policy, group["lrsf_seed"])
                projected_updates = project_mixed_update(
                    effective_update, state, self._project_update,
                )
                update_scale = lr * (
                    group["sf_beta1"] * (1.0 - ckp1) - 1.0
                )
                delta_scale = -(
                    1.0 - ckp1
                ) * lr * group["sf_beta1"]

                def apply_update(matrix):
                    add_mixed_delta(
                        matrix, state, ckp1, self._add_low_rank, policy,
                    )
                    matrix.add_(effective_update.reshape(matrix.shape), alpha=update_scale)

                self._apply_to_parameter(parameter, apply_update)
                update_mixed_delta(
                    state, projected_updates, 1.0 - ckp1, delta_scale,
                )
                advance_refresh(state, policy)

            group["k"] = k + 1
        return loss


__all__ = ["APOLLOCAMELRSF"]
