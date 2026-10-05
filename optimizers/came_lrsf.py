"""CAME with a low-rank Schedule-Free delta state.

The CAME update direction remains full-rank; only the Schedule-Free hidden
delta is represented in a randomized subspace. Optional interval and
per-step orthogonal refreshes affect this delta projection only.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .came import CAME
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


class CAMELRSF(CAME):
    """CAME with a fixed low-rank Schedule-Free hidden delta.

    Matrix parameters use one randomized orthonormal projection and one
    low-rank delta coefficient tensor.  Vector parameters and matrices whose
    effective rank would be full use the ordinary CAME update instead.
    """

    def __init__(
        self,
        params,
        *,
        lr=0.001,
        rank=4,
        sf_beta1=0.9,
        warmup_steps=0,
        r=0.0,
        weight_lr_power=2.0,
        seed=0,
        eps=(1e-30, 1e-16),
        clip_threshold=1.0,
        betas=(0.9, 0.999, 0.9999),
        weight_decay=0.0,
        projection_refresh=None,
        orthogonal_refresh=None,
    ):
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not 0.0 < sf_beta1 < 1.0:
            raise ValueError("sf_beta1 must be between 0 and 1")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if r < 0.0:
            raise ValueError("r must be non-negative")
        if weight_lr_power < 0.0:
            raise ValueError("weight_lr_power must be non-negative")
        refresh_policy = ProjectionRefreshPolicy.from_value(projection_refresh)
        orthogonal_policy = OrthogonalRefreshPolicy.from_value(
            orthogonal_refresh, default_seed=int(seed),
        )
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
            backend="torch",
        )
        for group in self.param_groups:
            group.update(
                rank=int(rank),
                sf_beta1=float(sf_beta1),
                warmup_steps=int(warmup_steps),
                r=float(r),
                weight_lr_power=float(weight_lr_power),
                seed=int(seed),
                k=0,
                train_mode=False,
                weight_sum=0.0,
                lr_max=-1.0,
                scheduled_lr=0.0,
                projection_refresh=refresh_policy.as_dict(),
                orthogonal_refresh=orthogonal_policy.as_dict(),
            )

    def load_state_dict(self, state_dict) -> None:
        """Restore backend markers that older PyTorch casts as iterables."""
        super().load_state_dict(state_dict)
        for state in self.state.values():
            backend = state.get("backend")
            if backend in {"came", "lrsf"}:
                continue
            state["backend"] = (
                "lrsf" if "lrsf_projection" in state else "came"
            )

    @staticmethod
    def _matrix_view(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1)

    @classmethod
    def _effective_rank(
        cls, parameter: torch.Tensor, requested_rank: int,
    ) -> int:
        matrix = cls._matrix_view(parameter)
        return min(int(requested_rank), min(matrix.shape))

    @classmethod
    def _use_lrsf(cls, parameter: torch.Tensor, rank: int) -> bool:
        if parameter.ndim < 2:
            return False
        matrix = cls._matrix_view(parameter)
        # A full-rank random basis adds state without providing a low-rank
        # approximation.  Let ordinary CAME handle that case.
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
            projection = torch.linalg.qr(random, mode="reduced").Q
        else:
            random = torch.randn(
                rows, rank, generator=generator,
                device=parameter.device, dtype=torch.float32,
            )
            projection = torch.linalg.qr(random, mode="reduced").Q.transpose(0, 1)
        return projection.contiguous()

    @classmethod
    def _ensure_lrsf_state(
        cls, parameter: torch.Tensor, state: dict, group: dict,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]:
        rank = cls._effective_rank(parameter, group["rank"])
        if not cls._use_lrsf(parameter, rank):
            state.setdefault("backend", "came")
            return None, None, False

        matrix = cls._matrix_view(parameter)
        rows, cols = matrix.shape
        if state.get("backend") not in {None, "lrsf"}:
            return None, None, False
        if "lrsf_projection" not in state:
            state["lrsf_projection"] = cls._make_projection(
                parameter, rank, group["seed"],
            )
            state["lrsf_projection_rank"] = rank
            if rows >= cols:
                delta_shape = (rows, rank)
            else:
                delta_shape = (rank, cols)
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
            group["seed"],
            cls._make_projection,
        )
        projection = state["lrsf_projection"]
        delta = state["lrsf_delta"]
        if projection.device != parameter.device or delta.device != parameter.device:
            raise RuntimeError("CAME-LRSF state and parameter must share a device")
        if state.get("shadow_active", False) and (
            state["lrsf_shadow_projection"].device != parameter.device
            or state["lrsf_shadow_delta"].device != parameter.device
        ):
            raise RuntimeError("CAME-LRSF shadow state and parameter must share a device")
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
    def _project_update(
        update: torch.Tensor, projection: torch.Tensor,
    ) -> torch.Tensor:
        matrix = update.reshape(update.shape[0], -1)
        if matrix.shape[0] >= matrix.shape[1]:
            return matrix.matmul(projection)
        return projection.matmul(matrix)

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

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            beta = group["sf_beta1"]
            restore_scale = 1.0 / beta - 1.0
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if not state or state.get("backend") != "lrsf":
                    continue
                self._apply_to_parameter(
                    parameter,
                    lambda matrix, state=state, group=group: add_mixed_delta(
                        matrix, state, restore_scale, self._add_low_rank,
                        ProjectionRefreshPolicy.from_value(
                            group.get("projection_refresh")
                        ),
                    ),
                )
            group["train_mode"] = True

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            if not group["train_mode"]:
                continue
            beta = group["sf_beta1"]
            eval_scale = 1.0 - 1.0 / beta
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if not state or state.get("backend") != "lrsf":
                    continue
                self._apply_to_parameter(
                    parameter,
                    lambda matrix, state=state, group=group: add_mixed_delta(
                        matrix, state, eval_scale, self._add_low_rank,
                        ProjectionRefreshPolicy.from_value(
                            group.get("projection_refresh")
                        ),
                    ),
                )
            group["train_mode"] = False

    def _came_update(
        self, parameter: torch.Tensor, grad: torch.Tensor, state: dict, group: dict,
    ) -> torch.Tensor:
        if grad.dtype in {torch.float16, torch.bfloat16}:
            grad = grad.float()
        if grad.is_sparse:
            raise RuntimeError("CAME-LRSF does not support sparse gradients.")

        beta1, beta2, beta3 = group["betas"]
        eps_square, eps_instability = group["eps"]
        factored = self._get_options(grad.shape)
        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(parameter)
            if factored:
                state["exp_avg_sq_row"] = torch.zeros(
                    grad.shape[:-1], device=grad.device, dtype=grad.dtype,
                )
                state["exp_avg_sq_col"] = torch.zeros(
                    grad.shape[:-2] + grad.shape[-1:],
                    device=grad.device, dtype=grad.dtype,
                )
                state["exp_avg_res_row"] = torch.zeros(
                    grad.shape[:-1], device=grad.device, dtype=grad.dtype,
                )
                state["exp_avg_res_col"] = torch.zeros(
                    grad.shape[:-2] + grad.shape[-1:],
                    device=grad.device, dtype=grad.dtype,
                )
            else:
                state["exp_avg_sq"] = torch.zeros_like(parameter)
            state["RMS"] = 0
        self._ensure_full_state_dtype(parameter, state)

        state["step"] += 1
        state["RMS"] = self._rms(parameter.data)
        update = grad.square().add_(eps_square)
        if factored:
            state["exp_avg_sq_row"].mul_(beta2).add_(
                update.mean(dim=-1), alpha=1.0 - beta2,
            )
            state["exp_avg_sq_col"].mul_(beta2).add_(
                update.mean(dim=-2), alpha=1.0 - beta2,
            )
            update = self._approx_sq_grad(
                state["exp_avg_sq_row"], state["exp_avg_sq_col"],
            ).mul_(grad)
        else:
            state["exp_avg_sq"].mul_(beta2).add_(
                update, alpha=1.0 - beta2,
            )
            update = state["exp_avg_sq"].rsqrt().mul_(grad)

        update.div_(
            (self._rms(update) / group["clip_threshold"]).clamp_(min=1.0),
        )
        state["exp_avg"].mul_(beta1).add_(
            update, alpha=1.0 - beta1,
        )
        update.sub_(state["exp_avg"]).square_().add_(eps_instability)
        if factored:
            state["exp_avg_res_row"].mul_(beta3).add_(
                update.mean(dim=-1), alpha=1.0 - beta3,
            )
            state["exp_avg_res_col"].mul_(beta3).add_(
                update.mean(dim=-2), alpha=1.0 - beta3,
            )
            return self._approx_sq_grad(
                state["exp_avg_res_row"], state["exp_avg_res_col"],
            ).mul_(state["exp_avg"])
        return state["exp_avg"].clone()

    @torch.no_grad()
    def step(
        self, closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        if any(not group["train_mode"] for group in self.param_groups):
            raise RuntimeError(
                "CAME-LRSF requires optimizer.train() before step().",
            )
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta_sf = group["sf_beta1"]
            k = int(group["k"])
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

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                _projection, _delta, use_lrsf = self._ensure_lrsf_state(
                    parameter, state, group,
                )
                update = self._came_update(parameter, parameter.grad, state, group)

                if not use_lrsf:
                    if group["weight_decay"] != 0.0:
                        parameter.mul_(1.0 - lr * group["weight_decay"])
                    parameter.add_(update, alpha=-lr)
                    continue
                policy = ProjectionRefreshPolicy.from_value(
                    group.get("projection_refresh")
                )
                orthogonal_policy = OrthogonalRefreshPolicy.from_value(
                    group.get("orthogonal_refresh"),
                )
                update_matrix = self._matrix_view(update)
                if group["weight_decay"] != 0.0:
                    # Schedule-Free applies decay to the update direction used
                    # by both y and the hidden state.  Keep the CAME fallback
                    # decoupled, since it is an ordinary CAME parameter path.
                    parameter_matrix = self._matrix_view(parameter).float()
                    effective_update = update_matrix.add(
                        parameter_matrix, alpha=group["weight_decay"],
                    )
                else:
                    effective_update = update_matrix
                rotation_signal = (
                    effective_update
                    if orthogonal_policy.signal == "effective_update"
                    else parameter.grad
                )
                rotate_projection_state(
                    state, orthogonal_policy, gradient=rotation_signal,
                )
                maybe_start_refresh(
                    parameter, state, policy, k, group["seed"],
                    self._make_projection,
                )
                prepare_stochastic_refresh(state, policy, group["seed"])
                projected_updates = project_mixed_update(
                    effective_update, state, self._project_update,
                )
                update_scale = lr * (beta_sf * (1.0 - ckp1) - 1.0)
                delta_scale = -(1.0 - ckp1) * lr * beta_sf

                def apply_update(matrix):
                    add_mixed_delta(
                        matrix, state, ckp1, self._add_low_rank, policy,
                    )
                    matrix.add_(effective_update, alpha=update_scale)

                self._apply_to_parameter(parameter, apply_update)
                update_mixed_delta(
                    state, projected_updates, 1.0 - ckp1, delta_scale,
                )
                advance_refresh(state, policy)

            group["k"] = k + 1
        return loss


class CAMESF(CAME):
    """Full-state Schedule-Free CAME oracle.

    This intentionally keeps a full-size hidden delta in parameter dtype.  It is a
    reference for separating Schedule-Free behavior from the low-rank
    approximation in :class:`CAMELRSF`, not a memory-saving optimizer.
    """

    def __init__(
        self,
        params,
        *,
        lr=0.001,
        sf_beta1=0.9,
        warmup_steps=0,
        r=0.0,
        weight_lr_power=2.0,
        eps=(1e-30, 1e-16),
        clip_threshold=1.0,
        betas=(0.9, 0.999, 0.9999),
        weight_decay=0.0,
    ):
        if not 0.0 < sf_beta1 < 1.0:
            raise ValueError("sf_beta1 must be between 0 and 1")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if r < 0.0 or weight_lr_power < 0.0:
            raise ValueError("Schedule-Free weighting values must be non-negative")
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
            backend="torch",
        )
        for group in self.param_groups:
            group.update(
                sf_beta1=float(sf_beta1),
                warmup_steps=int(warmup_steps),
                sf_r=float(r),
                sf_weight_lr_power=float(weight_lr_power),
                k=0,
                train_mode=False,
                weight_sum=0.0,
                lr_max=-1.0,
                scheduled_lr=0.0,
            )

    @staticmethod
    def _apply_to_parameter(
        parameter: torch.Tensor,
        callback: Callable[[torch.Tensor], None],
    ) -> None:
        CAMELRSF._apply_to_parameter(parameter, callback)

    def _came_update(self, parameter, grad, state, group):
        return CAMELRSF._came_update(self, parameter, grad, state, group)

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            restore_scale = 1.0 / group["sf_beta1"] - 1.0
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if not state or "sf_delta" not in state:
                    continue
                self._apply_to_parameter(
                    parameter,
                    lambda matrix, state=state: matrix.add_(
                        state["sf_delta"].reshape(matrix.shape),
                        alpha=restore_scale,
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
                if not state or "sf_delta" not in state:
                    continue
                self._apply_to_parameter(
                    parameter,
                    lambda matrix, state=state: matrix.add_(
                        state["sf_delta"].reshape(matrix.shape),
                        alpha=eval_scale,
                    ),
                )
            group["train_mode"] = False

    def load_state_dict(self, state_dict) -> None:
        super().load_state_dict(state_dict)
        for state in self.state.values():
            backend = state.get("backend")
            if backend not in {"came", "sf_full"}:
                state["backend"] = "sf_full" if "sf_delta" in state else "came"

    @torch.no_grad()
    def step(
        self, closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        if any(not group["train_mode"] for group in self.param_groups):
            raise RuntimeError("CAME-SF requires optimizer.train() before step().")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta_sf = group["sf_beta1"]
            k = int(group["k"])
            warmup_steps = group["warmup_steps"]
            sched = (k + 1) / warmup_steps if k < warmup_steps else 1.0
            lr = group["lr"] * sched
            group["scheduled_lr"] = lr
            group["lr_max"] = max(lr, group["lr_max"])
            weight = ((k + 1) ** group["sf_r"]) * (
                group["lr_max"] ** group["sf_weight_lr_power"]
            )
            group["weight_sum"] += weight
            ckp1 = weight / group["weight_sum"] if group["weight_sum"] else 0.0

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if "sf_delta" not in state:
                    state["sf_delta"] = torch.zeros_like(
                        parameter,
                    )
                    state["backend"] = "sf_full"
                elif state["sf_delta"].dtype != parameter.dtype:
                    state["sf_delta"] = state["sf_delta"].to(
                        dtype=parameter.dtype,
                    )
                update = self._came_update(parameter, parameter.grad, state, group)
                if group["weight_decay"] != 0.0:
                    effective_update = update.add(
                        parameter.float(), alpha=group["weight_decay"],
                    )
                else:
                    effective_update = update
                update_scale = lr * (beta_sf * (1.0 - ckp1) - 1.0)
                delta_scale = -(1.0 - ckp1) * lr * beta_sf

                def apply_update(matrix):
                    matrix.add_(
                        state["sf_delta"].reshape(matrix.shape), alpha=ckp1,
                    )
                    matrix.add_(
                        effective_update.reshape(matrix.shape),
                        alpha=update_scale,
                    )

                self._apply_to_parameter(parameter, apply_update)
                state["sf_delta"].mul_(1.0 - ckp1).add_(
                    effective_update, alpha=delta_scale,
                )
            group["k"] = k + 1
        return loss


__all__ = ["CAMELRSF", "CAMESF"]
