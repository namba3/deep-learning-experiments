"""Memory-efficient APOLLO optimizers.

This module is an independent PyTorch implementation of the APOLLO update
rule described in ``APOLLO: SGD-like Memory, AdamW-level Performance``.
Unlike the reference implementation, the projection is kept as ordinary
tensor state so optimizer checkpoints remain serializable without storing a
Python projector object.

APOLLO keeps Adam moments only in a random low-rank space.  The resulting
scaling is applied to the original full-rank gradient.  ``APOLLOMini`` uses
the same rank-one auxiliary state but applies one scale to the whole tensor.
Fallbacks are selected per parameter: the default policy routes 1D parameters
to AdamW-SF and sends matrices to AdamW-SF only when its estimated persistent
state is smaller than APOLLO's. APOLLO is intended primarily for matrix-like
weights.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, cast

import torch
from torch.optim import Optimizer

from .auto_schedule import AutoScheduleMixin
from .apollo_triton import fused_came_adaptive_update
from .projection_refresh import (
    OrthogonalRefreshPolicy,
    ProjectionRefreshPolicy,
    advance_refresh_mix,
    initialize_refresh_mix,
    prepare_stochastic_refresh,
    refresh_mix_weight,
    rotate_orthogonal_projection,
)
from .update_norm import cap_update_norm_variance_scale


@dataclass(frozen=True)
class APOLLOFallbackPolicy:
    """Per-parameter fallback policy for APOLLO.

    The legacy string form (``fallback="came"``) remains a one-dimensional
    fallback only. A policy object can select a matrix fallback by comparing
    estimated persistent optimizer-state sizes.
    """

    one_dimensional: str = "adamw-sf"
    small_matrix: str = "auto-sf"
    state_margin: float = 1.0
    min_savings_bytes: int = 0

    def __post_init__(self) -> None:
        if self.one_dimensional not in {"came", "sgd", "adamw-sf"}:
            raise ValueError(
                "one_dimensional must be 'came', 'sgd', or 'adamw-sf'"
            )
        if self.small_matrix not in {
            "apollo", "came", "auto", "adamw-sf", "auto-sf",
        }:
            raise ValueError(
                "small_matrix must be 'apollo', 'came', 'auto', 'adamw-sf', "
                "or 'auto-sf'"
            )
        if self.state_margin <= 0.0:
            raise ValueError("state_margin must be positive")
        if self.min_savings_bytes < 0:
            raise ValueError("min_savings_bytes must be non-negative")

    @classmethod
    def from_value(
        cls,
        value: "APOLLOFallbackPolicy | Mapping[str, object] | str | None",
    ) -> "APOLLOFallbackPolicy":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            # Preserve the old public contract: a string controlled only the
            # unsupported 1D path and never changed matrix parameters.
            return cls(one_dimensional=value, small_matrix="apollo")
        if not isinstance(value, Mapping):
            raise TypeError(
                "fallback must be a string, mapping, or APOLLOFallbackPolicy"
            )

        known = {
            "1d", "one_dimensional", "small_matrix", "state_margin",
            "min_savings_bytes",
        }
        unknown = set(value) - known
        if unknown:
            raise ValueError(
                "unknown APOLLO fallback keys: "
                + ", ".join(sorted(str(key) for key in unknown))
            )
        one_dimensional = value.get(
            "one_dimensional", value.get("1d", "adamw-sf")
        )
        small_matrix = value.get("small_matrix", "auto-sf")
        state_margin = value.get("state_margin", 1.0)
        min_savings_bytes = value.get("min_savings_bytes", 0)
        if not isinstance(one_dimensional, str):
            raise TypeError("fallback['1d'] must be a string")
        if not isinstance(small_matrix, str):
            raise TypeError("fallback['small_matrix'] must be a string")
        if not isinstance(state_margin, (int, float)):
            raise TypeError("fallback['state_margin'] must be numeric")
        if not isinstance(min_savings_bytes, int):
            raise TypeError("fallback['min_savings_bytes'] must be an integer")
        return cls(
            one_dimensional=one_dimensional,
            small_matrix=small_matrix,
            state_margin=float(state_margin),
            min_savings_bytes=min_savings_bytes,
        )

    def as_dict(self) -> dict[str, object]:
        """Return a checkpoint-safe representation."""
        return {
            "1d": self.one_dimensional,
            "small_matrix": self.small_matrix,
            "state_margin": self.state_margin,
            "min_savings_bytes": self.min_savings_bytes,
        }


def _stable_seed(seed: int) -> int:
    """Return a deterministic next seed without touching the global RNG."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return int(
        torch.randint(
            0,
            torch.iinfo(torch.int64).max,
            (1,),
            generator=generator,
        ).item()
    )


class _APOLLOBase(AutoScheduleMixin, Optimizer):
    """Shared implementation for channel-wise and tensor-wise APOLLO."""

    scale_type = "channel"

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1e-3,
        rank: int = 8,
        scale: float = 1.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        update_proj_gap: int = 200,
        seed: int = 0,
        scale_front: bool = False,
        norm_growth_limiter: bool = False,
        norm_growth_rate: float = 1.01,
        projection_refresh_state: str = "reset",
        projection_refresh=None,
        orthogonal_refresh=None,
        update_norm_variance_cap: float | None = None,
        fallback: APOLLOFallbackPolicy | Mapping[str, object] | str | None = None,
    ):
        if lr < 0.0:
            raise ValueError("lr must be non-negative")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        if len(betas) != 2 or not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if eps < 0.0:
            raise ValueError("eps must be non-negative")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if update_proj_gap <= 0:
            raise ValueError("update_proj_gap must be positive")
        if norm_growth_rate <= 1.0:
            raise ValueError("norm_growth_rate must be greater than 1")
        if update_norm_variance_cap is not None and (
            not math.isfinite(update_norm_variance_cap)
            or update_norm_variance_cap < 0.0
        ):
            raise ValueError(
                "update_norm_variance_cap must be finite and non-negative"
            )
        if projection_refresh_state not in {"reset", "transport"}:
            raise ValueError(
                "projection_refresh_state must be 'reset' or 'transport'"
            )
        refresh_policy = ProjectionRefreshPolicy.from_value(
            projection_refresh
            if projection_refresh is not None
            else {"mode": "hard", "interval": int(update_proj_gap)}
        )
        orthogonal_policy = OrthogonalRefreshPolicy.from_value(
            orthogonal_refresh, default_seed=int(seed),
        )
        fallback_policy = APOLLOFallbackPolicy.from_value(fallback)

        defaults = dict(
            lr=float(lr),
            rank=int(rank),
            scale=float(scale),
            betas=tuple(betas),
            eps=float(eps),
            weight_decay=float(weight_decay),
            update_proj_gap=int(update_proj_gap),
            seed=int(seed),
            scale_front=bool(scale_front),
            norm_growth_limiter=bool(norm_growth_limiter),
            norm_growth_rate=float(norm_growth_rate),
            projection_refresh_state=projection_refresh_state,
            projection_refresh=refresh_policy.as_dict(),
            orthogonal_refresh=orthogonal_policy.as_dict(),
            update_norm_variance_cap=(
                None if update_norm_variance_cap is None
                else float(update_norm_variance_cap)
            ),
            fallback=fallback_policy.one_dimensional,
            fallback_policy=fallback_policy.as_dict(),
            sf_k=0,
            sf_lr_max=0.0,
            sf_weight_sum=0.0,
            sf_train_mode=True,
        )
        super().__init__(params, defaults)
        self.fallback_policy = fallback_policy

    @torch.no_grad()
    def train(self):
        """Switch AdamW-SF fallback parameters to their training weights."""
        for group in self.param_groups:
            if not any(
                self._is_adamw_sf_fallback(parameter, group)
                for parameter in group["params"]
            ):
                continue
            if group["sf_train_mode"]:
                continue
            beta1 = float(group["betas"][0])
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if state is not None and state.get("backend") == "adamw-sf":
                    z = state.get("z")
                    if z is not None:
                        parameter.lerp_(
                            z.to(device=parameter.device, dtype=parameter.dtype),
                            1.0 - beta1,
                        )
            group["sf_train_mode"] = True
        return self

    @torch.no_grad()
    def eval(self):
        """Switch AdamW-SF fallback parameters to averaged evaluation weights."""
        for group in self.param_groups:
            if not any(
                self._is_adamw_sf_fallback(parameter, group)
                for parameter in group["params"]
            ):
                continue
            if not group["sf_train_mode"]:
                continue
            beta1 = float(group["betas"][0])
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if state is not None and state.get("backend") == "adamw-sf":
                    z = state.get("z")
                    if z is not None:
                        parameter.lerp_(
                            z.to(device=parameter.device, dtype=parameter.dtype),
                            1.0 - 1.0 / beta1,
                        )
            group["sf_train_mode"] = False
        return self

    def load_state_dict(self, state_dict) -> None:
        """Load optimizer state and repair PyTorch's Iterable string cast.

        Some PyTorch versions treat the non-tensor ``backend`` string as an
        iterable while casting optimizer state, replacing it with a generator
        representation.  The state tensor layout is authoritative, so recover
        the backend from its keys after the standard load operation.
        """
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            group.setdefault("sf_k", 0)
            group.setdefault("sf_lr_max", 0.0)
            group.setdefault("sf_weight_sum", 0.0)
            group.setdefault("sf_train_mode", True)
            policy = self._fallback_policy_for_group(group)
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if not state:
                    continue
                backend = state.get("backend")
                if backend in {"apollo", "came", "sgd", "adamw-sf"}:
                    continue
                state.pop("backend", None)
                if parameter.ndim >= 2 and "projection" in state:
                    state["backend"] = "apollo"
                elif parameter.ndim >= 2 and "exp_avg_sq_row" in state:
                    state["backend"] = "came"
                elif "z" in state and "exp_avg_sq" in state:
                    state["backend"] = "adamw-sf"
                else:
                    state["backend"] = (
                        policy.one_dimensional
                        if parameter.ndim < 2
                        else policy.small_matrix
                    )
                    if state["backend"] in {"auto", "auto-sf"}:
                        state.pop("backend", None)
                        self._select_backend(parameter, state, group)

    @staticmethod
    def _matrix_view(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1)

    @staticmethod
    def _parameter_group_for(parameter: torch.Tensor, groups: list[dict]) -> dict:
        for group in groups:
            if any(parameter is candidate for candidate in group["params"]):
                return group
        raise ValueError("parameter is not present in an APOLLO parameter group")

    @staticmethod
    def _matrix_dimensions(parameter: torch.Tensor) -> tuple[int, int]:
        matrix = _APOLLOBase._matrix_view(parameter)
        return matrix.shape[0], matrix.shape[1]

    @staticmethod
    def _ensure_parameter_dtype_state(
        parameter: torch.Tensor, state: dict, keys: tuple[str, ...],
    ) -> None:
        """Keep full-size fallback moments in parameter storage dtype."""
        for key in keys:
            value = state.get(key)
            if value is not None and value.dtype != parameter.dtype:
                state[key] = value.to(dtype=parameter.dtype)

    @staticmethod
    def _came_state_elements(
        parameter: torch.Tensor,
        *,
        include_limiter: bool = False,
    ) -> int:
        """Estimate tensor elements in the full CAME state.

        The row/column factors follow CAME's original parameter shape.  This
        differs from APOLLO, which flattens all axes after axis zero.
        """
        numel = parameter.numel()
        limiter = int(include_limiter)
        if parameter.ndim < 2:
            return 2 * numel + 1 + limiter  # exp_avg, exp_avg_sq, RMS
        row_numel = numel // parameter.shape[-1]
        col_numel = numel // parameter.shape[-2]
        return numel + 2 * (row_numel + col_numel) + 1 + limiter

    @classmethod
    def _came_state_bytes(
        cls,
        parameter: torch.Tensor,
        *,
        include_limiter: bool = False,
    ) -> int:
        elements = cls._came_state_elements(
            parameter, include_limiter=include_limiter,
        )
        # Full-size moments follow the parameter dtype.  CAME factor tensors
        # and the APOLLO vector fallback's limiter scalar remain FP32.
        if parameter.ndim < 2:
            # The legacy APOLLO vector fallback is CAME-like rather than the
            # reference CAME vector state: it has two full-size moments and
            # no RMS entry.  Its optional scalar is the APOLLO limiter.
            return (
                2 * parameter.numel() * parameter.element_size()
                + int(include_limiter) * 4
            )
        if parameter.ndim >= 2 and not include_limiter:
            return (elements - 1) * 4 + parameter.element_size()
        return elements * 4

    def _is_apollo_came(self, group: dict) -> bool:
        return "came_betas" in group

    def _apollo_state_elements(self, parameter: torch.Tensor, group: dict) -> int:
        rows, cols = self._matrix_dimensions(parameter)
        rank = min(int(group["rank"]), min(rows, cols))
        projection = rank * min(rows, cols)
        limiter = int(bool(group["norm_growth_limiter"]))
        if self._is_apollo_came(group):
            # APOLLO-CAME keeps the low-rank input/work buffers in state in
            # addition to exp_avg and the factored CAME statistics.
            return 3 * rank * max(rows, cols) + 2 * (
                max(rows, cols) + rank
            ) + projection + limiter
        return 2 * rank * max(rows, cols) + projection + limiter

    @staticmethod
    def _fallback_policy_for_group(group: dict) -> APOLLOFallbackPolicy:
        return APOLLOFallbackPolicy.from_value(
            group.get("fallback_policy", group.get("fallback", "came"))
        )

    def _is_adamw_sf_fallback(self, parameter: torch.Tensor, group: dict) -> bool:
        state = self.state.get(parameter, {})
        return self._select_backend(parameter, state, group) == "adamw-sf"

    @staticmethod
    def _ensure_adamw_sf_state(parameter: torch.Tensor, state: dict) -> None:
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

    @classmethod
    @torch.no_grad()
    def _adamw_sf_fallback_update(
        cls,
        parameter: torch.Tensor,
        gradient: torch.Tensor,
        state: dict,
        group: dict,
        *,
        lr: float,
        k: int,
        ckp1: float,
    ) -> None:
        """Apply the reference full-state AdamW-SF update to one fallback."""
        beta1, beta2 = group["betas"]
        cls._ensure_adamw_sf_state(parameter, state)
        z = state["z"]
        exp_avg_sq = state["exp_avg_sq"]
        grad = gradient.float()
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        bias_correction2 = 1.0 - beta2 ** (k + 1)
        denom = exp_avg_sq.float().div(bias_correction2).sqrt_().add_(group["eps"])
        grad_normalized = grad.div(denom)
        y = parameter.float()
        if group["weight_decay"] != 0.0:
            grad_normalized.add_(y, alpha=group["weight_decay"])
        y.lerp_(z.float(), weight=ckp1)
        y.add_(grad_normalized, alpha=lr * (beta1 * (1.0 - ckp1) - 1.0))
        z.sub_(grad_normalized, alpha=lr)
        parameter.copy_(y.to(dtype=parameter.dtype))

    def _select_backend(
        self,
        parameter: torch.Tensor,
        state: dict,
        group: dict,
    ) -> str:
        """Select a stable backend before allocating optimizer state."""
        existing = state.get("backend")
        if existing in {"apollo", "came", "sgd", "adamw-sf"}:
            return existing
        # Old checkpoints have no backend marker.  Existing APOLLO matrix
        # state must remain on APOLLO rather than being silently migrated.
        if parameter.ndim >= 2 and "projection" in state:
            state["backend"] = "apollo"
            return "apollo"

        policy = self._fallback_policy_for_group(group)
        if parameter.ndim < 2:
            backend = policy.one_dimensional
        elif policy.small_matrix not in {"auto", "auto-sf"}:
            backend = policy.small_matrix
        else:
            apollo_elements = self._apollo_state_elements(parameter, group)
            apollo_bytes = apollo_elements * 4
            if policy.small_matrix == "auto-sf":
                fallback_bytes = 2 * parameter.numel() * parameter.element_size()
                fallback_backend = "adamw-sf"
            else:
                # Full CAME does not use APOLLO's norm limiter. Compare
                # against its actual state, not an APOLLO-only scalar.
                fallback_bytes = self._came_state_bytes(parameter)
                fallback_backend = "came"
            backend = fallback_backend if (
                fallback_bytes + policy.min_savings_bytes
                <= apollo_bytes * policy.state_margin
            ) else "apollo"
        state["backend"] = backend
        return backend

    def estimate_parameter_state_bytes(self, parameter: torch.Tensor) -> int:
        """Estimate persistent state bytes for one parameter."""
        group = self._parameter_group_for(parameter, self.param_groups)
        state = self.state.get(parameter, {})
        backend = self._select_backend(parameter, state, group)
        estimated_bytes: int
        if backend == "came":
            estimated_bytes = self._came_state_bytes(
                parameter,
                include_limiter=parameter.ndim < 2
                and bool(group["norm_growth_limiter"]),
            )
        elif backend == "sgd":
            estimated_bytes = int(bool(group["norm_growth_limiter"])) * 4
        elif backend == "adamw-sf":
            estimated_bytes = 2 * parameter.numel() * parameter.element_size()
        else:
            estimated_bytes = self._apollo_state_elements(parameter, group) * 4

        # Smooth refresh temporarily owns a second projection and moment
        # branch.  The static formula above describes the steady state; add
        # the live transient tensors so the estimate remains comparable with
        # state_metrics() while the PA/PB transition is active.
        if state.get("refresh_active", False):
            estimated_bytes += sum(
                value.numel() * value.element_size()
                for key, value in state.items()
                if key.startswith("refresh_next_")
                and isinstance(value, torch.Tensor)
            )
        if group.get("update_norm_variance_cap") is not None:
            estimated_bytes += 3 * 4
        return estimated_bytes

    @staticmethod
    def _projection_shape(rows: int, cols: int, rank: int) -> tuple[int, ...]:
        if rows >= cols:
            return cols, rank
        return rank, rows

    @staticmethod
    def _project(
        grad_matrix: torch.Tensor,
        projection: torch.Tensor,
    ) -> torch.Tensor:
        rows, cols = grad_matrix.shape
        if rows >= cols:
            return grad_matrix.matmul(projection)
        return projection.matmul(grad_matrix)

    @staticmethod
    def _project_into(
        grad_matrix: torch.Tensor,
        projection: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Project into a caller-owned 2D buffer without allocating output."""
        rows, cols = grad_matrix.shape
        if rows >= cols:
            torch.mm(grad_matrix, projection, out=output)
        else:
            torch.mm(projection, grad_matrix, out=output)
        return output

    @staticmethod
    def _make_projection(
        parameter: torch.Tensor,
        rank: int,
        seed: int,
    ) -> torch.Tensor:
        matrix = _APOLLOBase._matrix_view(parameter)
        shape = _APOLLOBase._projection_shape(
            matrix.shape[0], matrix.shape[1], rank
        )
        generator = torch.Generator(device=parameter.device).manual_seed(int(seed))
        projection = torch.randn(
            shape,
            generator=generator,
            device=parameter.device,
            dtype=torch.float32,
        )
        return projection.div_(math.sqrt(rank))

    def _ensure_state(
        self,
        parameter: torch.Tensor,
        state: dict,
        group: dict,
    ) -> None:
        if "step" not in state:
            state["step"] = 0
            state["projection_seed"] = int(group["seed"])

        if parameter.ndim < 2:
            return

        matrix = self._matrix_view(parameter)
        rank = min(int(group["rank"]), min(matrix.shape))
        if rank <= 0:
            return
        if "projection" not in state or state["projection"].shape != self._projection_shape(
            matrix.shape[0], matrix.shape[1], rank
        ):
            state["projection"] = self._make_projection(
                parameter, rank, state["projection_seed"]
            )
            state["projection_rank"] = rank

    def _refresh_projection(
        self,
        parameter: torch.Tensor,
        state: dict,
        group: dict,
    ) -> bool:
        if parameter.ndim < 2:
            return False
        configured_policy = group.get("projection_refresh")
        policy = ProjectionRefreshPolicy.from_value(
            configured_policy
            if configured_policy is not None
            else {
                "mode": "hard",
                "interval": int(group.get("update_proj_gap", 200)),
            }
        )
        step = int(state["step"])
        if (
            policy.mode != "none"
            and not state.get("refresh_active", False)
            and step > 1
            and step % policy.interval == 0
        ):
            rank = int(state.get("projection_rank", group["rank"]))
            old_projection = state["projection"]
            next_seed = _stable_seed(state["projection_seed"])
            new_projection = self._make_projection(
                parameter, rank, next_seed
            )
            if policy.mode == "smooth":
                self._start_smooth_projection_refresh(
                    state,
                    old_projection,
                    new_projection,
                    next_seed=next_seed,
                    rows_ge_cols=(
                        self._matrix_dimensions(parameter)[0]
                        >= self._matrix_dimensions(parameter)[1]
                    ),
                    transport=(
                        group.get("projection_refresh_state", "reset")
                        == "transport"
                    ),
                    policy=policy,
                )
                state["projection_seed"] = next_seed
                return True
            if group.get("projection_refresh_state", "reset") == "transport":
                self._transport_projection_state(
                    state, old_projection, new_projection,
                    rows_ge_cols=self._matrix_dimensions(parameter)[0]
                    >= self._matrix_dimensions(parameter)[1],
                )
            state["projection"] = new_projection
            state["projection_seed"] = _stable_seed(state["projection_seed"])
            return True
        return False

    def _rotate_orthogonal_projection_state(
        self,
        state: dict,
        group: dict,
        *,
        parameter: torch.Tensor,
    ) -> bool:
        """Apply one tangent-space rotation and transport APOLLO moments."""
        policy = OrthogonalRefreshPolicy.from_value(
            group.get("orthogonal_refresh"),
        )
        if policy.rate == 0.0 or "projection" not in state:
            return False
        if policy.direction == "loss_lowering":
            raise ValueError(
                "loss_lowering orthogonal refresh is only supported for LRSF"
            )
        count = int(state.get("orthogonal_refresh_count", 0))
        seed = int(policy.seed) + 2 * count
        old_projection = state["projection"]
        new_projection = rotate_orthogonal_projection(
            old_projection,
            policy.rate,
            seed,
            direction=policy.direction,
            gradient=parameter.grad,
        )
        rows, cols = self._matrix_dimensions(parameter)
        self._transport_projection_state(
            state,
            old_projection,
            new_projection,
            rows_ge_cols=rows >= cols,
        )
        state["projection"] = new_projection

        if state.get("refresh_active", False):
            old_next_projection = state["refresh_next_projection"]
            new_next_projection = rotate_orthogonal_projection(
                old_next_projection,
                policy.rate,
                seed + 1,
                direction=policy.direction,
                gradient=parameter.grad,
            )
            next_state = {"projection": old_next_projection}
            for key in self._smooth_state_keys():
                next_key = f"refresh_next_{key}"
                if next_key in state:
                    next_state[key] = state[next_key]
            self._transport_projection_state(
                next_state,
                old_next_projection,
                new_next_projection,
                rows_ge_cols=rows >= cols,
            )
            state["refresh_next_projection"] = new_next_projection
            for key in self._smooth_state_keys():
                if key in next_state:
                    state[f"refresh_next_{key}"] = next_state[key]

        state["orthogonal_refresh_count"] = count + 1
        return True

    @staticmethod
    def _smooth_state_keys() -> tuple[str, ...]:
        return (
            "exp_avg",
            "exp_avg_sq",
            "exp_avg_sq_row",
            "exp_avg_sq_col",
            "exp_avg_res_row",
            "exp_avg_res_col",
            "came_low_rank_grad",
            "came_work",
        )

    def _start_smooth_projection_refresh(
        self,
        state: dict,
        old_projection: torch.Tensor,
        new_projection: torch.Tensor,
        *,
        next_seed: int,
        rows_ge_cols: bool,
        transport: bool,
        policy: ProjectionRefreshPolicy,
    ) -> None:
        """Create a second low-rank moment branch for a smooth refresh."""
        moment_keys = self._smooth_state_keys()[:6]
        if "exp_avg" not in state:
            return

        next_moments = {
            key: (
                state[key].detach().clone()
                if transport
                else torch.zeros_like(state[key])
            )
            for key in moment_keys
            if key in state
        }
        if transport:
            transported_state = {
                "exp_avg": next_moments["exp_avg"],
                "projection": old_projection,
            }
            for key in moment_keys:
                if key in next_moments:
                    transported_state[key] = next_moments[key]
            self._transport_projection_state(
                transported_state,
                old_projection,
                new_projection,
                rows_ge_cols=rows_ge_cols,
            )
            next_moments = {
                key: transported_state[key]
                for key in moment_keys
                if key in transported_state
            }

        state["refresh_next_projection"] = new_projection
        state["refresh_next_projection_seed"] = next_seed
        for key, value in next_moments.items():
            state[f"refresh_next_{key}"] = value
        state["refresh_progress"] = 0
        state["refresh_active"] = True
        initialize_refresh_mix(state, policy)

    def _smooth_low_rank_update(
        self,
        parameter: torch.Tensor,
        grad: torch.Tensor,
        state: dict,
        group: dict,
        grad_matrix: torch.Tensor,
        policy: ProjectionRefreshPolicy,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update PA/PB moments and combine their scaling factors."""
        low_rank_grad, scaling = self._low_rank_update(
            parameter, grad, state, group, grad_matrix=grad_matrix,
        )
        if not state.get("refresh_active", False):
            return low_rank_grad, scaling

        next_state = {
            "projection": state["refresh_next_projection"],
            "step": state["step"],
        }
        for key in self._smooth_state_keys():
            next_key = f"refresh_next_{key}"
            if next_key in state:
                next_state[key] = state[next_key]
        next_low_rank_grad, next_scaling = self._low_rank_update(
            parameter, grad, next_state, group, grad_matrix=grad_matrix,
        )
        for key in self._smooth_state_keys():
            if key in next_state:
                state[f"refresh_next_{key}"] = next_state[key]

        weight = refresh_mix_weight(state, policy)
        if policy.mix == "stochastic":
            if int(state.get("refresh_stochastic_choice", 0)):
                scaling = next_scaling
            return low_rank_grad, scaling
        scaling = scaling.mul(1.0 - weight).add_(next_scaling, alpha=weight)
        return low_rank_grad, scaling

    def _advance_smooth_projection_refresh(
        self, state: dict, policy: ProjectionRefreshPolicy,
    ) -> None:
        if not state.get("refresh_active", False):
            return
        advance_refresh_mix(state, policy)
        progress = int(state.get("refresh_progress", 0)) + 1
        if progress < policy.window:
            state["refresh_progress"] = progress
            return
        state["projection"] = state.pop("refresh_next_projection")
        state["projection_seed"] = state.pop("refresh_next_projection_seed")
        for key in self._smooth_state_keys():
            next_key = f"refresh_next_{key}"
            if next_key in state:
                state[key] = state.pop(next_key)
        state.pop("refresh_progress", None)
        state.pop("refresh_active", None)
        state.pop("refresh_ema_weight", None)

    @staticmethod
    def _transport_projection_state(
        state: dict,
        old_projection: torch.Tensor,
        new_projection: torch.Tensor,
        *,
        rows_ge_cols: bool,
    ) -> None:
        """Transport low-rank moments across an APOLLO projection refresh.

        ``reset`` is the historical behavior and intentionally discards the
        old coordinate system.  ``transport`` preserves the first moment and
        maps elementwise second moments through the squared overlap.  CAME's
        row factors stay in the unchanged parameter-axis coordinates while
        its low-rank column/row factors follow the projected axis.
        """
        if "exp_avg" not in state:
            return
        if rows_ge_cols:
            overlap = old_projection.transpose(0, 1).matmul(new_projection)
            state["exp_avg"] = state["exp_avg"].matmul(overlap)
            if "exp_avg_sq" in state:
                state["exp_avg_sq"] = state["exp_avg_sq"].matmul(
                    overlap.square()
                )
            for key in ("exp_avg_sq_col", "exp_avg_res_col"):
                if key in state:
                    state[key] = overlap.square().transpose(0, 1).matmul(
                        state[key]
                    )
        else:
            overlap = new_projection.matmul(old_projection.transpose(0, 1))
            state["exp_avg"] = overlap.matmul(state["exp_avg"])
            if "exp_avg_sq" in state:
                state["exp_avg_sq"] = overlap.square().matmul(
                    state["exp_avg_sq"]
                )
            for key in ("exp_avg_sq_row", "exp_avg_res_row"):
                if key in state:
                    state[key] = overlap.square().matmul(state[key])

    def _low_rank_update(
        self,
        parameter: torch.Tensor,
        grad: torch.Tensor,
        state: dict,
        group: dict,
        grad_matrix: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the low-rank gradient and its APOLLO scaling factor."""
        matrix = (
            grad_matrix
            if grad_matrix is not None
            else self._matrix_view(grad.float())
        )
        low_rank_grad = self._project(matrix, state["projection"].float())

        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(low_rank_grad)
            state["exp_avg_sq"] = torch.zeros_like(low_rank_grad)

        beta1, beta2 = group["betas"]
        state["exp_avg"].lerp_(low_rank_grad, 1.0 - beta1)
        # Equivalent to lerp_(low_rank_grad.square(), 1 - beta2), but avoids
        # materializing a separate low-rank square tensor.
        state["exp_avg_sq"].mul_(beta2).addcmul_(
            low_rank_grad, low_rank_grad, value=1.0 - beta2,
        )

        denominator = state["exp_avg_sq"].sqrt().add_(group["eps"])
        normalized = state["exp_avg"] / denominator

        step = int(state["step"])
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        normalized.mul_(math.sqrt(bias_correction2) / bias_correction1)

        rows, cols = matrix.shape
        if self.scale_type == "tensor":
            scaling = normalized.norm() / low_rank_grad.norm().add_(group["eps"])
            scaling = scaling.reshape(1)
        elif rows >= cols:
            scaling = normalized.norm(dim=1) / (
                low_rank_grad.norm(dim=1).add_(group["eps"])
            )
            scaling = scaling.reshape(rows, 1)
        else:
            scaling = normalized.norm(dim=0) / (
                low_rank_grad.norm(dim=0).add_(group["eps"])
            )
            scaling = scaling.reshape(1, cols)
        return low_rank_grad, scaling

    def _add_auto_schedule_low_rank_stats(
        self,
        stats: dict,
        state: dict,
    ) -> None:
        """Add optimizer-specific low-rank stability statistics."""

    @staticmethod
    def _apply_scaling(
        grad_matrix: torch.Tensor,
        scaling: torch.Tensor,
        scale_type: str,
        output_shape: torch.Size,
    ) -> torch.Tensor:
        if scale_type == "tensor":
            return (
                grad_matrix
                * scaling.reshape(() if scaling.numel() == 1 else scaling.shape)
            ).reshape(output_shape)
        return (grad_matrix * scaling).reshape(output_shape)

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float().square().mean().sqrt()

    @staticmethod
    def _rms_without_square_buffer(tensor: torch.Tensor) -> torch.Tensor:
        """Compute RMS without materializing a full-size squared tensor."""
        tensor = tensor.float()
        return torch.linalg.vector_norm(tensor) / math.sqrt(tensor.numel())

    @staticmethod
    def _came_approx_sq_grad(
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
    ) -> torch.Tensor:
        row_factor = (
            exp_avg_sq_row
            / exp_avg_sq_row.mean(dim=-1, keepdim=True).clamp_min(1e-30)
        ).rsqrt_().unsqueeze(-1)
        col_factor = exp_avg_sq_col.clamp_min(1e-30).rsqrt().unsqueeze(-2)
        return row_factor * col_factor

    def _full_came_matrix_update(
        self,
        parameter: torch.Tensor,
        grad: torch.Tensor,
        state: dict,
        group: dict,
    ) -> torch.Tensor:
        """Return a full CAME update for a size-based matrix fallback.

        This is deliberately separate from ``_fallback_scale``: that method
        is the existing unfactored 1D path, while this path preserves CAME's
        row/column second moments and residual confidence statistics.
        """
        gradient = grad.float()
        beta1, beta2, beta3 = group.get(
            "came_betas", (0.9, 0.999, 0.9999)
        )
        eps_square, eps_instability = group.get(
            "came_eps", (1e-30, group["eps"])
        )
        clip_threshold = group.get("clip_threshold", 1.0)

        state.setdefault("step", 0)
        state["step"] = int(state["step"]) + 1
        # CAME keeps this scalar in the parameter dtype.  The update
        # statistics remain FP32, but preserving the scalar dtype makes a
        # matrix fallback's checkpoint state match CAME in BF16 as well.
        state["RMS"] = self._rms_without_square_buffer(parameter).to(
            dtype=parameter.dtype
        )

        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq_row"] = torch.zeros(
                gradient.shape[:-1],
                device=gradient.device,
                dtype=gradient.dtype,
            )
            state["exp_avg_sq_col"] = torch.zeros(
                gradient.shape[:-2] + gradient.shape[-1:],
                device=gradient.device,
                dtype=gradient.dtype,
            )
            state["exp_avg_res_row"] = torch.zeros(
                gradient.shape[:-1],
                device=gradient.device,
                dtype=gradient.dtype,
            )
            state["exp_avg_res_col"] = torch.zeros(
                gradient.shape[:-2] + gradient.shape[-1:],
                device=gradient.device,
                dtype=gradient.dtype,
            )
        self._ensure_parameter_dtype_state(parameter, state, ("exp_avg",))

        second_moment = gradient.square().add_(eps_square)
        state["exp_avg_sq_row"].mul_(beta2).add_(
            second_moment.mean(dim=-1), alpha=1.0 - beta2
        )
        state["exp_avg_sq_col"].mul_(beta2).add_(
            second_moment.mean(dim=-2), alpha=1.0 - beta2
        )
        update = self._came_approx_sq_grad(
            state["exp_avg_sq_row"], state["exp_avg_sq_col"]
        ).mul(gradient)
        update.div_(
            (self._rms_without_square_buffer(update) / clip_threshold)
            .clamp_min(1.0)
        )

        state["exp_avg"].mul_(beta1).add_(
            update, alpha=1.0 - beta1
        )
        residual = update.sub(state["exp_avg"]).square_().add_(
            eps_instability
        )
        state["exp_avg_res_row"].mul_(beta3).add_(
            residual.mean(dim=-1), alpha=1.0 - beta3
        )
        state["exp_avg_res_col"].mul_(beta3).add_(
            residual.mean(dim=-2), alpha=1.0 - beta3
        )
        return self._came_approx_sq_grad(
            state["exp_avg_res_row"], state["exp_avg_res_col"]
        ).mul_(state["exp_avg"])

    def _fallback_scale(
        self,
        parameter: torch.Tensor,
        grad: torch.Tensor,
        state: dict,
        group: dict,
        *,
        materialize_update: bool = False,
    ) -> torch.Tensor:
        """Use CAME's unfactored update for bias/norm vectors by default."""
        if group["fallback"] == "sgd":
            return grad.float()

        beta1, beta2 = group["betas"]
        eps_square = group.get("came_eps", (1e-30, group["eps"]))[0]
        clip_threshold = group.get("clip_threshold", 1.0)
        grad = grad.float()
        if "fallback_exp_avg" not in state:
            state["fallback_exp_avg"] = torch.zeros_like(
                grad, dtype=parameter.dtype,
            )
            state["fallback_exp_avg_sq"] = torch.zeros_like(
                grad, dtype=parameter.dtype,
            )
        self._ensure_parameter_dtype_state(
            parameter, state, ("fallback_exp_avg", "fallback_exp_avg_sq"),
        )
        exp_avg = state["fallback_exp_avg"]
        exp_avg_sq = state["fallback_exp_avg_sq"]
        # Equivalent to updating with ``grad.square() + eps_square`` while
        # avoiding a full-size temporary for that expression.  The scalar
        # epsilon term is kept separate so the CAME update remains unchanged
        # mathematically.
        one_minus_beta2 = 1.0 - beta2
        exp_avg_sq.mul_(beta2).addcmul_(
            grad, grad, value=one_minus_beta2,
        )
        if eps_square != 0.0:
            exp_avg_sq.add_(one_minus_beta2 * eps_square)
        update = exp_avg_sq.rsqrt().mul_(grad)
        update.div_(
            (
                self._rms_without_square_buffer(update) / clip_threshold
            ).clamp_min(1.0)
        )
        exp_avg.mul_(beta1).add_(update, alpha=1.0 - beta1)
        # Ordinary APOLLO uses scalar factors at parameter.apply time and does
        # not mutate the fallback update. Return the EMA directly in that
        # case, avoiding a full-size copy and an extra persistent state tensor.
        # AutoSchedule needs a materialized update for its update-norm
        # statistics and for the existing in-place scale path, so retain the
        # reusable work buffer there.
        if not materialize_update:
            return exp_avg

        fallback_update = state.get("fallback_update")
        if (
            fallback_update is None
            or fallback_update.shape != exp_avg.shape
            or fallback_update.dtype != exp_avg.dtype
            or fallback_update.device != exp_avg.device
        ):
            fallback_update = torch.empty_like(exp_avg)
            state["fallback_update"] = fallback_update
        fallback_update.copy_(exp_avg)
        return fallback_update

    def _step_vector_came_foreach(
        self,
        parameters: list[torch.Tensor],
        group: dict,
        add_metric: Callable[[str, float], None],
        measure_optimizer: Callable[[str], object],
    ) -> None:
        """Update ordinary CAME fallback vectors with foreach operations.

        Vector parameters are numerous and individually small in the image
        model.  Their CAME state update is independent, so batching the common
        elementwise operations reduces Python dispatch and CUDA kernel launch
        overhead.  The limiter and parameter application remain per tensor to
        preserve the existing state and scaling semantics.
        """
        states: list[dict] = []
        gradients: list[torch.Tensor] = []
        exp_avgs: list[torch.Tensor] = []
        exp_avg_sqs: list[torch.Tensor] = []

        for parameter in parameters:
            gradient = parameter.grad
            if gradient is None or gradient.is_sparse:
                raise RuntimeError("APOLLO does not support sparse gradients")
            state = self.state[parameter]
            state["backend"] = "came"
            self._ensure_state(parameter, state, group)
            state["step"] = int(state.get("step", 0)) + 1
            gradient = gradient.float()
            if "fallback_exp_avg" not in state:
                state["fallback_exp_avg"] = torch.zeros_like(
                    gradient, dtype=parameter.dtype,
                )
                state["fallback_exp_avg_sq"] = torch.zeros_like(
                    gradient, dtype=parameter.dtype,
                )
            self._ensure_parameter_dtype_state(
                parameter, state,
                ("fallback_exp_avg", "fallback_exp_avg_sq"),
            )
            states.append(state)
            gradients.append(gradient)
            exp_avgs.append(state["fallback_exp_avg"])
            exp_avg_sqs.append(state["fallback_exp_avg_sq"])
            add_metric("apollo_parameter_tensors")
            add_metric("apollo_gradient_elements", gradient.numel())
            add_metric("apollo_vector_parameters")
            if group["weight_decay"] != 0.0:
                add_metric("apollo_weight_decay_parameters")

        beta1, beta2 = group["betas"]
        eps_square = group.get("came_eps", (1e-30, group["eps"]))[0]
        clip_threshold = group.get("clip_threshold", 1.0)
        one_minus_beta2 = 1.0 - beta2

        with measure_optimizer("apollo_fallback_update"):
            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(
                exp_avg_sqs,
                gradients,
                gradients,
                value=one_minus_beta2,
            )
            if eps_square != 0.0:
                torch._foreach_add_(
                    exp_avg_sqs, one_minus_beta2 * eps_square,
                )
            inverse_rms = torch._foreach_rsqrt(exp_avg_sqs)
            updates = torch._foreach_mul(inverse_rms, gradients)
            for update in updates:
                update.div_(
                    (
                        self._rms_without_square_buffer(update)
                        / clip_threshold
                    ).clamp_min(1.0)
                )
            torch._foreach_mul_(exp_avgs, beta1)
            torch._foreach_add_(
                exp_avgs, updates, alpha=1.0 - beta1,
            )

        lr = group["lr"]
        scale_factor = math.sqrt(group["scale"])
        scale_is_identity = scale_factor == 1.0
        with measure_optimizer("apollo_update_apply"):
            with measure_optimizer("apollo_limiter"):
                limiter_ratios: list[Optional[torch.Tensor]] = []
                if group["norm_growth_limiter"]:
                    current_norms = torch._foreach_norm(exp_avgs)
                    for current_norm, state in zip(current_norms, states):
                        previous_norm = state.get("scaled_grad_norm")
                        if previous_norm is None:
                            limiter_ratios.append(None)
                            state["scaled_grad_norm"] = current_norm.detach()
                            continue
                        max_norm = previous_norm.mul_(
                            group["norm_growth_rate"]
                        )
                        limiter_ratio = max_norm.div(
                            current_norm.clamp_min(group["eps"])
                        ).clamp_max_(1.0)
                        limiter_ratios.append(limiter_ratio)
                        torch.minimum(
                            current_norm, max_norm, out=current_norm,
                        )
                        state["scaled_grad_norm"] = current_norm.detach()
                else:
                    limiter_ratios = [None] * len(exp_avgs)

            with measure_optimizer("apollo_parameter_apply"):
                parameter_foreach_compatible = all(
                    parameter.device == parameters[0].device
                    and parameter.dtype == parameters[0].dtype
                    for parameter in parameters
                )
                if parameter_foreach_compatible:
                    if group["weight_decay"] != 0.0:
                        torch._foreach_mul_(
                            parameters, 1.0 - lr * group["weight_decay"]
                        )
                    no_limiter = [
                        index
                        for index, ratio in enumerate(limiter_ratios)
                        if ratio is None
                    ]
                    with_limiter = [
                        index
                        for index, ratio in enumerate(limiter_ratios)
                        if ratio is not None
                    ]
                    if no_limiter:
                        torch._foreach_add_(
                            [parameters[index] for index in no_limiter],
                            [exp_avgs[index] for index in no_limiter],
                            alpha=-lr * scale_factor,
                        )
                    if with_limiter:
                        ratios = [
                            cast(torch.Tensor, limiter_ratios[index])
                            for index in with_limiter
                        ]
                        if group["scale_front"] and not scale_is_identity:
                            ratios = [ratio * scale_factor for ratio in ratios]
                        torch._foreach_addcmul_(
                            [parameters[index] for index in with_limiter],
                            [exp_avgs[index] for index in with_limiter],
                            ratios,
                            value=-lr,
                        )
                else:
                    for parameter, update, limiter_ratio in zip(
                        parameters, exp_avgs, limiter_ratios
                    ):
                        if group["weight_decay"] != 0.0:
                            parameter.mul_(1.0 - lr * group["weight_decay"])
                        if limiter_ratio is None:
                            parameter.add_(update, alpha=-lr * scale_factor)
                            continue
                        if group["scale_front"] and not scale_is_identity:
                            limiter_ratio = limiter_ratio * scale_factor
                        parameter.addcmul_(update, limiter_ratio, value=-lr)

    def step_with_performance(self, performance):
        """Run one step while exposing lightweight diagnostics to the trainer."""
        self._performance = performance
        try:
            return self.step()
        finally:
            self._performance = None
            self._active_low_rank_measure = None

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        performance = getattr(self, "_performance", None)
        auto_schedule_enabled = getattr(self, "auto_schedule_kind", None) is not None
        metric_totals = {}
        measure_optimizer = (
            performance.measure_optimizer
            if performance is not None and hasattr(performance, "measure_optimizer")
            else lambda _name: torch.no_grad()
        )
        self._active_low_rank_measure = (
            measure_optimizer
            if (
                performance is not None
                and getattr(performance, "optimizer_breakdown", False)
                and hasattr(performance, "measure_optimizer")
            )
            else None
        )

        def add_metric(name: str, value: float = 1.0) -> None:
            if performance is not None:
                metric_totals[name] = metric_totals.get(name, 0) + value

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = self._auto_schedule_begin_group(group)
            scale_factor = math.sqrt(group["scale"])
            scale_is_identity = scale_factor == 1.0
            sf_parameters = [
                parameter for parameter in group["params"]
                if self._is_adamw_sf_fallback(parameter, group)
            ]
            if sf_parameters and not group["sf_train_mode"]:
                raise RuntimeError(
                    "APOLLO AdamW-SF fallback requires train() before step()."
                )
            sf_k = int(group["sf_k"])
            sf_ckp1 = 0.0
            if sf_parameters:
                sf_lr_max = max(float(lr), float(group["sf_lr_max"]))
                sf_weight = sf_lr_max ** 2
                sf_weight_sum = float(group["sf_weight_sum"]) + sf_weight
                sf_ckp1 = sf_weight / sf_weight_sum if sf_weight_sum else 0.0
                group["sf_lr_max"] = sf_lr_max
                group["sf_weight_sum"] = sf_weight_sum
            stats = None
            fast_vector_ids: set[int] = set()
            group_policy = self._fallback_policy_for_group(group)
            if (
                not auto_schedule_enabled
                and group_policy.one_dimensional == "came"
            ):
                fast_vector_parameters = [
                    parameter
                    for parameter in group["params"]
                    if parameter.ndim < 2 and parameter.grad is not None
                ]
                fast_vector_devices = {
                    parameter.device for parameter in fast_vector_parameters
                }
                if (
                    len(fast_vector_parameters) >= 2
                    and len(fast_vector_devices) == 1
                ):
                    self._step_vector_came_foreach(
                        fast_vector_parameters,
                        group,
                        add_metric,
                        measure_optimizer,
                    )
                    fast_vector_ids = {
                        id(parameter) for parameter in fast_vector_parameters
                    }
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("APOLLO does not support sparse gradients")
                if id(parameter) in fast_vector_ids:
                    continue
                add_metric("apollo_parameter_tensors")
                add_metric("apollo_gradient_elements", grad.numel())
                refresh_policy = None

                if auto_schedule_enabled and stats is None:
                    stats = self._auto_schedule_new_stats(parameter)
                if auto_schedule_enabled:
                    assert stats is not None
                    self._auto_schedule_add_norm(
                        stats, "parameter_norm_sq", parameter
                    )

                state = self.state[parameter]
                backend = self._select_backend(parameter, state, group)
                if backend == "adamw-sf":
                    add_metric("apollo_adamw_sf_fallback_parameters")
                    with measure_optimizer("apollo_adamw_sf_fallback_update"):
                        self._adamw_sf_fallback_update(
                            parameter, grad, state, group,
                            lr=float(lr), k=sf_k, ckp1=sf_ckp1,
                        )
                    continue
                came_matrix_fallback = (
                    parameter.ndim >= 2 and backend == "came"
                )
                if came_matrix_fallback:
                    add_metric("apollo_matrix_fallback_parameters")
                    with measure_optimizer("apollo_came_fallback_update"):
                        update = self._full_came_matrix_update(
                            parameter, grad, state, group,
                        )
                    projection_refreshed = False
                else:
                    self._ensure_state(parameter, state, group)
                    state["step"] = int(state.get("step", 0)) + 1
                    if parameter.ndim >= 2:
                        self._rotate_orthogonal_projection_state(
                            state, group, parameter=parameter,
                        )
                        refresh_policy = ProjectionRefreshPolicy.from_value(
                            group.get("projection_refresh")
                        )
                    refresh_due = (
                        parameter.ndim >= 2
                        and state["step"] > 1
                        and state["step"] % int(group["update_proj_gap"]) == 0
                    )
                    if performance is not None and refresh_due:
                        with performance.measure("apollo_projection_refresh"):
                            projection_refreshed = self._refresh_projection(
                                parameter, state, group,
                            )
                    else:
                        projection_refreshed = self._refresh_projection(
                            parameter, state, group,
                        )
                    if parameter.ndim >= 2:
                        assert refresh_policy is not None
                        prepare_stochastic_refresh(
                            state, refresh_policy, int(group["seed"]),
                        )
                if projection_refreshed:
                    if auto_schedule_enabled:
                        assert stats is not None
                        stats["projection_refresh"] = True
                    add_metric("apollo_projection_refreshes")

                fallback_state_update = False
                if parameter.ndim >= 2 and not came_matrix_fallback:
                    add_metric("apollo_matrix_parameters")
                    # Keep one FP32 matrix view for both the low-rank
                    # statistics and the full-rank scaled update.  The old
                    # path converted the BF16 gradient again in
                    # _apply_scaling, creating another large temporary.
                    grad_matrix = self._matrix_view(grad.float())
                    with measure_optimizer("apollo_low_rank_stats"):
                        if refresh_policy is not None and refresh_policy.mode == "smooth":
                            _, scaling = self._smooth_low_rank_update(
                                parameter, grad, state, group,
                                grad_matrix, refresh_policy,
                            )
                        else:
                            _, scaling = self._low_rank_update(
                                parameter, grad, state, group,
                                grad_matrix=grad_matrix,
                            )
                    add_metric("apollo_projection_elements", state["projection"].numel())
                    with measure_optimizer("apollo_update_apply"):
                        update = self._apply_scaling(
                            grad_matrix, scaling, self.scale_type, grad.shape,
                        )
                    if auto_schedule_enabled:
                        assert stats is not None
                        self._add_auto_schedule_low_rank_stats(stats, state)
                        self._auto_schedule_add_norm(
                            stats, "scale_norm_sq", scaling
                        )
                        stats["scale_count"] += scaling.numel()
                        stats["scale_seen"] = True
                else:
                    if not came_matrix_fallback:
                        add_metric("apollo_vector_parameters")
                    fallback_state_update = (
                        not auto_schedule_enabled
                        and state.get("backend") == "came"
                        and parameter.ndim < 2
                    )
                    if not came_matrix_fallback:
                        with measure_optimizer("apollo_fallback_update"):
                            update = self._fallback_scale(
                                parameter,
                                grad,
                                state,
                                group,
                                materialize_update=auto_schedule_enabled,
                            )

                with measure_optimizer("apollo_update_apply"):
                    with measure_optimizer("apollo_limiter"):
                        # Non-AutoSchedule fallback updates are the EMA state
                        # itself, so every post-processing operation must be
                        # represented as a scalar at parameter.apply time.
                        # AutoSchedule retains the historical materialized
                        # update path because it consumes update norms.
                        if (
                            auto_schedule_enabled
                            and not came_matrix_fallback
                            and group["scale_front"]
                            and not scale_is_identity
                        ):
                            update.mul_(scale_factor)

                        # The limiter activity is only consumed by AutoSchedule.
                        # Avoid allocating a device scalar for ordinary APOLLO,
                        # where this value is otherwise unused.
                        limiter_active = (
                            update.new_zeros(()) if auto_schedule_enabled else None
                        )
                        limiter_ratio = None
                        if group["norm_growth_limiter"] and not came_matrix_fallback:
                            current_norm = update.norm()
                            if (
                                fallback_state_update
                                and group["scale_front"]
                                and not scale_is_identity
                            ):
                                # The state-owned EMA cannot be scaled in
                                # place. Apply scale_front to its norm so the
                                # limiter observes the same value as the
                                # historical materialized-update path.
                                current_norm.mul_(scale_factor)
                            previous_norm = state.get("scaled_grad_norm")
                            if previous_norm is not None:
                                # Reuse the previous norm scalar as scratch.
                                # It is replaced by the current norm below, so
                                # no state value is lost and no extra 0-d
                                # tensor is allocated for max_norm or the
                                # limiter ratio.
                                max_norm = previous_norm.mul_(
                                    group["norm_growth_rate"]
                                )
                                limiter_ratio = (
                                    max_norm.div_(
                                        current_norm.clamp_min(group["eps"])
                                    )
                                ).clamp_max_(1.0)
                                if auto_schedule_enabled:
                                    update.mul_(limiter_ratio)
                                    limiter_active = (current_norm > max_norm).to(
                                        dtype=update.dtype
                                    )
                                torch.minimum(
                                    current_norm, max_norm, out=current_norm,
                                )
                            state["scaled_grad_norm"] = current_norm.detach()
                        if auto_schedule_enabled:
                            assert stats is not None
                            stats["limiter_active"].add_(limiter_active.float())
                            stats["limiter_count"] += 1

                        if (
                            auto_schedule_enabled
                            and not came_matrix_fallback
                            and not group["scale_front"]
                            and not scale_is_identity
                        ):
                            update.mul_(scale_factor)

                    variance_capped_scale = None
                    variance_cap = group.get("update_norm_variance_cap")
                    if variance_cap is not None:
                        if auto_schedule_enabled:
                            effective_scale = 1.0
                        elif limiter_ratio is None:
                            effective_scale = (
                                1.0
                                if came_matrix_fallback
                                else (
                                    scale_factor
                                    if fallback_state_update
                                    else (
                                        1.0
                                        if group["scale_front"]
                                        else scale_factor
                                    )
                                )
                            )
                        else:
                            effective_scale = limiter_ratio
                            if (
                                fallback_state_update
                                and not group["scale_front"]
                                and not scale_is_identity
                            ) or (
                                not fallback_state_update
                                and not came_matrix_fallback
                                and not group["scale_front"]
                                and not scale_is_identity
                            ):
                                effective_scale = effective_scale * scale_factor
                        effective_scale, _ = cap_update_norm_variance_scale(
                            update,
                            state,
                            float(variance_cap),
                            scale=effective_scale,
                        )
                        variance_capped_scale = effective_scale

                    with measure_optimizer("apollo_parameter_apply"):
                        if auto_schedule_enabled:
                            assert stats is not None
                            self._auto_schedule_add_norm(
                                stats, "update_norm_sq", update
                            )
                        if group["weight_decay"] != 0.0:
                            add_metric("apollo_weight_decay_parameters")
                            parameter.mul_(1.0 - lr * group["weight_decay"])
                        if variance_capped_scale is not None:
                            # Apply the capped effective update through a
                            # scalar multiplier; avoid a full-size temporary.
                            parameter.addcmul_(
                                update, variance_capped_scale, value=-lr,
                            )
                        elif auto_schedule_enabled:
                            # Keep the established path for AutoSchedule,
                            # which consumes the materialized scaled update.
                            parameter.add_(update, alpha=-lr)
                        elif limiter_ratio is None:
                            # No dynamic limiter ratio is needed on the first
                            # step (or when the limiter is disabled). Use the
                            # scalar alpha to avoid an elementwise scale.
                            parameter.add_(
                                update,
                                alpha=-lr * (
                                    1.0
                                    if came_matrix_fallback
                                    else (
                                        scale_factor
                                        if fallback_state_update
                                        else (
                                            1.0
                                            if group["scale_front"]
                                            else scale_factor
                                        )
                                    )
                                ),
                            )
                        else:
                            # For ordinary APOLLO, fold the dynamic scalar
                            # limiter into the destination update. This avoids
                            # materializing update * limiter_ratio before the
                            # parameter add.
                            if (
                                fallback_state_update
                                and not scale_is_identity
                            ) or (
                                not fallback_state_update
                                and not came_matrix_fallback
                                and not group["scale_front"]
                                and not scale_is_identity
                            ):
                                limiter_ratio = limiter_ratio * scale_factor
                            parameter.addcmul_(
                                update, limiter_ratio, value=-lr,
                            )
                    if refresh_policy is not None and refresh_policy.mode == "smooth":
                        self._advance_smooth_projection_refresh(
                            state, refresh_policy,
                        )

            self._auto_schedule_finish_group(group, stats)
            if sf_parameters:
                group["sf_k"] = sf_k + 1

        if performance is not None and hasattr(performance, "add_metric"):
            for name, value in metric_totals.items():
                performance.add_metric(name, value)
        return loss


class APOLLO(_APOLLOBase):
    """Channel-wise APOLLO with low-rank Adam statistics."""

    scale_type = "channel"


class APOLLOConfidence(APOLLO):
    """APOLLO with innovation-variance confidence normalization.

    This variant keeps APOLLO's state layout and full-gradient application:
    ``projection``, ``exp_avg`` and ``exp_avg_sq`` are all low-rank (for
    matrix parameters), and the resulting channel-wise scaling is applied to
    the original gradient.  Unlike ordinary APOLLO, ``exp_avg_sq`` tracks the
    innovation ``(g_lr - m_previous)^2`` rather than the squared gradient.
    """

    def __init__(
        self,
        params,
        *,
        confidence_beta: float = 0.99,
        confidence_alpha: float = 1e-3,
        **kwargs,
    ):
        if not 0.0 <= confidence_beta < 1.0:
            raise ValueError("confidence_beta must be between 0 and 1")
        if confidence_alpha < 0.0:
            raise ValueError("confidence_alpha must be non-negative")
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            group["apollo_confidence"] = True
            group["apollo_confidence_beta"] = float(confidence_beta)
            group["apollo_confidence_alpha"] = float(confidence_alpha)

    def _low_rank_update(
        self,
        parameter: torch.Tensor,
        grad: torch.Tensor,
        state: dict,
        group: dict,
        grad_matrix: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        matrix = (
            grad_matrix
            if grad_matrix is not None
            else self._matrix_view(grad.float())
        )
        low_rank_grad = self._project(matrix, state["projection"].float())
        state["apollo_confidence"] = True
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(low_rank_grad)
            state["exp_avg_sq"] = torch.zeros_like(low_rank_grad)

        beta1 = group["betas"][0]
        confidence_beta = group["apollo_confidence_beta"]
        alpha = group["apollo_confidence_alpha"]
        previous_mean = state["exp_avg"]
        residual = low_rank_grad - previous_mean
        state["exp_avg_sq"].mul_(confidence_beta).addcmul_(
            residual, residual, value=1.0 - confidence_beta,
        )
        state["exp_avg"].mul_(beta1).add_(
            low_rank_grad, alpha=1.0 - beta1,
        )

        step = int(state["step"])
        mean = state["exp_avg"].float().div(1.0 - beta1**step)
        variance = state["exp_avg_sq"].float().div(
            1.0 - confidence_beta**step,
        )
        normalized = mean.div(
            variance.addcmul(mean, mean, value=alpha).sqrt_().add_(
                group["eps"],
            )
        )

        rows, cols = matrix.shape
        if rows >= cols:
            scaling = normalized.norm(dim=1) / (
                low_rank_grad.norm(dim=1).add_(group["eps"])
            )
            scaling = scaling.reshape(rows, 1)
        else:
            scaling = normalized.norm(dim=0) / (
                low_rank_grad.norm(dim=0).add_(group["eps"])
            )
            scaling = scaling.reshape(1, cols)
        return low_rank_grad, scaling


class APOLLOADAMW(APOLLO):
    """APOLLO with an explicitly AdamW-named low-rank matrix path.

    APOLLO's matrix path already uses Adam moments in the projected space and
    applies decoupled weight decay to the original parameter.  This class
    makes that interpretation explicit; its fallback behavior follows the
    shared APOLLO fallback policy.
    """


class APOLLOLion(APOLLO):
    """APOLLO whose low-rank matrix update uses the plain Lion rule.

    The projected gradient is processed with Lion's two momentum coefficients
    and sign update.  The resulting channel norms are still applied as
    APOLLO scaling factors to the original full-rank gradient. Fallback
    behavior follows the shared APOLLO fallback policy.
    """

    def __init__(self, params, *, betas=(0.9, 0.99), **kwargs):
        super().__init__(params, betas=betas, **kwargs)

    def _low_rank_update(
        self,
        parameter: torch.Tensor,
        grad: torch.Tensor,
        state: dict,
        group: dict,
        grad_matrix: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        matrix = (
            grad_matrix
            if grad_matrix is not None
            else self._matrix_view(grad.float())
        )
        low_rank_grad = self._project(matrix, state["projection"].float())
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(low_rank_grad)

        beta1, beta2 = group["betas"]
        momentum = state["exp_avg"]
        update = momentum.mul(beta1).add(low_rank_grad, alpha=1.0 - beta1)
        lion_update = update.sign()
        momentum.mul_(beta2).add_(low_rank_grad, alpha=1.0 - beta2)

        rows, cols = matrix.shape
        if rows >= cols:
            scaling = lion_update.norm(dim=1) / (
                low_rank_grad.norm(dim=1).add_(group["eps"])
            )
            scaling = scaling.reshape(rows, 1)
        else:
            scaling = lion_update.norm(dim=0) / (
                low_rank_grad.norm(dim=0).add_(group["eps"])
            )
            scaling = scaling.reshape(1, cols)
        return low_rank_grad, scaling


class RotAPOLLO(APOLLO):
    """APOLLO with a slowly rotating, gradient-aware projection basis.

    The basis is kept orthonormal and is updated every ``rotation_frequency``
    steps.  Most of the tangent direction follows the current gradient
    covariance, while ``exploration_ratio`` preserves motion into unexplored
    directions.  Low-rank moments are transported through the overlap between
    the old and new bases before the next update.
    """

    def __init__(
        self,
        params,
        *,
        rotation_frequency: int = 10,
        rotation_rate: float = 0.02,
        exploration_ratio: float = 0.2,
        **kwargs,
    ):
        if rotation_frequency <= 0:
            raise ValueError("rotation_frequency must be positive")
        if rotation_rate < 0.0:
            raise ValueError("rotation_rate must be non-negative")
        if not 0.0 <= exploration_ratio <= 1.0:
            raise ValueError("exploration_ratio must be in [0, 1]")
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            group["rotation_frequency"] = int(rotation_frequency)
            group["rotation_rate"] = float(rotation_rate)
            group["exploration_ratio"] = float(exploration_ratio)

    @staticmethod
    def _make_projection(parameter, rank, seed):
        matrix = _APOLLOBase._matrix_view(parameter)
        rows, cols = matrix.shape
        generator = torch.Generator(device=parameter.device).manual_seed(int(seed))
        if rows >= cols:
            raw = torch.randn(
                (cols, rank), generator=generator, device=parameter.device,
                dtype=torch.float32,
            )
            return torch.linalg.qr(raw, mode="reduced").Q
        raw = torch.randn(
            (rows, rank), generator=generator, device=parameter.device,
            dtype=torch.float32,
        )
        return torch.linalg.qr(raw, mode="reduced").Q.t().contiguous()

    @staticmethod
    def _transport_state(state, old_projection, new_projection, rows_ge_cols):
        if "exp_avg" not in state:
            return
        if rows_ge_cols:
            overlap = old_projection.t().mm(new_projection)
            state["exp_avg"] = state["exp_avg"].mm(overlap)
            if "exp_avg_sq" in state:
                state["exp_avg_sq"] = state["exp_avg_sq"].mm(overlap.square())
        else:
            overlap = new_projection.mm(old_projection.t())
            state["exp_avg"] = overlap.mm(state["exp_avg"])
            if "exp_avg_sq" in state:
                state["exp_avg_sq"] = overlap.square().mm(state["exp_avg_sq"])

    def _refresh_projection(self, parameter, state, group):
        if parameter.ndim < 2 or "projection" not in state:
            return False
        step = int(state["step"])
        if step <= 1 or step % group["rotation_frequency"] != 0:
            return False

        matrix = self._matrix_view(parameter.grad.float())
        old_projection = state["projection"]
        rows, cols = matrix.shape
        rank = int(state["projection_rank"])
        if rows >= cols:
            projected = matrix.mm(old_projection)
            gradient_direction = matrix.t().mm(projected)
            gradient_direction.sub_(
                old_projection.mm(old_projection.t().mm(gradient_direction))
            )
            noise = torch.randn_like(old_projection)
            noise.sub_(old_projection.mm(old_projection.t().mm(noise)))
        else:
            projected = old_projection.mm(matrix)
            gradient_direction = projected.mm(matrix.t())
            gradient_direction.sub_(
                gradient_direction.mm(old_projection.t()).mm(old_projection)
            )
            noise = torch.randn_like(old_projection)
            noise.sub_(noise.mm(old_projection.t()).mm(old_projection))

        def normalized(tensor):
            return tensor / tensor.norm().clamp_min(group["eps"])

        direction = (
            (1.0 - group["exploration_ratio"]) * normalized(gradient_direction)
            + group["exploration_ratio"] * normalized(noise)
        )
        direction = normalized(direction)
        candidate = old_projection + group["rotation_rate"] * math.sqrt(rank) * direction
        if rows >= cols:
            new_projection = torch.linalg.qr(candidate, mode="reduced").Q
        else:
            new_projection = torch.linalg.qr(candidate.t(), mode="reduced").Q.t().contiguous()
        self._transport_state(state, old_projection, new_projection, rows >= cols)
        state["projection"] = new_projection
        state["projection_seed"] = _stable_seed(state["projection_seed"])
        return True


class DualRotAPOLLO(APOLLO):
    """APOLLO with two independent projections and adaptive exploration.

    Each matrix parameter owns two low-rank Adam states.  Their temporal
    roughness controls a soft mixture of the two APOLLO scaling factors.  The
    more stable branch is periodically rotated into a nearby direction, while
    the other branch remains an anchor. Fallback behavior follows the shared
    APOLLO fallback policy.
    """

    def __init__(
        self,
        params,
        *,
        rotation_frequency: int = 10,
        rotation_rate: float = 0.02,
        exploration_ratio: float = 0.2,
        roughness_beta: float = 0.95,
        branch_temperature: float = 5.0,
        **kwargs,
    ):
        if rotation_frequency <= 0:
            raise ValueError("rotation_frequency must be positive")
        if rotation_rate < 0.0:
            raise ValueError("rotation_rate must be non-negative")
        if not 0.0 <= exploration_ratio <= 1.0:
            raise ValueError("exploration_ratio must be in [0, 1]")
        if not 0.0 <= roughness_beta < 1.0:
            raise ValueError("roughness_beta must be in [0, 1)")
        if branch_temperature < 0.0:
            raise ValueError("branch_temperature must be non-negative")
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            group["rotation_frequency"] = int(rotation_frequency)
            group["rotation_rate"] = float(rotation_rate)
            group["exploration_ratio"] = float(exploration_ratio)
            group["roughness_beta"] = float(roughness_beta)
            group["branch_temperature"] = float(branch_temperature)

    @staticmethod
    def _new_branch(parameter, rank, seed):
        return {
            "step": 0,
            "projection": RotAPOLLO._make_projection(parameter, rank, seed),
            "projection_rank": rank,
            "exp_avg": None,
            "exp_avg_sq": None,
            "previous_gradient": None,
            "roughness": torch.zeros((), device=parameter.device, dtype=torch.float32),
        }

    @staticmethod
    def _branch_update(matrix, branch, group):
        projection = branch["projection"]
        if matrix.shape[0] >= matrix.shape[1]:
            low_rank_gradient = matrix.mm(projection)
        else:
            low_rank_gradient = projection.mm(matrix)
        if branch["exp_avg"] is None:
            branch["exp_avg"] = torch.zeros_like(low_rank_gradient)
            branch["exp_avg_sq"] = torch.zeros_like(low_rank_gradient)

        beta1, beta2 = group["betas"]
        branch["exp_avg"].mul_(beta1).add_(low_rank_gradient, alpha=1.0 - beta1)
        branch["exp_avg_sq"].mul_(beta2).addcmul_(
            low_rank_gradient, low_rank_gradient, value=1.0 - beta2,
        )
        bias1 = 1.0 - beta1 ** branch["step"]
        bias2 = 1.0 - beta2 ** branch["step"]
        normalized = (
            branch["exp_avg"] / bias1
            / (branch["exp_avg_sq"] / bias2).sqrt().add_(group["eps"])
        )

        if branch["previous_gradient"] is not None:
            current = low_rank_gradient.reshape(-1)
            previous = branch["previous_gradient"].reshape(-1)
            roughness = 1.0 - torch.nn.functional.cosine_similarity(
                current.unsqueeze(0), previous.unsqueeze(0), dim=1,
                eps=group["eps"],
            ).squeeze(0)
            beta = group["roughness_beta"]
            branch["roughness"].mul_(beta).add_(roughness, alpha=1.0 - beta)
        branch["previous_gradient"] = low_rank_gradient.detach().clone()

        rows, cols = matrix.shape
        if rows >= cols:
            scaling = normalized.norm(dim=1) / (
                low_rank_gradient.norm(dim=1).add_(group["eps"])
            )
            return scaling.reshape(rows, 1)
        scaling = normalized.norm(dim=0) / (
            low_rank_gradient.norm(dim=0).add_(group["eps"])
        )
        return scaling.reshape(1, cols)

    @staticmethod
    def _rotate_branch(matrix, branch, group):
        projection = branch["projection"]
        rows, cols = matrix.shape
        rank = branch["projection_rank"]
        if rows >= cols:
            projected = matrix.mm(projection)
            gradient_direction = matrix.t().mm(projected)
            gradient_direction.sub_(
                projection.mm(projection.t().mm(gradient_direction))
            )
            noise = torch.randn_like(projection)
            noise.sub_(projection.mm(projection.t().mm(noise)))
        else:
            projected = projection.mm(matrix)
            gradient_direction = projected.mm(matrix.t())
            gradient_direction.sub_(
                gradient_direction.mm(projection.t()).mm(projection)
            )
            noise = torch.randn_like(projection)
            noise.sub_(noise.mm(projection.t()).mm(projection))

        def normalized(tensor):
            return tensor / tensor.norm().clamp_min(group["eps"])

        direction = (
            (1.0 - group["exploration_ratio"]) * normalized(gradient_direction)
            + group["exploration_ratio"] * normalized(noise)
        )
        direction = normalized(direction)
        candidate = projection + group["rotation_rate"] * math.sqrt(rank) * direction
        if rows >= cols:
            new_projection = torch.linalg.qr(candidate, mode="reduced").Q
        else:
            new_projection = torch.linalg.qr(candidate.t(), mode="reduced").Q.t().contiguous()
        RotAPOLLO._transport_state(
            branch, projection, new_projection, rows >= cols,
        )
        branch["projection"] = new_projection

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad.float()
                if grad.is_sparse:
                    raise RuntimeError("DualRotAPOLLO does not support sparse gradients")
                state = self.state[parameter]
                state["step"] = int(state.get("step", 0)) + 1
                if parameter.ndim < 2:
                    # This legacy/experimental path still applies scale and
                    # limiter in-place below, so it must not receive the
                    # state-owned fallback EMA directly.
                    update = self._fallback_scale(
                        parameter, grad, state, group, materialize_update=True,
                    )
                else:
                    matrix = self._matrix_view(grad)
                    rows, cols = matrix.shape
                    rank = min(int(group["rank"]), min(rows, cols))
                    if "branches" not in state:
                        seed = int(group["seed"])
                        state["branches"] = [
                            self._new_branch(parameter, rank, seed),
                            self._new_branch(parameter, rank, _stable_seed(seed)),
                        ]
                    branches = state["branches"]
                    for branch in branches:
                        branch["step"] += 1

                    if state["step"] > 1 and state["step"] % group["rotation_frequency"] == 0:
                        stable_index = int(
                            (branches[1]["roughness"] < branches[0]["roughness"]).item()
                        )
                        self._rotate_branch(matrix, branches[stable_index], group)

                    scalings = [self._branch_update(matrix, branch, group) for branch in branches]
                    roughness = torch.stack([branch["roughness"] for branch in branches])
                    weights = torch.softmax(
                        -group["branch_temperature"] * roughness, dim=0,
                    )
                    scaling = weights[0] * scalings[0] + weights[1] * scalings[1]
                    update = (matrix * scaling).reshape_as(parameter)

                if group["scale_front"]:
                    update.mul_(math.sqrt(group["scale"]))
                if group["norm_growth_limiter"]:
                    current_norm = update.norm()
                    previous_norm = state.get("scaled_grad_norm")
                    if previous_norm is not None:
                        max_norm = previous_norm.mul_(group["norm_growth_rate"])
                        update.mul_(
                            max_norm.div_(
                                current_norm.clamp_min(group["eps"])
                            ).clamp_max_(1.0)
                        )
                        torch.minimum(current_norm, max_norm, out=current_norm)
                    state["scaled_grad_norm"] = current_norm.detach()
                if not group["scale_front"]:
                    update.mul_(math.sqrt(group["scale"]))
                if group["weight_decay"] != 0.0:
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
        return loss


class APOLLOMini(_APOLLOBase):
    """Rank-one tensor-wise APOLLO-Mini."""

    scale_type = "tensor"

    def __init__(self, params, *, rank: int = 1, **kwargs):
        super().__init__(params, rank=1, **kwargs)


class APOLLOCAME(_APOLLOBase):
    """CAME confidence scaling computed in APOLLO's low-rank space.

    The full-rank gradient is still used for the parameter update.  CAME is
    used only to produce channel-wise scaling factors from the projected
    gradient, so the optimizer does not allocate a full-size first moment.
    """

    scale_type = "channel"

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        rank: int = 8,
        scale: float = 1.0,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        eps: tuple[float, float] = (1e-30, 1e-16),
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
        update_proj_gap: int = 200,
        seed: int = 0,
        scale_front: bool = False,
        norm_growth_limiter: bool = False,
        norm_growth_rate: float = 1.01,
        projection_refresh_state: str = "reset",
        projection_refresh=None,
        orthogonal_refresh=None,
        update_norm_variance_cap: float | None = None,
        fallback: APOLLOFallbackPolicy | Mapping[str, object] | str | None = None,
        came_backend: str = "torch",
    ):
        if len(betas) != 3 or not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError("betas must contain three values in [0, 1)")
        if len(eps) != 2 or any(value <= 0.0 for value in eps):
            raise ValueError("eps must contain two positive values")
        if clip_threshold <= 0.0:
            raise ValueError("clip_threshold must be positive")
        if came_backend not in {"auto", "torch", "triton"}:
            raise ValueError("came_backend must be 'auto', 'torch', or 'triton'")
        super().__init__(
            params,
            lr=lr,
            rank=rank,
            scale=scale,
            betas=betas[:2],
            eps=eps[1],
            weight_decay=weight_decay,
            update_proj_gap=update_proj_gap,
            seed=seed,
            scale_front=scale_front,
            norm_growth_limiter=norm_growth_limiter,
            norm_growth_rate=norm_growth_rate,
            projection_refresh_state=projection_refresh_state,
            projection_refresh=projection_refresh,
            orthogonal_refresh=orthogonal_refresh,
            update_norm_variance_cap=update_norm_variance_cap,
            fallback=fallback,
        )
        for group in self.param_groups:
            group["came_betas"] = tuple(betas)
            group["came_eps"] = tuple(eps)
            group["clip_threshold"] = float(clip_threshold)
        self.came_backend = came_backend

    @staticmethod
    def _approx_sq_grad(
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
    ) -> torch.Tensor:
        row_factor = (
            exp_avg_sq_row
            / exp_avg_sq_row.mean(dim=-1, keepdim=True).clamp_min(1e-30)
        ).rsqrt_().unsqueeze(-1)
        col_factor = exp_avg_sq_col.clamp_min(1e-30).rsqrt().unsqueeze(-2)
        return row_factor * col_factor

    def _low_rank_update(
        self,
        parameter: torch.Tensor,
        grad: torch.Tensor,
        state: dict,
        group: dict,
        grad_matrix: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        matrix = (
            grad_matrix
            if grad_matrix is not None
            else self._matrix_view(grad.float())
        )
        projection = state["projection"].float()
        component_measure = getattr(self, "_active_low_rank_measure", None)

        def measure_component(name: str):
            if component_measure is None:
                return nullcontext()
            return component_measure(name)

        gradient = state.get("came_low_rank_grad")
        expected_shape = (
            (matrix.shape[0], projection.shape[1])
            if matrix.shape[0] >= matrix.shape[1]
            else (projection.shape[0], matrix.shape[1])
        )
        if (
            gradient is None
            or gradient.shape != expected_shape
            or gradient.dtype != matrix.dtype
            or gradient.device != matrix.device
        ):
            gradient = torch.empty(
                expected_shape, dtype=matrix.dtype, device=matrix.device
            )
            state["came_low_rank_grad"] = gradient
        with measure_component("apollo_low_rank_project"):
            self._project_into(matrix, projection, gradient)
        beta1, beta2, beta3 = group["came_betas"]
        eps_square, eps_instability = group["came_eps"]

        if "exp_avg" not in state:
            rows, cols = gradient.shape
            state["exp_avg"] = torch.zeros_like(gradient)
            state["exp_avg_sq_row"] = torch.zeros(
                rows, dtype=gradient.dtype, device=gradient.device
            )
            state["exp_avg_sq_col"] = torch.zeros(
                cols, dtype=gradient.dtype, device=gradient.device
            )
            state["exp_avg_res_row"] = torch.zeros(
                rows, dtype=gradient.dtype, device=gradient.device
            )
            state["exp_avg_res_col"] = torch.zeros(
                cols, dtype=gradient.dtype, device=gradient.device
            )

        # APOLLO-CAME materializes several full low-rank tensors during one
        # update (the normalized update, residual, and final CAME update).
        # Keep one reusable workspace in the optimizer state so the CUDA
        # allocator and the allocator synchronization do not have to handle
        # those same-sized tensors on every step.  This is initialized lazily
        # so checkpoints created before this optimization remain loadable.
        came_work = state.get("came_work")
        if (
            came_work is None
            or came_work.shape != gradient.shape
            or came_work.dtype != gradient.dtype
            or came_work.device != gradient.device
        ):
            came_work = torch.empty_like(gradient)
            state["came_work"] = came_work

        with measure_component("apollo_came_moments"):
            second_moment = gradient.square().add_(eps_square)
            state["exp_avg_sq_row"].mul_(beta2).add_(
                second_moment.mean(dim=-1), alpha=1.0 - beta2
            )
            state["exp_avg_sq_col"].mul_(beta2).add_(
                second_moment.mean(dim=-2), alpha=1.0 - beta2
            )
        # Reuse the workspace for the normalized update.  `gradient` must stay
        # intact because it is also used as the denominator of the CAME
        # scaling factor and is returned to the caller.
        with measure_component("apollo_came_adaptive_update"):
            fused = fused_came_adaptive_update(
                gradient,
                state["exp_avg_sq_row"],
                state["exp_avg_sq_col"],
                state["exp_avg"],
                came_work,
                beta1=beta1,
                clip_threshold=group["clip_threshold"],
                eps=eps_square,
                backend=self.came_backend,
            )
            if fused:
                update = came_work
            else:
                update = torch.mul(
                    self._approx_sq_grad(
                        state["exp_avg_sq_row"], state["exp_avg_sq_col"]
                    ),
                    gradient,
                    out=came_work,
                )
                update.div_(
                    (self._rms(update) / group["clip_threshold"]).clamp_min(1.0)
                )

                state["exp_avg"].mul_(beta1).add_(
                    update, alpha=1.0 - beta1
                )
        # The previous `update` is no longer needed after exp_avg is updated;
        # reuse the same buffer for the residual and then for came_update.
        with measure_component("apollo_came_residual"):
            residual = came_work.sub_(state["exp_avg"]).square_().add_(
                eps_instability
            )
            state["exp_avg_res_row"].mul_(beta3).add_(
                residual.mean(dim=-1), alpha=1.0 - beta3
            )
            state["exp_avg_res_col"].mul_(beta3).add_(
                residual.mean(dim=-2), alpha=1.0 - beta3
            )

        with measure_component("apollo_came_scaling"):
            came_update = torch.mul(
                self._approx_sq_grad(
                    state["exp_avg_res_row"], state["exp_avg_res_col"]
                ),
                state["exp_avg"],
                out=came_work,
            )

            rows, cols = gradient.shape
            if rows >= cols:
                scaling = came_update.norm(dim=1) / (
                    gradient.norm(dim=1).add_(group["eps"])
                )
                scaling = scaling.reshape(rows, 1)
            else:
                scaling = came_update.norm(dim=0) / (
                    gradient.norm(dim=0).add_(group["eps"])
                )
                scaling = scaling.reshape(1, cols)
        return gradient, scaling

    def _add_auto_schedule_low_rank_stats(
        self,
        stats: dict,
        state: dict,
    ) -> None:
        """Expose CAME's low-rank moment/residual confidence to the controller."""
        exp_avg = state.get("exp_avg")
        residual_row = state.get("exp_avg_res_row")
        residual_col = state.get("exp_avg_res_col")
        if exp_avg is None or residual_row is None or residual_col is None:
            return

        # Aggregate low-rank states by element count rather than by tensor
        # count.  Otherwise a large matrix and a small matrix contribute the
        # same confidence weight to one parameter group.  The row/column
        # residual factors provide an inexpensive estimate of mean residual
        # energy without materializing the low-rank residual matrix.
        element_count = exp_avg.numel()
        stats["moment_norm_sq"].add_(exp_avg.float().square().sum())
        residual_energy = 0.5 * (
            residual_row.float().mean() + residual_col.float().mean()
        )
        stats["noise_norm_sq"].add_(residual_energy * element_count)


class APOLLOAutoSchedule(APOLLO):
    """APOLLO with a bounded per-parameter-group LR controller."""

    def __init__(
        self,
        params,
        *,
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
        auto_schedule_fast_beta=0.9,
        auto_schedule_slow_beta=0.999,
        auto_schedule_gain=0.25,
        auto_schedule_controller_rate=0.05,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._enable_auto_schedule(
            kind="apollo",
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
            fast_beta=auto_schedule_fast_beta,
            slow_beta=auto_schedule_slow_beta,
            gain=auto_schedule_gain,
            controller_rate=auto_schedule_controller_rate,
        )


class APOLLOADAMWAutoSchedule(APOLLOAutoSchedule):
    """AutoSchedule variant of :class:`APOLLOADAMW`."""


class APOLLOCAMEAutoSchedule(APOLLOCAME):
    """APOLLO-CAME with a bounded per-parameter-group LR controller."""

    def __init__(
        self,
        params,
        *,
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
        auto_schedule_fast_beta=0.9,
        auto_schedule_slow_beta=0.999,
        auto_schedule_gain=0.25,
        auto_schedule_controller_rate=0.05,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._enable_auto_schedule(
            kind="apollo-came",
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
            fast_beta=auto_schedule_fast_beta,
            slow_beta=auto_schedule_slow_beta,
            gain=auto_schedule_gain,
            controller_rate=auto_schedule_controller_rate,
        )
