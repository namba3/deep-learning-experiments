"""APOLLO updates with full-rank Schedule-Free state storage variants.

The APOLLO update path is shared across all variants.  Only the persistent
Schedule-Free state representation changes:

``bf16_z``
    Full-rank ``z`` in parameter dtype (reference).
``blockwise_int8_z`` / ``blockwise_int4_z``
    Full-rank ``z`` with symmetric blockwise quantization.
``blockwise_int8_delta`` / ``blockwise_int4_delta``
    Full-rank ``sf_delta = z - y`` with symmetric blockwise quantization.
``low_rank_delta``
    ``sf_delta`` in a separate orthonormal low-rank basis.  The decoded
    full-rank delta is still materialized transiently for the Schedule-Free
    interpolation.

This is intentionally separate from LRSF: the Schedule-Free state remains
full-rank, while only the APOLLO update statistics use a low-rank projection.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch

from .apollo import APOLLO, APOLLOFallbackPolicy


_STORAGE_MODES = {
    "bf16_z",
    "blockwise_int8_z",
    "blockwise_int4_z",
    "blockwise_int8_delta",
    "blockwise_int4_delta",
    "low_rank_delta",
}


def _quantized_mode(mode: str) -> bool:
    return mode != "bf16_z"


def _delta_mode(mode: str) -> bool:
    return mode.endswith("_delta") or mode == "low_rank_delta"


def _low_rank_delta_mode(mode: str) -> bool:
    return mode == "low_rank_delta"


class APOLLOScheduleFree(APOLLO):
    """APOLLO low-rank updates with full-rank Schedule-Free state.

    The implementation keeps ``y`` as the live parameter.  During training,
    the hidden state is represented by either ``z`` or ``sf_delta`` and the
    Schedule-Free interpolation is applied before the APOLLO update.  The
    quantized modes are reference implementations for validation; they use
    explicit dequantization buffers and are not yet the optimized CUDA path.
    """

    def __init__(
        self,
        params,
        *,
        lr=1e-3,
        rank=8,
        sf_beta1=0.9,
        warmup_steps=0,
        sf_r=0.0,
        sf_weight_lr_power=2.0,
        sf_state_storage="bf16_z",
        sf_quant_block_size=256,
        sf_quant_scale_mode="max_abs",
        sf_delta_refresh="none",
        sf_delta_refresh_window=4,
        seed=0,
        scale=1.0,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        update_proj_gap=200,
        scale_front=False,
        norm_growth_limiter=False,
        norm_growth_rate=1.01,
        projection_refresh_state="reset",
        projection_refresh=None,
        orthogonal_refresh=None,
        update_norm_variance_cap=None,
        fallback=None,
    ):
        if sf_state_storage not in _STORAGE_MODES:
            raise ValueError(
                "sf_state_storage must be one of "
                + ", ".join(sorted(_STORAGE_MODES))
            )
        if sf_quant_block_size <= 0:
            raise ValueError("sf_quant_block_size must be positive")
        if sf_quant_scale_mode != "max_abs":
            raise ValueError("only sf_quant_scale_mode='max_abs' is supported")
        if sf_delta_refresh not in {"none", "commit_z", "blend"}:
            raise ValueError(
                "sf_delta_refresh must be 'none', 'commit_z', or 'blend'"
            )
        if sf_delta_refresh_window <= 0:
            raise ValueError("sf_delta_refresh_window must be positive")
        if not 0.0 < sf_beta1 < 1.0:
            raise ValueError("sf_beta1 must be between 0 and 1")
        if warmup_steps < 0 or sf_r < 0.0 or sf_weight_lr_power < 0.0:
            raise ValueError("invalid Schedule-Free weighting values")
        if fallback is None:
            fallback = APOLLOFallbackPolicy(
                one_dimensional="came", small_matrix="apollo",
            )
        super().__init__(
            params,
            lr=lr,
            rank=rank,
            scale=scale,
            betas=betas,
            eps=eps,
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
            group.update(
                sf_beta1=float(sf_beta1),
                sf_warmup_steps=int(warmup_steps),
                sf_r=float(sf_r),
                sf_weight_lr_power=float(sf_weight_lr_power),
                sf_state_storage=str(sf_state_storage),
                sf_quant_block_size=int(sf_quant_block_size),
                sf_quant_scale_mode=str(sf_quant_scale_mode),
                sf_delta_refresh=str(sf_delta_refresh),
                sf_delta_refresh_window=int(sf_delta_refresh_window),
                sf_k=0,
                sf_lr_max=0.0,
                sf_weight_sum=0.0,
                sf_train_mode=True,
                sf_delta_commit_count=0,
                sf_delta_commit_norm_sum=0.0,
                sf_delta_commit_norm_max=0.0,
                sf_delta_blend_count=0,
                sf_delta_blend_step_count=0,
                sf_delta_blend_norm_sum=0.0,
            )

    @staticmethod
    def _quantized_names(mode: str) -> tuple[str, str, str]:
        prefix = "sf_z" if not _delta_mode(mode) else "sf_delta"
        return f"{prefix}_q", f"{prefix}_scale", f"{prefix}_numel"

    @staticmethod
    def _encode(value: torch.Tensor, *, mode: str, block_size: int) -> tuple[
        torch.Tensor, torch.Tensor
    ]:
        bits = 8 if "int8" in mode else 4
        qmax = 2 ** (bits - 1) - 1
        flat = value.detach().float().reshape(-1)
        block_count = (flat.numel() + block_size - 1) // block_size
        padded = flat.new_zeros(block_count * block_size)
        padded[: flat.numel()].copy_(flat)
        blocks = padded.reshape(block_count, block_size)
        threshold = blocks.abs().amax(dim=1)
        scale = (threshold / qmax).clamp_min(torch.finfo(torch.float32).tiny)
        quantized = torch.round(blocks / scale[:, None]).clamp(-qmax, qmax)
        if bits == 8:
            return quantized.to(torch.int8).reshape(-1)[: flat.numel()], scale

        unsigned = (quantized.to(torch.int8) + 8).to(torch.uint8).reshape(-1)
        if unsigned.numel() % 2:
            unsigned = torch.cat((unsigned, unsigned.new_zeros(1)))
        packed = unsigned[0::2] | (unsigned[1::2] << 4)
        return packed[: (flat.numel() + 1) // 2], scale

    @staticmethod
    def _decode(
        state: dict,
        *,
        mode: str,
        block_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        q_name, scale_name, numel_name = APOLLOScheduleFree._quantized_names(mode)
        packed = state[q_name].to(device=device)
        scale = state[scale_name].to(device=device, dtype=torch.float32)
        bits = 8 if "int8" in mode else 4
        if bits == 8:
            values = packed.to(torch.float32)
        else:
            low = (packed & 0x0F).to(torch.float32) - 8.0
            high = (packed >> 4).to(torch.float32) - 8.0
            values = torch.empty(
                packed.numel() * 2, device=device, dtype=torch.float32,
            )
            values[0::2] = low
            values[1::2] = high
        expanded_scale = scale.repeat_interleave(block_size)
        return (values * expanded_scale[: values.numel()])[: state[numel_name]]

    @classmethod
    def _ensure_storage(
        cls, parameter: torch.Tensor, state: dict, group: dict,
    ) -> None:
        mode = group["sf_state_storage"]
        existing = state.get("sf_state_storage")
        if existing is not None and existing != mode:
            raise RuntimeError(
                f"Schedule-Free state storage mismatch: {existing!r} != {mode!r}"
            )
        state["sf_state_storage"] = mode
        if mode == "bf16_z":
            if "sf_z" not in state:
                state["sf_z"] = parameter.detach().clone(
                    memory_format=torch.preserve_format,
                )
            elif state["sf_z"].dtype != parameter.dtype:
                state["sf_z"] = state["sf_z"].to(dtype=parameter.dtype)
            return

        if _low_rank_delta_mode(mode):
            # The APOLLO projection is created by _ensure_state in step().
            # Initialization of the latent delta is deferred until then.
            return

        q_name, scale_name, numel_name = cls._quantized_names(mode)
        if q_name not in state:
            initial = parameter.detach() if not _delta_mode(mode) else torch.zeros_like(parameter)
            q_value, scale = cls._encode(
                initial,
                mode=mode,
                block_size=group["sf_quant_block_size"],
            )
            state[q_name] = q_value
            state[scale_name] = scale
            state[numel_name] = int(parameter.numel())

    @classmethod
    def _decode_state(
        cls, parameter: torch.Tensor, state: dict, group: dict,
    ) -> torch.Tensor:
        mode = group["sf_state_storage"]
        if mode == "bf16_z":
            return state["sf_z"].float() - parameter.float()
        if _low_rank_delta_mode(mode):
            latent = state.get("sf_delta_latent")
            projection = state.get("sf_delta_projection")
            if latent is None or projection is None:
                full = state.get("sf_delta_full")
                return (
                    full.to(device=parameter.device, dtype=torch.float32)
                    if full is not None
                    else torch.zeros_like(parameter, dtype=torch.float32)
                )
            matrix = cls._matrix_view(parameter)
            projection = projection.float()
            if matrix.shape[0] >= matrix.shape[1]:
                decoded = latent.float().matmul(projection.transpose(0, 1))
            else:
                decoded = projection.transpose(0, 1).matmul(latent.float())
            return decoded.reshape(parameter.shape)
        if _delta_mode(mode):
            return cls._decode(
                state,
                mode=mode,
                block_size=group["sf_quant_block_size"],
                device=parameter.device,
            ).reshape(parameter.shape)
        z = cls._decode(
            state,
            mode=mode,
            block_size=group["sf_quant_block_size"],
            device=parameter.device,
        )
        return z.reshape(-1).reshape(parameter.shape) - parameter.float()

    @classmethod
    def _decode_z(
        cls, parameter: torch.Tensor, state: dict, group: dict,
    ) -> torch.Tensor:
        """Decode the absolute hidden parameter for train/eval restoration."""
        mode = group["sf_state_storage"]
        if mode == "bf16_z":
            return state["sf_z"].float()
        if _delta_mode(mode):
            return parameter.float() + cls._decode_state(parameter, state, group)
        return cls._decode(
            state,
            mode=mode,
            block_size=group["sf_quant_block_size"],
            device=parameter.device,
        ).reshape(parameter.shape)

    @classmethod
    def _store_state(
        cls,
        parameter: torch.Tensor,
        state: dict,
        group: dict,
        *,
        z: torch.Tensor,
        delta: torch.Tensor,
    ) -> None:
        mode = group["sf_state_storage"]
        if mode == "bf16_z":
            state["sf_z"].copy_(z.to(dtype=parameter.dtype))
            return
        if _low_rank_delta_mode(mode):
            projection = state.get("sf_delta_projection")
            latent = state.get("sf_delta_latent")
            if projection is None or latent is None:
                state["sf_delta_full"] = delta.detach().clone()
                return
            matrix = cls._matrix_view(delta.float())
            projection = projection.float()
            if matrix.shape[0] >= matrix.shape[1]:
                projected = matrix.matmul(projection)
            else:
                projected = projection.matmul(matrix)
            latent.copy_(projected)
            state.pop("sf_delta_full", None)
            return
        value = delta if _delta_mode(mode) else z
        q_value, scale = cls._encode(
            value,
            mode=mode,
            block_size=group["sf_quant_block_size"],
        )
        q_name, scale_name, numel_name = cls._quantized_names(mode)
        state[q_name] = q_value
        state[scale_name] = scale
        state[numel_name] = int(parameter.numel())

    @classmethod
    def _ensure_low_rank_delta_storage(
        cls, parameter: torch.Tensor, state: dict, group: dict,
    ) -> None:
        if not _low_rank_delta_mode(group["sf_state_storage"]):
            return
        matrix = cls._matrix_view(parameter)
        rank = min(int(group["rank"]), min(matrix.shape))
        if rank <= 0:
            return
        projection_shape = (
            (matrix.shape[1], rank)
            if matrix.shape[0] >= matrix.shape[1]
            else (rank, matrix.shape[0])
        )
        projection = state.get("sf_delta_projection")
        if projection is None or tuple(projection.shape) != projection_shape:
            old_latent = state.get("sf_delta_latent")
            old_projection = state.get("sf_delta_projection")
            if old_projection is None:
                # Checkpoints created by the first prototype used the APOLLO
                # Gaussian projection for delta storage.  Decode it once so
                # that such a checkpoint can be migrated safely.
                old_projection = state.get("projection")
            if old_latent is None:
                old_latent = state.get("sf_delta_full")
            if old_latent is not None and old_projection is not None:
                old_matrix = cls._decode_low_rank_delta(
                    matrix.shape, old_latent, old_projection,
                )
            elif state.get("sf_delta_full") is not None:
                old_matrix = state["sf_delta_full"].float().reshape(matrix.shape)
            else:
                old_matrix = None
            seed = int(group.get("seed", 0)) + 104729
            state["sf_delta_projection"] = cls._make_delta_projection(
                parameter, rank, seed,
            )
            projection = state["sf_delta_projection"]
            state.pop("sf_delta_full", None)
            state.pop("sf_delta_latent", None)
            if old_matrix is not None:
                state["sf_delta_latent"] = cls._project_low_rank_delta(
                    old_matrix, projection,
                )
        shape = (
            (matrix.shape[0], rank)
            if matrix.shape[0] >= matrix.shape[1]
            else (rank, matrix.shape[1])
        )
        latent = state.get("sf_delta_latent")
        if latent is None or tuple(latent.shape) != shape:
            state["sf_delta_latent"] = torch.zeros(
                shape, device=parameter.device, dtype=torch.float32,
            )

    @staticmethod
    def _decode_low_rank_delta(
        matrix_shape: torch.Size | tuple[int, int],
        latent: torch.Tensor,
        projection: torch.Tensor,
    ) -> torch.Tensor:
        rows, cols = matrix_shape
        if rows >= cols:
            return latent.float().matmul(projection.float().transpose(0, 1))
        return projection.float().transpose(0, 1).matmul(latent.float())

    @staticmethod
    def _project_low_rank_delta(
        matrix: torch.Tensor, projection: torch.Tensor,
    ) -> torch.Tensor:
        if matrix.shape[0] >= matrix.shape[1]:
            return matrix.float().matmul(projection.float())
        return projection.float().matmul(matrix.float())

    @classmethod
    def _make_delta_projection(
        cls, parameter: torch.Tensor, rank: int, seed: int,
    ) -> torch.Tensor:
        """Create an orthonormal basis for the low-rank delta storage.

        APOLLO's Gaussian projection is intentionally not orthonormal and is
        suitable for APOLLO's channel scaling, but not for repeated
        encode/decode of a persistent delta.  This basis makes the delta
        reconstruction a non-expansive orthogonal projection.
        """
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

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            if group["sf_train_mode"]:
                continue
            beta = group["sf_beta1"]
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if not state or "sf_state_storage" not in state:
                    continue
                if _delta_mode(group["sf_state_storage"]):
                    delta = self._decode_state(parameter, state, group)
                    parameter.add_(delta.to(parameter.dtype), alpha=1.0 / beta - 1.0)
                else:
                    z = self._decode_z(parameter, state, group)
                    parameter.add_(z - parameter.float(), alpha=1.0 - beta)
            group["sf_train_mode"] = True
        return self

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            if not group["sf_train_mode"]:
                continue
            beta = group["sf_beta1"]
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if not state or "sf_state_storage" not in state:
                    continue
                delta = self._decode_state(parameter, state, group)
                parameter.add_(delta.to(parameter.dtype), alpha=1.0 - 1.0 / beta)
            group["sf_train_mode"] = False
        return self

    @staticmethod
    def _schedule(group: dict) -> tuple[float, float]:
        step = int(group["sf_k"]) + 1
        warmup = int(group["sf_warmup_steps"])
        base_lr = float(group["lr"])
        lr = base_lr * ((step / warmup) if step <= warmup and warmup else 1.0)
        group["sf_lr_max"] = max(float(group["sf_lr_max"]), lr)
        weight = (step ** group["sf_r"]) * (
            group["sf_lr_max"] ** group["sf_weight_lr_power"]
        )
        group["sf_weight_sum"] += weight
        ckp1 = weight / group["sf_weight_sum"] if group["sf_weight_sum"] else 0.0
        return lr, ckp1

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        if any(not group["sf_train_mode"] for group in self.param_groups):
            raise RuntimeError("APOLLO-SF requires train() before step().")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Keep checkpoints created before delta-commit support resumable.
            group.setdefault("sf_delta_refresh", "none")
            group.setdefault("sf_delta_commit_count", 0)
            group.setdefault("sf_delta_commit_norm_sum", 0.0)
            group.setdefault("sf_delta_commit_norm_max", 0.0)
            group.setdefault("sf_delta_refresh_window", 4)
            group.setdefault("sf_delta_blend_count", 0)
            group.setdefault("sf_delta_blend_step_count", 0)
            group.setdefault("sf_delta_blend_norm_sum", 0.0)
            lr, ckp1 = self._schedule(group)
            beta_sf = group["sf_beta1"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("APOLLO-SF does not support sparse gradients")
                state = self.state[parameter]
                self._ensure_storage(parameter, state, group)
                backend = self._select_backend(parameter, state, group)
                if backend != "came" and parameter.ndim >= 2:
                    self._ensure_state(parameter, state, group)
                    self._ensure_low_rank_delta_storage(parameter, state, group)
                original = parameter.float()
                delta = self._decode_state(parameter, state, group)
                parameter.copy_((original + ckp1 * delta).to(parameter.dtype))

                projection_refreshed = False
                if backend == "came" and parameter.ndim >= 2:
                    update = self._full_came_matrix_update(
                        parameter, gradient, state, group,
                    )
                elif backend == "came":
                    update = self._fallback_scale(parameter, gradient, state, group)
                else:
                    state["step"] = int(state.get("step", 0)) + 1
                    if parameter.ndim >= 2:
                        self._rotate_orthogonal_projection_state(
                            state, group, parameter=parameter,
                        )
                        projection_refreshed = bool(
                            self._refresh_projection(parameter, state, group)
                        )
                    grad_matrix = self._matrix_view(gradient.float())
                    _, scaling = self._low_rank_update(
                        parameter, gradient, state, group,
                        grad_matrix=grad_matrix,
                    )
                    update = self._apply_scaling(
                        grad_matrix, scaling, self.scale_type, gradient.shape,
                    )
                    scale_factor = math.sqrt(group["scale"])
                    if group["scale_front"]:
                        update = update * scale_factor
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
                        state["scaled_grad_norm"] = current_norm.detach()
                    if not group["scale_front"]:
                        update = update * scale_factor
                    if limiter_ratio is not None:
                        update = update * limiter_ratio

                effective_update = update.float()
                if group["weight_decay"] != 0.0:
                    effective_update = effective_update.add(
                        original, alpha=group["weight_decay"],
                    )
                update_scale = lr * (beta_sf * (1.0 - ckp1) - 1.0)
                delta_scale = -(1.0 - ckp1) * lr * beta_sf
                new_parameter = parameter.float().add(
                    effective_update, alpha=update_scale,
                )
                new_delta = delta.mul(1.0 - ckp1).add(
                    effective_update, alpha=delta_scale,
                )
                new_z = new_parameter + new_delta
                if _delta_mode(group["sf_state_storage"]):
                    refresh_policy = group.get("sf_delta_refresh", "none")
                    if refresh_policy == "commit_z" and projection_refreshed:
                        commit_norm = float(new_delta.norm().item())
                        group["sf_delta_commit_count"] += 1
                        group["sf_delta_commit_norm_sum"] += commit_norm
                        group["sf_delta_commit_norm_max"] = max(
                            group["sf_delta_commit_norm_max"], commit_norm,
                        )
                        # Commit the hidden Schedule-Free trajectory, rather
                        # than discarding it by merely clearing the delta.
                        new_parameter = new_z
                        new_delta = torch.zeros_like(new_delta)
                    elif refresh_policy == "blend":
                        if projection_refreshed:
                            remaining = group["sf_delta_refresh_window"]
                            group["sf_delta_blend_count"] += 1
                        else:
                            remaining = int(
                                state.get("sf_delta_blend_remaining", 0)
                            )
                        if remaining > 0:
                            merge_norm = float(new_delta.norm().item())
                            alpha = 1.0 / remaining
                            # Move part of the current drift into y while
                            # preserving z = y + delta exactly in FP32.
                            new_parameter = new_parameter.add(
                                new_delta, alpha=alpha,
                            )
                            new_delta = new_delta * (1.0 - alpha)
                            remaining -= 1
                            if remaining == 0:
                                new_delta = torch.zeros_like(new_delta)
                            state["sf_delta_blend_remaining"] = remaining
                            group["sf_delta_blend_step_count"] += 1
                            group["sf_delta_blend_norm_sum"] += merge_norm * alpha
                parameter.copy_(new_parameter.to(parameter.dtype))
                self._store_state(
                    parameter, state, group, z=new_z, delta=new_delta,
                )
            group["sf_k"] += 1
        return loss

    def estimate_parameter_state_bytes(self, parameter: torch.Tensor) -> int:
        group = self._parameter_group_for(parameter, self.param_groups)
        base = super().estimate_parameter_state_bytes(parameter)
        mode = group["sf_state_storage"]
        if mode == "bf16_z":
            return base + parameter.numel() * parameter.element_size()
        if _low_rank_delta_mode(mode):
            state = self.state.get(parameter, {})
            if "sf_delta_full" in state:
                return base + parameter.numel() * 4
            matrix = self._matrix_view(parameter)
            rank = min(int(group["rank"]), min(matrix.shape))
            # In addition to the latent delta, low_rank_delta owns a separate
            # orthonormal basis.  The APOLLO projection is already included
            # in ``base`` and must not be counted as the delta basis.
            return base + rank * (max(matrix.shape) + min(matrix.shape)) * 4
        bits = 8 if "int8" in mode else 4
        values = (parameter.numel() * bits + 7) // 8
        blocks = (parameter.numel() + group["sf_quant_block_size"] - 1) // group[
            "sf_quant_block_size"
        ]
        return base + values + blocks * 4


APOLLOSF = APOLLOScheduleFree


__all__ = ["APOLLOScheduleFree", "APOLLOSF"]
