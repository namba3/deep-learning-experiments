"""Reusable projection refresh policy and low-rank state transitions.

The helper keeps projection-refresh bookkeeping out of optimizer update rules.
It supports a fixed basis, a transported hard replacement, PA/PB smooth,
stochastic, or EMA transitions, and a persistent shadow-state transition.
The latter stores two low-rank coefficient/projection branches while sharing
the full optimizer state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Callable

import torch


@dataclass(frozen=True)
class ProjectionRefreshPolicy:
    """Checkpoint-safe configuration for a low-rank projection refresh."""

    mode: str = "none"
    interval: int = 0
    window: int = 0
    mix: str = "smoothstep"
    ema_decay: float | None = None
    diagnostics: bool = False
    transport_overlap: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"none", "hard", "smooth", "shadow"}:
            raise ValueError(
                "refresh mode must be 'none', 'hard', 'smooth', or 'shadow'"
            )
        if self.interval < 0:
            raise ValueError("refresh interval must be non-negative")
        if self.mode != "none" and self.interval <= 0:
            raise ValueError("refresh interval must be positive when enabled")
        if self.mode == "smooth" and self.window <= 0:
            raise ValueError("smooth refresh window must be positive")
        if self.mode != "smooth" and self.window < 0:
            raise ValueError("refresh window must be non-negative")
        if self.mix not in {"linear", "smoothstep", "stochastic", "ema"}:
            raise ValueError(
                "refresh mix must be 'linear', 'smoothstep', 'stochastic', or 'ema'"
            )
        if self.ema_decay is not None and (
            not math.isfinite(self.ema_decay)
            or self.ema_decay < 0.0
            or self.ema_decay >= 1.0
        ):
            raise ValueError("EMA refresh decay must be in [0, 1)")
        if not isinstance(self.diagnostics, bool):
            raise TypeError("refresh diagnostics must be a boolean")
        if self.transport_overlap is not None and (
            not math.isfinite(self.transport_overlap)
            or self.transport_overlap < 0.0
            or self.transport_overlap > 1.0
        ):
            raise ValueError("transport overlap must be in [0, 1]")

    @classmethod
    def from_value(
        cls,
        value: "ProjectionRefreshPolicy | Mapping[str, object] | str | None",
    ) -> "ProjectionRefreshPolicy":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(mode=value, interval=0 if value == "none" else 200,
                       window=0 if value != "smooth" else 200)
        if not isinstance(value, Mapping):
            raise TypeError(
                "projection refresh must be a string, mapping, or "
                "ProjectionRefreshPolicy"
            )
        known = {
            "mode", "interval", "window", "mix", "ema_decay", "diagnostics",
            "transport_overlap",
        }
        unknown = set(value) - known
        if unknown:
            raise ValueError(
                "unknown projection refresh keys: "
                + ", ".join(sorted(str(key) for key in unknown))
            )
        mode = value.get("mode", "none")
        interval = value.get("interval", 0 if mode == "none" else 200)
        window = value.get("window", interval if mode == "smooth" else 0)
        mix = value.get("mix", "smoothstep")
        ema_decay = value.get("ema_decay")
        diagnostics = value.get("diagnostics", False)
        transport_overlap = value.get("transport_overlap")
        if not isinstance(mode, str) or not isinstance(mix, str):
            raise TypeError("refresh mode and mix must be strings")
        if not isinstance(interval, int) or not isinstance(window, int):
            raise TypeError("refresh interval and window must be integers")
        if ema_decay is not None and (
            not isinstance(ema_decay, (float, int))
            or isinstance(ema_decay, bool)
        ):
            raise TypeError("EMA refresh decay must be numeric or None")
        if not isinstance(diagnostics, bool):
            raise TypeError("refresh diagnostics must be a boolean")
        if transport_overlap is not None and (
            not isinstance(transport_overlap, (float, int))
            or isinstance(transport_overlap, bool)
        ):
            raise TypeError("transport overlap must be numeric or None")
        return cls(
            mode=mode,
            interval=interval,
            window=window,
            mix=mix,
            ema_decay=None if ema_decay is None else float(ema_decay),
            diagnostics=diagnostics,
            transport_overlap=(
                None
                if transport_overlap is None
                else float(transport_overlap)
            ),
        )

    def as_dict(self) -> dict[str, object]:
        result = {
            "mode": self.mode,
            "interval": self.interval,
            "window": self.window,
            "mix": self.mix,
        }
        if self.ema_decay is not None:
            result["ema_decay"] = self.ema_decay
        if self.diagnostics:
            result["diagnostics"] = True
        if self.transport_overlap is not None:
            result["transport_overlap"] = self.transport_overlap
        return result


@dataclass(frozen=True)
class OrthogonalRefreshPolicy:
    """Per-step tangent-space rotation for a low-rank projection.

    ``direction="loss_directed"`` uses a first-order local proxy: it moves
    toward a larger projected-gradient energy rather than sampling an
    isotropic tangent direction.  ``direction="loss_lowering"`` is available
    for LRSF and moves the basis down the first-order loss gradient of the
    decoded hidden delta.  Neither direction evaluates the model loss.
    """

    rate: float = 0.0
    seed: int = 0
    direction: str = "random"
    signal: str = "gradient"

    def __post_init__(self) -> None:
        if not math.isfinite(self.rate) or self.rate < 0.0:
            raise ValueError("orthogonal refresh rate must be finite and non-negative")
        if self.direction not in {"random", "loss_directed", "loss_lowering"}:
            raise ValueError(
                "orthogonal refresh direction must be 'random' or "
                "'loss_directed' or 'loss_lowering'"
            )
        if self.signal not in {"gradient", "effective_update"}:
            raise ValueError(
                "orthogonal refresh signal must be 'gradient' or "
                "'effective_update'"
            )

    @classmethod
    def from_value(
        cls,
        value: "OrthogonalRefreshPolicy | Mapping[str, object] | float | int | None",
        *,
        default_seed: int = 0,
    ) -> "OrthogonalRefreshPolicy":
        if value is None:
            return cls(seed=int(default_seed))
        if isinstance(value, cls):
            return value
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            return cls(rate=float(value), seed=int(default_seed))
        if not isinstance(value, Mapping):
            raise TypeError(
                "orthogonal refresh must be a number, mapping, or "
                "OrthogonalRefreshPolicy"
            )
        unknown = set(value) - {"rate", "seed", "direction", "signal"}
        if unknown:
            raise ValueError(
                "unknown orthogonal refresh keys: "
                + ", ".join(sorted(str(key) for key in unknown))
            )
        rate = value.get("rate", 0.0)
        seed = value.get("seed", default_seed)
        direction = value.get("direction", "random")
        signal = value.get("signal", "gradient")
        if not isinstance(rate, (float, int)) or isinstance(rate, bool):
            raise TypeError("orthogonal refresh rate must be numeric")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("orthogonal refresh seed must be an integer")
        if not isinstance(direction, str):
            raise TypeError("orthogonal refresh direction must be a string")
        if not isinstance(signal, str):
            raise TypeError("orthogonal refresh signal must be a string")
        return cls(
            rate=float(rate), seed=int(seed), direction=direction, signal=signal,
        )

    def as_dict(self) -> dict[str, object]:
        result = {"rate": self.rate, "seed": self.seed}
        if self.direction != "random":
            result["direction"] = self.direction
        if self.signal != "gradient":
            result["signal"] = self.signal
        return result


def _transport_delta(
    delta: torch.Tensor,
    old_projection: torch.Tensor,
    new_projection: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Transport coefficient matrix while preserving its decoded component."""
    if rows >= cols:
        return delta.matmul(old_projection.transpose(0, 1)).matmul(new_projection)
    return new_projection.matmul(old_projection.transpose(0, 1)).matmul(delta)


def _blend_refresh_projection(
    old_projection: torch.Tensor,
    new_projection: torch.Tensor,
    overlap: float | None,
) -> torch.Tensor:
    """Blend a new random basis with the old basis before orthogonalizing."""
    if overlap is None:
        return new_projection
    if overlap == 1.0:
        return old_projection.detach().clone()
    if overlap == 0.0:
        return new_projection
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


def _record_transport_diagnostic(
    state: dict,
    old_delta: torch.Tensor,
    old_projection: torch.Tensor,
    new_delta: torch.Tensor,
    new_projection: torch.Tensor,
    rows: int,
    cols: int,
    step: int,
) -> None:
    """Record decoded-delta distortion for one projection transport event.

    This intentionally decodes full matrices only when diagnostics are enabled.
    The tensors are temporary and are not retained in optimizer state.
    """
    if rows >= cols:
        old_decoded = old_delta.matmul(old_projection.transpose(0, 1))
        new_decoded = new_delta.matmul(new_projection.transpose(0, 1))
    else:
        old_decoded = old_projection.transpose(0, 1).matmul(old_delta)
        new_decoded = new_projection.transpose(0, 1).matmul(new_delta)
    old_flat = old_decoded.float().reshape(-1)
    new_flat = new_decoded.float().reshape(-1)
    old_norm = torch.linalg.vector_norm(old_flat)
    new_norm = torch.linalg.vector_norm(new_flat)
    denominator = old_norm.clamp_min(1e-30)
    error = torch.linalg.vector_norm(new_flat - old_flat) / denominator
    norm_ratio = new_norm / denominator
    cosine = torch.dot(old_flat, new_flat) / (
        old_norm.clamp_min(1e-30) * new_norm.clamp_min(1e-30)
    )

    count = int(state.get("refresh_transport_diagnostic_count", 0)) + 1
    state["refresh_transport_diagnostic_count"] = count
    state["refresh_transport_error_sum"] = (
        float(state.get("refresh_transport_error_sum", 0.0)) + float(error)
    )
    state["refresh_transport_error_max"] = max(
        float(state.get("refresh_transport_error_max", 0.0)), float(error),
    )
    state["refresh_transport_norm_ratio_sum"] = (
        float(state.get("refresh_transport_norm_ratio_sum", 0.0))
        + float(norm_ratio)
    )
    state["refresh_transport_cosine_sum"] = (
        float(state.get("refresh_transport_cosine_sum", 0.0))
        + float(cosine)
    )
    state["refresh_transport_last_step"] = int(step)


def _mix_weight(state: dict, policy: ProjectionRefreshPolicy) -> float:
    if not state.get("refresh_active", False):
        return 0.0
    progress = min(int(state.get("refresh_progress", 0)), policy.window)
    value = progress / policy.window
    if policy.mix == "smoothstep":
        return value * value * (3.0 - 2.0 * value)
    if policy.mix == "ema":
        return float(state.get("refresh_ema_weight", 0.0))
    return value


def _ema_decay(policy: ProjectionRefreshPolicy) -> float:
    if policy.ema_decay is not None:
        return policy.ema_decay
    return math.exp(-1.0 / max(policy.window, 1))


def initialize_refresh_mix(
    state: dict,
    policy: ProjectionRefreshPolicy,
) -> None:
    if policy.mix == "ema":
        state["refresh_ema_weight"] = 0.0


def advance_refresh_mix(
    state: dict,
    policy: ProjectionRefreshPolicy,
) -> None:
    if policy.mix != "ema" or not state.get("refresh_active", False):
        return
    decay = _ema_decay(policy)
    previous = float(state.get("refresh_ema_weight", 0.0))
    state["refresh_ema_weight"] = decay * previous + (1.0 - decay)


def refresh_mix_weight(state: dict, policy: ProjectionRefreshPolicy) -> float:
    """Return the current PA-to-PB transition weight or PB probability."""
    return _mix_weight(state, policy)


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (value ^ (value >> 31)) & 0xFFFFFFFFFFFFFFFF


def prepare_stochastic_refresh(
    state: dict,
    policy: ProjectionRefreshPolicy,
    seed: int,
) -> int:
    """Select PA or PB once for the current step.

    Both EMA buffers continue to receive the projected update.  The sampled
    branch is used for decoding the parameter, and the same branch is reused
    by train/eval restoration during that step.
    """
    if policy.mix != "stochastic":
        return 0
    if not state.get("refresh_active", False):
        state["refresh_stochastic_choice"] = 0
        return 0
    counter = int(state.get("refresh_stochastic_counter", 0))
    random_bits = _splitmix64(int(seed) + counter)
    uniform = (random_bits >> 11) / float(1 << 53)
    probability_pb = _mix_weight(state, policy)
    choice = int(uniform < probability_pb)
    state["refresh_stochastic_counter"] = counter + 1
    state["refresh_stochastic_choice"] = choice
    return choice


def _transport_projected_delta(
    delta: torch.Tensor,
    old_projection: torch.Tensor,
    new_projection: torch.Tensor,
) -> torch.Tensor:
    if old_projection.shape[0] >= old_projection.shape[1]:
        return delta.matmul(old_projection.transpose(0, 1)).matmul(new_projection)
    return new_projection.matmul(old_projection.transpose(0, 1)).matmul(delta)


def ensure_shadow_state(
    parameter: torch.Tensor,
    state: dict,
    policy: ProjectionRefreshPolicy,
    seed: int,
    make_projection: Callable[[torch.Tensor, int, int], torch.Tensor],
    projection_key: str = "lrsf_projection",
    delta_key: str = "lrsf_delta",
    shadow_projection_key: str = "lrsf_shadow_projection",
    shadow_delta_key: str = "lrsf_shadow_delta",
) -> None:
    """Create the persistent low-rank shadow branch when requested."""
    if policy.mode != "shadow":
        state["shadow_active"] = False
        return
    if shadow_projection_key in state:
        state["shadow_active"] = True
        return
    projection = state[projection_key]
    delta = state[delta_key]
    rank = (
        projection.shape[-1]
        if projection.shape[0] >= projection.shape[1]
        else projection.shape[0]
    )
    shadow_projection = make_projection(
        parameter, rank, int(seed) + 1,
    )
    shadow_projection = _blend_refresh_projection(
        projection, shadow_projection, policy.transport_overlap,
    )
    matrix = parameter.reshape(parameter.shape[0], -1)
    state[shadow_projection_key] = shadow_projection
    state[shadow_delta_key] = _transport_delta(
        delta,
        projection,
        shadow_projection,
        matrix.shape[0],
        matrix.shape[1],
    )
    state["shadow_active"] = True
    state["shadow_generation"] = 0


def _record_shadow_gap(
    state: dict,
    active_delta: torch.Tensor,
    active_projection: torch.Tensor,
    shadow_delta: torch.Tensor,
    shadow_projection: torch.Tensor,
    rows: int,
    cols: int,
    step: int,
) -> None:
    """Record the decoded active/shadow gap immediately before promotion."""
    if rows >= cols:
        active_decoded = active_delta.matmul(active_projection.transpose(0, 1))
        shadow_decoded = shadow_delta.matmul(shadow_projection.transpose(0, 1))
    else:
        active_decoded = active_projection.transpose(0, 1).matmul(active_delta)
        shadow_decoded = shadow_projection.transpose(0, 1).matmul(shadow_delta)
    active_flat = active_decoded.float().reshape(-1)
    shadow_flat = shadow_decoded.float().reshape(-1)
    active_norm = torch.linalg.vector_norm(active_flat)
    shadow_norm = torch.linalg.vector_norm(shadow_flat)
    denominator = active_norm.clamp_min(1e-30)
    error = torch.linalg.vector_norm(shadow_flat - active_flat) / denominator
    norm_ratio = shadow_norm / denominator
    cosine = torch.dot(active_flat, shadow_flat) / (
        active_norm.clamp_min(1e-30) * shadow_norm.clamp_min(1e-30)
    )
    count = int(state.get("shadow_gap_count", 0)) + 1
    state["shadow_gap_count"] = count
    state["shadow_gap_error_sum"] = (
        float(state.get("shadow_gap_error_sum", 0.0)) + float(error)
    )
    state["shadow_gap_error_max"] = max(
        float(state.get("shadow_gap_error_max", 0.0)), float(error),
    )
    state["shadow_gap_norm_ratio_sum"] = (
        float(state.get("shadow_gap_norm_ratio_sum", 0.0))
        + float(norm_ratio)
    )
    state["shadow_gap_cosine_sum"] = (
        float(state.get("shadow_gap_cosine_sum", 0.0)) + float(cosine)
    )
    state["shadow_gap_last_step"] = int(step)


def _start_shadow_refresh(
    parameter: torch.Tensor,
    state: dict,
    policy: ProjectionRefreshPolicy,
    step: int,
    seed: int,
    make_projection: Callable[[torch.Tensor, int, int], torch.Tensor],
    projection_key: str,
    delta_key: str,
    shadow_projection_key: str,
    shadow_delta_key: str,
) -> bool:
    """Promote the trained shadow branch and start a new shadow branch."""
    active_projection = state[projection_key]
    active_delta = state[delta_key]
    shadow_projection = state[shadow_projection_key]
    shadow_delta = state[shadow_delta_key]
    matrix = parameter.reshape(parameter.shape[0], -1)
    if policy.diagnostics:
        _record_shadow_gap(
            state,
            active_delta,
            active_projection,
            shadow_delta,
            shadow_projection,
            matrix.shape[0],
            matrix.shape[1],
            step,
        )

    state[projection_key] = shadow_projection
    state[delta_key] = shadow_delta
    refresh_count = int(state.get("refresh_count", 0)) + 1
    rank = (
        shadow_projection.shape[-1]
        if shadow_projection.shape[0] >= shadow_projection.shape[1]
        else shadow_projection.shape[0]
    )
    new_projection = make_projection(
        parameter, rank, int(seed) + refresh_count + 1,
    )
    new_projection = _blend_refresh_projection(
        state[projection_key], new_projection, policy.transport_overlap,
    )
    new_delta = _transport_delta(
        state[delta_key], state[projection_key], new_projection,
        matrix.shape[0], matrix.shape[1],
    )
    if policy.diagnostics:
        _record_transport_diagnostic(
            state,
            state[delta_key],
            state[projection_key],
            new_delta,
            new_projection,
            matrix.shape[0],
            matrix.shape[1],
            step,
        )
    state[shadow_projection_key] = new_projection
    state[shadow_delta_key] = new_delta
    state["refresh_count"] = refresh_count
    state["shadow_generation"] = refresh_count
    return True


def rotate_orthogonal_projection(
    projection: torch.Tensor,
    rate: float,
    seed: int,
    *,
    direction: str = "random",
    gradient: torch.Tensor | None = None,
    delta: torch.Tensor | None = None,
) -> torch.Tensor:
    if rate == 0.0:
        return projection
    if direction not in {"random", "loss_directed", "loss_lowering"}:
        raise ValueError(
            "orthogonal refresh direction must be 'random', 'loss_directed', "
            "or 'loss_lowering'"
        )
    if direction in {"loss_directed", "loss_lowering"}:
        if gradient is None:
            raise ValueError(
                f"{direction} orthogonal refresh requires a gradient"
            )
        matrix = gradient.detach().float().reshape(gradient.shape[0], -1)
        projection_float = projection.float()
        if direction == "loss_lowering":
            if delta is None:
                raise ValueError(
                    "loss_lowering orthogonal refresh requires a low-rank delta"
                )
            delta_matrix = delta.detach().float()
            if matrix.shape[0] >= matrix.shape[1]:
                if delta_matrix.shape != (
                    matrix.shape[0], projection.shape[1]
                ):
                    raise ValueError(
                        "loss_lowering delta shape must be (rows, rank) "
                        "for a tall projection"
                    )
                tangent = -matrix.transpose(0, 1).matmul(delta_matrix)
                tangent.sub_(
                    projection_float.matmul(
                        projection_float.transpose(0, 1).matmul(tangent)
                    )
                )
            else:
                if delta_matrix.shape != (
                    projection.shape[0], matrix.shape[1]
                ):
                    raise ValueError(
                        "loss_lowering delta shape must be (rank, cols) "
                        "for a wide projection"
                    )
                tangent = -delta_matrix.matmul(matrix.transpose(0, 1))
                tangent.sub_(
                    tangent.matmul(
                        projection_float.transpose(0, 1).matmul(projection_float)
                    )
                )
        elif matrix.shape[0] >= matrix.shape[1]:
            projected = matrix.matmul(projection_float)
            tangent = matrix.transpose(0, 1).matmul(projected)
            tangent.sub_(
                projection_float.matmul(
                    projection_float.transpose(0, 1).matmul(tangent)
                )
            )
        else:
            projected = projection_float.matmul(matrix)
            tangent = projected.matmul(matrix.transpose(0, 1))
            tangent.sub_(
                tangent.matmul(
                    projection_float.transpose(0, 1).matmul(projection_float)
                )
            )
        tangent = tangent.to(dtype=projection.dtype)
    else:
        generator = torch.Generator(device=projection.device).manual_seed(int(seed))
        noise = torch.randn(
            projection.shape,
            generator=generator,
            device=projection.device,
            dtype=projection.dtype,
        )
        if projection.shape[0] >= projection.shape[1]:
            tangent = noise - projection.matmul(
                projection.transpose(0, 1).matmul(noise)
            )
        else:
            tangent = noise - noise.matmul(
                projection.transpose(0, 1).matmul(projection)
            )
    if projection.shape[0] >= projection.shape[1]:
        rank = projection.shape[1]
        candidate = projection + tangent * (
            rate * math.sqrt(rank)
            / torch.linalg.vector_norm(tangent).clamp_min(1e-30)
        )
        return torch.linalg.qr(candidate, mode="reduced").Q.contiguous()
    rank = projection.shape[0]
    candidate = projection + tangent * (
        rate * math.sqrt(rank)
        / torch.linalg.vector_norm(tangent).clamp_min(1e-30)
    )
    return torch.linalg.qr(candidate.transpose(0, 1), mode="reduced").Q.transpose(
        0, 1
    ).contiguous()


def rotate_projection_state(
    state: dict,
    policy: OrthogonalRefreshPolicy,
    *,
    gradient: torch.Tensor | None = None,
    projection_key: str = "lrsf_projection",
    delta_key: str = "lrsf_delta",
    shadow_projection_key: str = "lrsf_shadow_projection",
    shadow_delta_key: str = "lrsf_shadow_delta",
) -> bool:
    """Rotate the active projection and transport all active EMA buffers."""
    if policy.rate == 0.0 or projection_key not in state:
        return False
    count = int(state.get("orthogonal_refresh_count", 0))
    seed = int(policy.seed) + 2 * count
    old_projection = state[projection_key]
    new_projection = rotate_orthogonal_projection(
        old_projection,
        policy.rate,
        seed,
        direction=policy.direction,
        gradient=gradient,
        delta=state.get(delta_key),
    )
    state[projection_key] = new_projection
    if delta_key in state:
        state[delta_key] = _transport_projected_delta(
            state[delta_key], old_projection, new_projection,
        )
    if state.get("shadow_active", False) and shadow_projection_key in state:
        old_shadow_projection = state[shadow_projection_key]
        new_shadow_projection = rotate_orthogonal_projection(
            old_shadow_projection,
            policy.rate,
            seed + 1,
            direction=policy.direction,
            gradient=gradient,
            delta=state.get(shadow_delta_key),
        )
        state[shadow_projection_key] = new_shadow_projection
        if shadow_delta_key in state:
            state[shadow_delta_key] = _transport_projected_delta(
                state[shadow_delta_key],
                old_shadow_projection,
                new_shadow_projection,
            )
    if state.get("refresh_active", False):
        old_next_projection = state["refresh_next_projection"]
        new_next_projection = rotate_orthogonal_projection(
            old_next_projection,
            policy.rate,
            seed + 1,
            direction=policy.direction,
            gradient=gradient,
            delta=state.get("refresh_next_delta"),
        )
        state["refresh_next_projection"] = new_next_projection
        state["refresh_next_delta"] = _transport_projected_delta(
            state["refresh_next_delta"],
            old_next_projection,
            new_next_projection,
        )
    state["orthogonal_refresh_count"] = count + 1
    return True


def maybe_start_refresh(
    parameter: torch.Tensor,
    state: dict,
    policy: ProjectionRefreshPolicy,
    step: int,
    seed: int,
    make_projection: Callable[[torch.Tensor, int, int], torch.Tensor],
    projection_key: str = "lrsf_projection",
    delta_key: str = "lrsf_delta",
    shadow_projection_key: str = "lrsf_shadow_projection",
    shadow_delta_key: str = "lrsf_shadow_delta",
) -> bool:
    """Start or apply a refresh before the current optimizer update."""
    if policy.mode == "none" or step <= 0 or step % policy.interval != 0:
        return False
    if policy.mode == "shadow":
        return _start_shadow_refresh(
            parameter,
            state,
            policy,
            step,
            seed,
            make_projection,
            projection_key,
            delta_key,
            shadow_projection_key,
            shadow_delta_key,
        )
    if state.get("refresh_active", False):
        return False
    old_projection = state[projection_key]
    old_delta = state[delta_key]
    refresh_count = int(state.get("refresh_count", 0)) + 1
    rank = old_projection.shape[-1] if old_projection.shape[0] >= old_projection.shape[1] else old_projection.shape[0]
    new_projection = make_projection(parameter, rank, int(seed) + refresh_count)
    new_projection = _blend_refresh_projection(
        old_projection, new_projection, policy.transport_overlap,
    )
    matrix = parameter.reshape(parameter.shape[0], -1)
    new_delta = _transport_delta(
        old_delta, old_projection, new_projection, matrix.shape[0], matrix.shape[1],
    )
    state["refresh_count"] = refresh_count
    if policy.diagnostics:
        _record_transport_diagnostic(
            state,
            old_delta,
            old_projection,
            new_delta,
            new_projection,
            matrix.shape[0],
            matrix.shape[1],
            step,
        )
    if policy.mode == "hard":
        state[projection_key] = new_projection
        state[delta_key] = new_delta
        return True
    state["refresh_next_projection"] = new_projection
    state["refresh_next_delta"] = new_delta
    state["refresh_progress"] = 0
    state["refresh_active"] = True
    initialize_refresh_mix(state, policy)
    return True


def add_mixed_delta(
    matrix: torch.Tensor,
    state: dict,
    alpha: float,
    add_low_rank: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], None],
    policy: ProjectionRefreshPolicy,
    projection_key: str = "lrsf_projection",
    delta_key: str = "lrsf_delta",
) -> None:
    if policy.mix == "stochastic" and state.get("refresh_active", False):
        if int(state.get("refresh_stochastic_choice", 0)) == 0:
            add_low_rank(
                matrix, state[delta_key], state[projection_key], alpha,
            )
        else:
            add_low_rank(
                matrix,
                state["refresh_next_delta"],
                state["refresh_next_projection"],
                alpha,
            )
        return
    weight = _mix_weight(state, policy)
    add_low_rank(matrix, state[delta_key], state[projection_key], alpha * (1.0 - weight))
    if state.get("refresh_active", False):
        add_low_rank(
            matrix,
            state["refresh_next_delta"],
            state["refresh_next_projection"],
            alpha * weight,
        )


def project_mixed_update(
    update: torch.Tensor,
    state: dict,
    project_update: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    projection_key: str = "lrsf_projection",
) -> tuple[torch.Tensor, ...]:
    current = project_update(update, state[projection_key])
    if state.get("shadow_active", False):
        shadow = project_update(update, state["lrsf_shadow_projection"])
        return current, shadow
    if not state.get("refresh_active", False):
        return (current,)
    next_value = project_update(update, state["refresh_next_projection"])
    return current, next_value


def update_mixed_delta(
    state: dict,
    projected_updates: tuple[torch.Tensor, ...],
    decay: float,
    update_scale: float,
    delta_key: str = "lrsf_delta",
) -> None:
    state[delta_key].mul_(decay).add_(projected_updates[0], alpha=update_scale)
    if state.get("shadow_active", False):
        state["lrsf_shadow_delta"].mul_(decay).add_(
            projected_updates[-1], alpha=update_scale,
        )
        return
    if state.get("refresh_active", False):
        state["refresh_next_delta"].mul_(decay).add_(
            projected_updates[-1], alpha=update_scale,
        )


def advance_refresh(
    state: dict,
    policy: ProjectionRefreshPolicy,
    projection_key: str = "lrsf_projection",
    delta_key: str = "lrsf_delta",
) -> None:
    if not state.get("refresh_active", False):
        return
    advance_refresh_mix(state, policy)
    state["refresh_progress"] = int(state.get("refresh_progress", 0)) + 1
    if state["refresh_progress"] < policy.window:
        return
    state[projection_key] = state.pop("refresh_next_projection")
    state[delta_key] = state.pop("refresh_next_delta")
    state["refresh_progress"] = 0
    state["refresh_active"] = False
    state.pop("refresh_ema_weight", None)


__all__ = [
    "OrthogonalRefreshPolicy",
    "ProjectionRefreshPolicy",
    "add_mixed_delta",
    "advance_refresh",
    "advance_refresh_mix",
    "ensure_shadow_state",
    "initialize_refresh_mix",
    "maybe_start_refresh",
    "prepare_stochastic_refresh",
    "project_mixed_update",
    "refresh_mix_weight",
    "rotate_orthogonal_projection",
    "rotate_projection_state",
    "update_mixed_delta",
]
