"""Common optimizer registry, CLI choices, and construction helpers."""

from __future__ import annotations

import argparse
import warnings
from typing import Any, TypeVar, cast

from .adamw import AdamWAutoSchedule, AdamWFP32State
from .adamw_lr_ema import (
    AdamWLowRankGradientEMA,
    AdamWLowRankGradientEMAConfidence,
)
from .adamw_lr_ema_conf_lrsf import AdamWLRSEMAConfLRSF
from .adamw_lrsf import AdamWLRSF
from .adamw_lrsf_lr import AdamWLRSLowRankPreconditioner
from .adamw_sf_lr import AdamWSFLowRankPreconditioner
from .apollo import (
    APOLLO,
    APOLLOConfidence,
    APOLLOFallbackPolicy,
    APOLLOADAMW,
    APOLLOADAMWAutoSchedule,
    APOLLOLion,
    APOLLOAutoSchedule,
    APOLLOCAME,
    APOLLOCAMEAutoSchedule,
    APOLLOMini,
    DualRotAPOLLO,
    RotAPOLLO,
)
from .came import CAME, CAMEAutoSchedule
from .came_lrsf import CAMESF, CAMELRSF
from .apollo_lrsf import APOLLOCAMELRSF
from .apollo_sf import APOLLOScheduleFree
from .lion import Lion
from .muon import SingleDeviceMuon as Muon
from .muon_variants import AdaMuon, NorMuon
from .schedulefree import AdamWScheduleFree, RAdamScheduleFree
from .soap import SOAP

_T = TypeVar("_T")


CORE_OPTIMIZER_CHOICES = (
    "AdamW",
    "CAME",
    "APOLLO",
    "APOLLO-CAME",
)
LEGACY_OPTIMIZER_CHOICES = ("RAdamSF", "AdamWSF")
EXPERIMENTAL_OPTIMIZER_CHOICES = (
    "CAME-SF",
    "CAME-LRSF",
    "AdamW-SF",
    "AdamW-LRSF",
    "AdamW-SF-LR",
    "AdamW-LRSF-LR",
    "AdamW-LR-EMA",
    "AdamW-LR-EMA-Conf",
    "AdamW-LR-EMA-Conf-LRSF",
    "APOLLO-Conf",
    "APOLLO-CAME-LRSF",
    "APOLLO-SF",
    "APOLLO-SF-LRSF",
    "APOLLO-SF-INT8-Z",
    "APOLLO-SF-INT8-Delta",
    "APOLLO-SF-INT4-Z",
    "APOLLO-SF-INT4-Delta",
    "CAME-AutoSchedule",
    "AdamW-AutoSchedule",
    "APOLLO-AutoSchedule",
    "APOLLO-CAME-AutoSchedule",
    "APOLLO-Mini",
    "APOLLO-AdamW",
    "APOLLO-AdamW-AutoSchedule",
    "APOLLO-Lion",
    "RotAPOLLO",
    "DualRotAPOLLO",
    "Lion",
    "Muon",
    "SOAP",
    "NorMuon",
    "AdaMuon",
)
IMAGE_GEN_OPTIMIZER_CHOICES = (
    *CORE_OPTIMIZER_CHOICES,
    *LEGACY_OPTIMIZER_CHOICES,
    *EXPERIMENTAL_OPTIMIZER_CHOICES,
)

_AUTO_SCHEDULE_DEFAULTS = {
    "auto_schedule_target_update_ratio": 1e-3,
    "auto_schedule_ema_beta": 0.99,
    "auto_schedule_trust_alpha": 0.1,
    "auto_schedule_min_factor": 0.5,
    "auto_schedule_max_factor": 4.0,
    "auto_schedule_max_increase": 1.05,
    "auto_schedule_max_decrease": 0.95,
    "auto_schedule_confidence_floor": 0.25,
    "auto_schedule_stability_gain": 2.0,
    "auto_schedule_limiter_gain": 2.0,
    "auto_schedule_cooldown_steps": 4,
    "auto_schedule_warmup_steps": 200,
}


def add_optimizer_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str = "AdamW",
    include_experimental: bool = False,
    include_apollo_mini: bool = False,
    include_came_lrsf: bool = False,
    include_came_sf: bool = False,
    include_apollo_came_lrsf: bool = False,
    include_legacy: bool = False,
):
    """Add the common optimizer option to a training parser."""
    choices: list[str] = list(CORE_OPTIMIZER_CHOICES)
    if include_apollo_mini:
        choices.append("APOLLO-Mini")
    if include_came_lrsf:
        choices.append("CAME-LRSF")
    if include_came_sf:
        choices.append("CAME-SF")
    if include_apollo_came_lrsf:
        choices.append("APOLLO-CAME-LRSF")
    if include_legacy:
        choices.extend(LEGACY_OPTIMIZER_CHOICES)
    if include_experimental:
        choices.extend(EXPERIMENTAL_OPTIMIZER_CHOICES)
    parser.add_argument(
        "--optimizer",
        type=lambda value: normalize_optimizer_name(value, choices),
        default=default,
        metavar="{" + ",".join(choices) + "}",
        help=f"Optimizer to use. Default: {default}.",
    )
    parser.add_argument(
        "--auto-schedule",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the optimizer's adaptive learning-rate controller. "
            "AutoSchedule-named legacy variants remain accepted."
        ),
    )


def add_optimizer_override_argument(
    parser: argparse.ArgumentParser,
    option: str,
    *,
    role: str,
):
    """Add an image_gen role-specific optimizer using the same registry."""
    choices = ("same",) + IMAGE_GEN_OPTIMIZER_CHOICES
    parser.add_argument(
        option,
        dest=option.lstrip("-").replace("-", "_"),
        type=lambda value: value if value == "same" else normalize_optimizer_name(
            value, IMAGE_GEN_OPTIMIZER_CHOICES
        ),
        default="same",
        metavar="{" + ",".join(choices) + "}",
        help=(
            f"Optimizer for {role} parameters only; 'same' uses --optimizer. "
            "Experimental choices are retained for image_gen compatibility."
        ),
    )


def _get(args: object | None, name: str, default: _T) -> _T:
    """Read a dynamically populated argparse namespace with a typed default."""
    if args is None:
        return default
    return cast(_T, getattr(args, name, default))


def normalize_optimizer_name(name, allowed):
    """Validate a public name while retaining legacy CLI compatibility."""
    if name in allowed:
        return name
    if name in LEGACY_OPTIMIZER_CHOICES:
        warnings.warn(
            f"optimizer {name!r} is a legacy compatibility alias and is not "
            "listed among the new common choices",
            DeprecationWarning,
            stacklevel=2,
        )
        return name
    allowed_text = ", ".join(allowed)
    raise argparse.ArgumentTypeError(
        f"invalid optimizer {name!r}; choose one of: {allowed_text}"
    )


def is_schedule_free_optimizer(name):
    """Return whether an optimizer needs train/eval mode transitions."""
    return name in (
        *LEGACY_OPTIMIZER_CHOICES, "AdamW-SF", "AdamW-LRSF", "AdamW-SF-LR",
        "AdamW-LRSF-LR",
        "AdamW-LR-EMA-Conf-LRSF",
        "APOLLO-SF", "APOLLO-SF-INT8-Z", "APOLLO-SF-INT8-Delta",
        "APOLLO-SF-INT4-Z", "APOLLO-SF-INT4-Delta",
        "CAME-SF", "CAME-LRSF",
        "APOLLO-CAME-LRSF",
    )


def _auto_schedule_kwargs(args: object | None) -> dict[str, object]:
    return {
        name: _get(args, name, default)
        for name, default in _AUTO_SCHEDULE_DEFAULTS.items()
    }


def _apollo_fallback_policy(args: object | None) -> APOLLOFallbackPolicy:
    """Build the checkpoint-safe per-parameter APOLLO fallback policy."""
    return APOLLOFallbackPolicy(
        one_dimensional=cast(str, _get(args, "apollo_fallback", "adamw-sf")),
        small_matrix=cast(str, _get(args, "apollo_matrix_fallback", "auto-sf")),
        state_margin=float(_get(args, "apollo_fallback_state_margin", 1.0)),
        min_savings_bytes=int(
            _get(args, "apollo_fallback_min_savings_bytes", 0)
        ),
    )


def _lrsf_refresh_policy(args: object | None) -> dict[str, object]:
    """Build the shared checkpoint-safe refresh policy for LRSF delta state."""
    return {
        "mode": cast(str, _get(args, "came_lrsf_refresh_mode", "none")),
        "interval": int(_get(args, "came_lrsf_refresh_interval", 200)),
        "window": int(_get(args, "came_lrsf_refresh_window", 200)),
        "mix": cast(str, _get(args, "came_lrsf_refresh_mix", "smoothstep")),
        "diagnostics": bool(
            _get(args, "came_lrsf_refresh_diagnostics", False)
        ),
        "transport_overlap": _get(
            args, "came_lrsf_refresh_transport_overlap", None
        ),
    }


def _lrsf_orthogonal_refresh_policy(args: object | None) -> dict[str, object]:
    """Build the per-step orthogonal LRSF projection rotation policy."""
    policy = {
        "rate": float(_get(args, "came_lrsf_orthogonal_refresh_rate", 0.0)),
        "seed": int(_get(args, "came_lrsf_seed", 0)),
    }
    direction = str(
        _get(args, "came_lrsf_orthogonal_refresh_direction", "random")
    )
    if direction != "random":
        policy["direction"] = direction
    signal = str(
        _get(args, "came_lrsf_orthogonal_refresh_signal", "gradient")
    )
    if signal != "gradient":
        policy["signal"] = signal
    return policy


def _apollo_orthogonal_refresh_policy(args: object | None) -> dict[str, object]:
    """Build the per-step orthogonal APOLLO projection policy."""
    policy = {
        "rate": float(_get(args, "apollo_orthogonal_refresh_rate", 0.0)),
        "seed": int(_get(args, "seed", 0)),
    }
    direction = str(
        _get(args, "apollo_orthogonal_refresh_direction", "random")
    )
    if direction != "random":
        policy["direction"] = direction
    return policy


def _resolve_auto_schedule_name(name: str, args: object | None) -> str:
    if not _get(args, "auto_schedule", False):
        return name
    if name == "AdamW":
        return "AdamW-AutoSchedule"
    if name == "CAME":
        return "CAME-AutoSchedule"
    if name == "APOLLO":
        return "APOLLO-AutoSchedule"
    if name == "APOLLO-CAME":
        return "APOLLO-CAME-AutoSchedule"
    return name


def build_optimizer(
    name: str,
    parameters,
    args: object | None = None,
    *,
    lr: float | None = None,
    weight_decay: float | None = None,
    role: str | None = None,
    adamw_betas: tuple[float, float] = (0.9, 0.999),
):
    """Build any registered optimizer with one shared parameter contract.

    ``args`` is optional so the four common optimizers can be used from small
    training scripts without copying image_gen's larger APOLLO configuration.
    When supplied, image_gen-specific role and experimental settings are read
    from it with safe defaults.
    """
    parameters = list(parameters)
    if not parameters:
        raise ValueError(f"No parameters available for optimizer {name}")
    name = _resolve_auto_schedule_name(name, args)
    lr = _get(args, "lr", 1e-3) if lr is None else lr
    weight_decay = (
        _get(args, "weight_decay", 0.0)
        if weight_decay is None
        else weight_decay
    )
    if role == "linear":
        role_lr = cast(float | None, _get(args, "linear_lr", None))
        role_weight_decay = cast(
            float | None, _get(args, "linear_weight_decay", None),
        )
        if role_lr is not None:
            lr = role_lr
        if role_weight_decay is not None:
            weight_decay = role_weight_decay
    elif role == "conv":
        role_lr = cast(float | None, _get(args, "conv_lr", None))
        role_weight_decay = cast(
            float | None, _get(args, "conv_weight_decay", None),
        )
        if role_lr is not None:
            lr = role_lr
        if role_weight_decay is not None:
            weight_decay = role_weight_decay

    auto_kwargs = cast(dict[str, Any], _auto_schedule_kwargs(args))
    if name == "AdamW":
        return AdamWFP32State(
            parameters, lr=lr, weight_decay=weight_decay, betas=adamw_betas
        )
    if name == "AdamW-AutoSchedule":
        return AdamWAutoSchedule(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=adamw_betas,
            **auto_kwargs,
        )
    if name == "AdamW-SF":
        return AdamWScheduleFree(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(
                float(_get(args, "adamw_lrsf_beta1", 0.9)),
                float(_get(args, "adamw_lrsf_beta2", 0.999)),
            ),
            warmup_steps=int(_get(args, "adamw_lrsf_warmup_steps", 0)),
            r=float(_get(args, "adamw_lrsf_r", 0.0)),
            weight_lr_power=float(
                _get(args, "adamw_lrsf_weight_lr_power", 2.0)
            ),
            backend=_get(args, "adamw_sf_backend", "auto"),
        )
    if name == "AdamW-LRSF":
        return AdamWLRSF(
            parameters,
            lr=lr,
            rank=int(_get(args, "adamw_lrsf_rank", 4)),
            sf_beta1=float(_get(args, "adamw_lrsf_beta1", 0.9)),
            beta2=float(_get(args, "adamw_lrsf_beta2", 0.999)),
            warmup_steps=int(_get(args, "adamw_lrsf_warmup_steps", 0)),
            r=float(_get(args, "adamw_lrsf_r", 0.0)),
            weight_lr_power=float(
                _get(args, "adamw_lrsf_weight_lr_power", 2.0)
            ),
            seed=int(_get(args, "adamw_lrsf_seed", 0)),
            eps=float(_get(args, "adamw_lrsf_eps", 1e-8)),
            weight_decay=weight_decay,
            projection_refresh=_get(args, "adamw_lrsf_projection_refresh", None),
            orthogonal_refresh=_get(args, "adamw_lrsf_orthogonal_refresh", None),
        )
    if name == "AdamW-LR-EMA":
        return AdamWLowRankGradientEMA(
            parameters,
            lr=lr,
            rank=int(_get(args, "adamw_lr_ema_rank", _get(args, "rank", 8))),
            ema_beta=float(_get(args, "adamw_lr_ema_beta", 0.9)),
            seed=int(_get(args, "adamw_lr_ema_seed", _get(args, "seed", 0))),
            weight_decay=weight_decay,
            projection_scale=str(
                _get(args, "adamw_lr_ema_projection_scale", "norm")
            ),
        )
    if name == "AdamW-LR-EMA-Conf":
        return AdamWLowRankGradientEMAConfidence(
            parameters,
            lr=lr,
            rank=int(_get(args, "adamw_lr_ema_rank", _get(args, "rank", 8))),
            ema_beta=float(_get(args, "adamw_lr_ema_beta", 0.9)),
            confidence_beta=float(
                _get(args, "adamw_lr_ema_confidence_beta", 0.99)
            ),
            confidence_alpha=float(
                _get(args, "adamw_lr_ema_confidence_alpha", 1e-3)
            ),
            seed=int(_get(args, "adamw_lr_ema_seed", _get(args, "seed", 0))),
            weight_decay=weight_decay,
            projection_scale=str(
                _get(args, "adamw_lr_ema_projection_scale", "norm")
            ),
            eps=float(_get(args, "adamw_lr_ema_eps", 1e-8)),
        )
    if name == "AdamW-LR-EMA-Conf-LRSF":
        return AdamWLRSEMAConfLRSF(
            parameters,
            lr=lr,
            rank=int(_get(args, "adamw_lrsf_rank", _get(args, "rank", 8))),
            sf_beta1=float(_get(args, "adamw_lrsf_beta1", 0.9)),
            beta2=float(_get(args, "adamw_lrsf_beta2", 0.999)),
            warmup_steps=int(_get(args, "adamw_lrsf_warmup_steps", 0)),
            r=float(_get(args, "adamw_lrsf_r", 0.0)),
            weight_lr_power=float(
                _get(args, "adamw_lrsf_weight_lr_power", 2.0)
            ),
            ema_beta=float(_get(args, "adamw_lr_ema_beta", 0.9)),
            confidence_beta=float(
                _get(args, "adamw_lr_ema_confidence_beta", 0.99)
            ),
            confidence_alpha=float(
                _get(args, "adamw_lr_ema_confidence_alpha", 1e-3)
            ),
            seed=int(_get(args, "adamw_lrsf_seed", _get(args, "seed", 0))),
            eps=float(_get(args, "adamw_lrsf_eps", 1e-8)),
            weight_decay=weight_decay,
            projection_refresh=_get(args, "adamw_lrsf_projection_refresh", None),
            orthogonal_refresh=_get(args, "adamw_lrsf_orthogonal_refresh", None),
            backend=str(_get(args, "adamw_sf_backend", "auto")),
        )
    if name == "AdamW-SF-LR":
        return AdamWSFLowRankPreconditioner(
            parameters,
            lr=lr,
            rank=int(_get(args, "adamw_sf_lr_rank", _get(args, "rank", 4))),
            sf_beta1=float(_get(args, "adamw_lrsf_beta1", 0.9)),
            beta2=float(_get(args, "adamw_lrsf_beta2", 0.999)),
            warmup_steps=int(_get(args, "adamw_lrsf_warmup_steps", 0)),
            r=float(_get(args, "adamw_lrsf_r", 0.0)),
            weight_lr_power=float(
                _get(args, "adamw_lrsf_weight_lr_power", 2.0)
            ),
            seed=int(_get(args, "adamw_sf_lr_seed", _get(args, "adamw_lrsf_seed", 0))),
            eps=float(_get(args, "adamw_lrsf_eps", 1e-8)),
            weight_decay=weight_decay,
            backend=_get(args, "adamw_sf_backend", "auto"),
        )
    if name == "AdamW-LRSF-LR":
        return AdamWLRSLowRankPreconditioner(
            parameters,
            lr=lr,
            rank=int(_get(args, "adamw_lrsf_rank", _get(args, "rank", 4))),
            sf_beta1=float(_get(args, "adamw_lrsf_beta1", 0.9)),
            beta2=float(_get(args, "adamw_lrsf_beta2", 0.999)),
            warmup_steps=int(_get(args, "adamw_lrsf_warmup_steps", 0)),
            r=float(_get(args, "adamw_lrsf_r", 0.0)),
            weight_lr_power=float(
                _get(args, "adamw_lrsf_weight_lr_power", 2.0)
            ),
            seed=int(_get(args, "adamw_lrsf_seed", 0)),
            eps=float(_get(args, "adamw_lrsf_eps", 1e-8)),
            weight_decay=weight_decay,
            projection_refresh=_get(args, "adamw_lrsf_projection_refresh", None),
            projection_refresh_state=_get(
                args, "adamw_lrsf_projection_refresh_state", "transport"
            ),
            backend=_get(args, "adamw_sf_backend", "auto"),
        )
    if name == "CAME":
        return CAME(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999, 0.9999),
            eps=(1e-30, 1e-16),
        )
    if name == "CAME-SF":
        return CAMESF(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            sf_beta1=float(_get(args, "came_lrsf_beta1", 0.9)),
            warmup_steps=int(_get(args, "came_lrsf_warmup_steps", 0)),
            r=float(_get(args, "came_lrsf_r", 0.0)),
            weight_lr_power=float(_get(args, "came_lrsf_weight_lr_power", 2.0)),
            betas=(0.9, 0.999, 0.9999),
            eps=(1e-30, 1e-16),
            clip_threshold=1.0,
        )
    if name == "CAME-LRSF":
        return CAMELRSF(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            rank=int(_get(args, "came_lrsf_rank", 4)),
            sf_beta1=float(_get(args, "came_lrsf_beta1", 0.9)),
            warmup_steps=int(_get(args, "came_lrsf_warmup_steps", 0)),
            r=float(_get(args, "came_lrsf_r", 0.0)),
            weight_lr_power=float(_get(args, "came_lrsf_weight_lr_power", 2.0)),
            seed=int(_get(args, "came_lrsf_seed", 0)),
            betas=(0.9, 0.999, 0.9999),
            eps=(1e-30, 1e-16),
            clip_threshold=1.0,
            projection_refresh=_lrsf_refresh_policy(args),
            orthogonal_refresh=_lrsf_orthogonal_refresh_policy(args),
        )
    if name == "CAME-AutoSchedule":
        return CAMEAutoSchedule(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999, 0.9999),
            eps=(1e-30, 1e-16),
            **auto_kwargs,
        )
    if name == "RAdamSF":
        return RAdamScheduleFree(
            parameters, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)  # pyright: ignore[reportArgumentType]
        )
    if name == "AdamWSF":
        return AdamWScheduleFree(
            parameters, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)  # pyright: ignore[reportArgumentType]
        )

    apollo_rank = _get(args, "apollo_rank", 8)
    apollo_scale = _get(args, "apollo_scale", 1.0)
    apollo_fallback = _apollo_fallback_policy(args)
    apollo_projection_refresh = _get(args, "apollo_projection_refresh", None)
    if apollo_projection_refresh is None:
        refresh_mode = _get(args, "apollo_projection_refresh_mode", "hard")
        apollo_projection_refresh = {
            "mode": refresh_mode,
            "interval": _get(args, "apollo_update_proj_gap", 200),
            "window": (
                _get(args, "apollo_projection_refresh_window", 0)
                if refresh_mode == "smooth" else 0
            ),
            "mix": _get(args, "apollo_projection_refresh_mix", "smoothstep"),
        }
    apollo_kwargs: dict[str, Any] = dict(
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        rank=apollo_rank,
        scale=apollo_scale,
        update_proj_gap=_get(args, "apollo_update_proj_gap", 200),
        scale_front=_get(args, "apollo_scale_front", False),
        norm_growth_limiter=not _get(args, "apollo_disable_norm_growth_limiter", True),
        norm_growth_rate=_get(args, "apollo_norm_growth_rate", 1.01),
        projection_refresh_state=_get(
            args, "apollo_projection_refresh_state", "reset"
        ),
        projection_refresh=apollo_projection_refresh,
        orthogonal_refresh=_apollo_orthogonal_refresh_policy(args),
        update_norm_variance_cap=_get(
            args, "apollo_update_norm_variance_cap", None
        ),
    )
    if name == "APOLLO":
        return APOLLO(parameters, **apollo_kwargs, fallback=apollo_fallback)
    if name == "APOLLO-Conf":
        return APOLLOConfidence(
            parameters,
            **apollo_kwargs,
            fallback=apollo_fallback,
            confidence_beta=float(
                _get(args, "apollo_confidence_beta", _get(
                    args, "adamw_lr_ema_confidence_beta", 0.99,
                ))
            ),
            confidence_alpha=float(
                _get(args, "apollo_confidence_alpha", _get(
                    args, "adamw_lr_ema_confidence_alpha", 1e-3,
                ))
            ),
        )
    if name in {
        "APOLLO-SF",
        "APOLLO-SF-LRSF",
        "APOLLO-SF-INT8-Z",
        "APOLLO-SF-INT8-Delta",
        "APOLLO-SF-INT4-Z",
        "APOLLO-SF-INT4-Delta",
    }:
        storage = {
            "APOLLO-SF": "bf16_z",
            "APOLLO-SF-LRSF": "low_rank_delta",
            "APOLLO-SF-INT8-Z": "blockwise_int8_z",
            "APOLLO-SF-INT8-Delta": "blockwise_int8_delta",
            "APOLLO-SF-INT4-Z": "blockwise_int4_z",
            "APOLLO-SF-INT4-Delta": "blockwise_int4_delta",
        }[name]
        return APOLLOScheduleFree(
            parameters,
            lr=lr,
            rank=int(_get(args, "apollo_sf_rank", _get(args, "rank", 8))),
            sf_beta1=float(_get(args, "apollo_sf_beta1", 0.9)),
            warmup_steps=int(_get(args, "apollo_sf_warmup_steps", 0)),
            sf_r=float(_get(args, "apollo_sf_r", 0.0)),
            sf_weight_lr_power=float(
                _get(args, "apollo_sf_weight_lr_power", 2.0)
            ),
            sf_state_storage=storage,
            sf_quant_block_size=int(
                _get(args, "apollo_sf_quant_block_size", 256)
            ),
            seed=int(_get(args, "apollo_sf_seed", _get(args, "seed", 0))),
            scale=float(_get(args, "apollo_sf_scale", _get(args, "apollo_scale", 1.0))),
            betas=(
                float(_get(args, "apollo_sf_beta1", 0.9)),
                float(_get(args, "apollo_sf_beta2", 0.999)),
            ),
            eps=float(_get(args, "apollo_sf_eps", 1e-8)),
            weight_decay=weight_decay,
            update_proj_gap=int(_get(args, "apollo_sf_update_proj_gap", 200)),
            norm_growth_limiter=bool(
                _get(args, "apollo_sf_norm_growth_limiter", False)
            ),
            projection_refresh=_get(args, "apollo_sf_projection_refresh", None),
        )
    if name == "APOLLO-AutoSchedule":
        return APOLLOAutoSchedule(
            parameters, **apollo_kwargs, fallback=apollo_fallback, **auto_kwargs
        )
    if name == "APOLLO-Mini":
        apollo_kwargs["rank"] = 1
        apollo_kwargs["scale"] = _get(args, "apollo_mini_scale", 1.0)
        return APOLLOMini(parameters, **apollo_kwargs, fallback=apollo_fallback)
    if name == "APOLLO-AdamW":
        return APOLLOADAMW(parameters, **apollo_kwargs, fallback=apollo_fallback)
    if name == "APOLLO-AdamW-AutoSchedule":
        return APOLLOADAMWAutoSchedule(
            parameters, **apollo_kwargs, fallback=apollo_fallback, **auto_kwargs
        )
    if name == "APOLLO-Lion":
        apollo_kwargs.pop("betas")
        apollo_kwargs.pop("eps")
        return APOLLOLion(parameters, **apollo_kwargs, fallback=apollo_fallback)
    if name == "APOLLO-CAME":
        return APOLLOCAME(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999, 0.9999),
            eps=(1e-30, 1e-16),
            clip_threshold=1.0,
            rank=apollo_rank,
            scale=apollo_scale,
            update_proj_gap=apollo_kwargs["update_proj_gap"],
            scale_front=apollo_kwargs["scale_front"],
            norm_growth_limiter=apollo_kwargs["norm_growth_limiter"],
            norm_growth_rate=apollo_kwargs["norm_growth_rate"],
            projection_refresh_state=apollo_kwargs["projection_refresh_state"],
            projection_refresh=apollo_kwargs["projection_refresh"],
            orthogonal_refresh=apollo_kwargs["orthogonal_refresh"],
            update_norm_variance_cap=apollo_kwargs[
                "update_norm_variance_cap"
            ],
            fallback=apollo_fallback,
            came_backend=cast(str, _get(args, "apollo_came_backend", "torch")),
        )
    if name == "APOLLO-CAME-LRSF":
        # APOLLO-CAME-LRSF's separate Schedule-Free delta path does not yet
        # implement the AdamW-SF fallback state machine. Keep its prior
        # CAME/auto policy while standard APOLLO variants use the new defaults.
        lrsf_fallback = APOLLOFallbackPolicy(
            one_dimensional="came",
            small_matrix="auto",
        )
        return APOLLOCAMELRSF(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999, 0.9999),
            eps=(1e-30, 1e-16),
            clip_threshold=1.0,
            rank=apollo_rank,
            lrsf_rank=int(_get(args, "came_lrsf_rank", 4)),
            sf_beta1=float(_get(args, "came_lrsf_beta1", 0.9)),
            warmup_steps=int(_get(args, "came_lrsf_warmup_steps", 0)),
            r=float(_get(args, "came_lrsf_r", 0.0)),
            weight_lr_power=float(_get(args, "came_lrsf_weight_lr_power", 2.0)),
            seed=int(_get(args, "came_lrsf_seed", 0)),
            scale=apollo_scale,
            update_proj_gap=apollo_kwargs["update_proj_gap"],
            scale_front=apollo_kwargs["scale_front"],
            norm_growth_limiter=apollo_kwargs["norm_growth_limiter"],
            norm_growth_rate=apollo_kwargs["norm_growth_rate"],
            fallback=lrsf_fallback,
            came_backend=cast(str, _get(args, "apollo_came_backend", "torch")),
            delta_refresh=_lrsf_refresh_policy(args),
            orthogonal_refresh=_lrsf_orthogonal_refresh_policy(args),
            apollo_orthogonal_refresh=_apollo_orthogonal_refresh_policy(args),
        )
    if name == "APOLLO-CAME-AutoSchedule":
        return APOLLOCAMEAutoSchedule(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999, 0.9999),
            eps=(1e-30, 1e-16),
            clip_threshold=1.0,
            rank=apollo_rank,
            scale=apollo_scale,
            update_proj_gap=apollo_kwargs["update_proj_gap"],
            scale_front=apollo_kwargs["scale_front"],
            norm_growth_limiter=apollo_kwargs["norm_growth_limiter"],
            norm_growth_rate=apollo_kwargs["norm_growth_rate"],
            projection_refresh_state=apollo_kwargs["projection_refresh_state"],
            projection_refresh=apollo_kwargs["projection_refresh"],
            orthogonal_refresh=apollo_kwargs["orthogonal_refresh"],
            fallback=apollo_fallback,
            came_backend=_get(args, "apollo_came_backend", "torch"),
            **auto_kwargs,
        )
    if name in {"Muon", "SOAP", "NorMuon", "AdaMuon"}:
        if role != "linear":
            raise ValueError(f"{name} is only supported as the Linear optimizer")
        if _get(args, "linear_lr", None) is None:
            lr = 1e-3 if name == "SOAP" else 0.02
        optimizer_class = {
            "Muon": Muon,
            "SOAP": SOAP,
            "NorMuon": NorMuon,
            "AdaMuon": AdaMuon,
        }[name]
        return optimizer_class(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            **(
                {"momentum": _get(args, "linear_muon_momentum", 0.95)}
                if name != "SOAP"
                else {}
            ),
        )
    if name == "Lion":
        return Lion(
            parameters, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99)
        )
    if name == "RotAPOLLO":
        return RotAPOLLO(
            parameters,
            **apollo_kwargs,
            fallback=apollo_fallback,
            rotation_frequency=cast(int, _get(args, "rot_apollo_frequency", 100)),
            rotation_rate=_get(args, "rot_apollo_rate", 0.02),
            exploration_ratio=_get(args, "rot_apollo_exploration_ratio", 0.2),
        )
    if name == "DualRotAPOLLO":
        return DualRotAPOLLO(
            parameters,
            **apollo_kwargs,
            fallback=apollo_fallback,
            rotation_frequency=cast(int, _get(args, "dual_rot_apollo_frequency", 100)),
            rotation_rate=_get(args, "dual_rot_apollo_rate", 0.02),
            exploration_ratio=_get(args, "dual_rot_apollo_exploration_ratio", 0.2),
            roughness_beta=_get(args, "dual_rot_apollo_roughness_beta", 0.95),
            branch_temperature=_get(args, "dual_rot_apollo_branch_temperature", 5.0),
        )
    raise ValueError(f"Unknown optimizer: {name}")


__all__ = [
    "CORE_OPTIMIZER_CHOICES",
    "EXPERIMENTAL_OPTIMIZER_CHOICES",
    "IMAGE_GEN_OPTIMIZER_CHOICES",
    "LEGACY_OPTIMIZER_CHOICES",
    "add_optimizer_argument",
    "add_optimizer_override_argument",
    "build_optimizer",
    "is_schedule_free_optimizer",
    "normalize_optimizer_name",
]
