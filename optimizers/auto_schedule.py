"""Low-memory optimizer-specific learning-rate controllers.

The optimizers compute statistics per parameter tensor while stepping, but the
controller state is kept per optimizer parameter group.  No parameter-sized
controller buffers are allocated here.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


class AutoScheduleMixin:
    """Shared bounded trust-ratio controller for custom optimizers."""

    def _enable_auto_schedule(
        self,
        *,
        kind: str,
        target_update_ratio: float = 1e-3,
        ema_beta: float = 0.99,
        trust_alpha: float = 0.1,
        min_factor: float = 0.5,
        max_factor: float = 4.0,
        max_increase: float = 1.05,
        max_decrease: float = 0.95,
        confidence_floor: float = 0.25,
        stability_gain: float = 2.0,
        limiter_gain: float = 2.0,
        cooldown_steps: int = 4,
        warmup_steps: int = 200,
        fast_beta: float = 0.9,
        slow_beta: float = 0.999,
        gain: float = 0.25,
        controller_rate: float = 0.05,
    ) -> None:
        if kind not in {"adamw", "came", "apollo", "apollo-came"}:
            raise ValueError(f"unknown auto-schedule kind: {kind}")
        if target_update_ratio <= 0.0:
            raise ValueError("target_update_ratio must be positive")
        if not 0.0 < ema_beta < 1.0:
            raise ValueError("ema_beta must be in (0, 1)")
        if trust_alpha <= 0.0:
            raise ValueError("trust_alpha must be positive")
        if not 0.0 < min_factor <= max_factor:
            raise ValueError("invalid auto-schedule factor range")
        if max_increase < 1.0 or max_decrease <= 0.0 or max_decrease > 1.0:
            raise ValueError("invalid per-step factor limits")
        if not 0.0 <= confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be in [0, 1]")
        if stability_gain < 0.0 or limiter_gain < 0.0:
            raise ValueError("stability gains must be non-negative")
        if cooldown_steps < 0:
            raise ValueError("cooldown_steps must be non-negative")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0.0 < fast_beta < 1.0 or not 0.0 < slow_beta < 1.0:
            raise ValueError("fast_beta and slow_beta must be in (0, 1)")
        if fast_beta >= slow_beta:
            raise ValueError("fast_beta must be smaller than slow_beta")
        if gain < 0.0:
            raise ValueError("gain must be non-negative")
        if not 0.0 < controller_rate <= 1.0:
            raise ValueError("controller_rate must be in (0, 1]")

        self.auto_schedule_kind = kind
        self._auto_schedule_options = {
            "target_update_ratio": float(target_update_ratio),
            "ema_beta": float(ema_beta),
            "trust_alpha": float(trust_alpha),
            "min_factor": float(min_factor),
            "max_factor": float(max_factor),
            "max_increase": float(max_increase),
            "max_decrease": float(max_decrease),
            "confidence_floor": float(confidence_floor),
            "stability_gain": float(stability_gain),
            "limiter_gain": float(limiter_gain),
            "cooldown_steps": int(cooldown_steps),
            "warmup_steps": int(warmup_steps),
            "fast_beta": float(fast_beta),
            "slow_beta": float(slow_beta),
            "gain": float(gain),
            "controller_rate": float(controller_rate),
        }
        for group in self.param_groups:
            self._initialize_auto_schedule_group(group)

    def _initialize_auto_schedule_group(self, group: dict) -> None:
        options = self._auto_schedule_options
        for key, value in self._auto_schedule_group_defaults(group, options).items():
            group.setdefault(key, value)

    @staticmethod
    def _auto_schedule_group_defaults(group: dict, options: dict) -> dict:
        """Return missing controller values without changing the parameter group."""
        return {
            "_auto_schedule_base_lr": float(group["lr"]),
            "_auto_schedule_multiplier": 1.0,
            "_auto_schedule_ema_ratio": 0.0,
            "_auto_schedule_has_ratio": False,
            "_auto_schedule_fast_ema": 0.0,
            "_auto_schedule_slow_ema": 0.0,
            "_auto_schedule_has_multiscale_ema": False,
            "_auto_schedule_ema_confidence": 1.0,
            "_auto_schedule_has_confidence": False,
            "_auto_schedule_ema_scale_delta": 0.0,
            "_auto_schedule_has_scale": False,
            "_auto_schedule_previous_scale_rms": 0.0,
            "_auto_schedule_ema_limiter": 0.0,
            "_auto_schedule_cooldown": 0,
            "_auto_schedule_step": 0,
            # The LR that was actually used by the most recent optimizer step.
            # Keep this separate from ``group["lr"]`` because the controller
            # updates its multiplier after applying the current step.
            "_auto_schedule_applied_lr": None,
            "_auto_schedule_target_update_ratio": options["target_update_ratio"],
        }

    def _auto_schedule_group_view(self, group: dict) -> dict:
        """Return a preview-only group copy with legacy defaults filled in."""
        view = dict(group)
        options = self._auto_schedule_options
        for key, value in self._auto_schedule_group_defaults(group, options).items():
            view.setdefault(key, value)
        return view

    def _auto_schedule_is_warmup(self, group: dict) -> bool:
        return int(group["_auto_schedule_step"]) < self._auto_schedule_options["warmup_steps"]

    def _auto_schedule_begin_group(self, group: dict) -> float:
        """Apply the persistent group multiplier to the scheduled LR."""
        if not getattr(self, "auto_schedule_kind", None):
            return float(group["lr"])
        self._initialize_auto_schedule_group(group)
        scheduled_lr = float(
            group.get(
                "_external_scheduled_lr",
                group.get("_auto_schedule_base_lr", group["lr"]),
            )
        )
        if self._auto_schedule_is_warmup(group):
            effective_lr = scheduled_lr
        else:
            effective_lr = self._auto_schedule_effective_lr(group, scheduled_lr)
        group["lr"] = effective_lr
        group["_auto_schedule_applied_lr"] = effective_lr
        return effective_lr

    def _auto_schedule_preview_lr(self, group: dict) -> float:
        """Return the effective LR for the next step without changing state."""
        if not getattr(self, "auto_schedule_kind", None):
            return float(group.get("_external_scheduled_lr", group["lr"]))
        group = self._auto_schedule_group_view(group)
        scheduled_lr = float(
            group.get(
                "_external_scheduled_lr",
                group.get("_auto_schedule_base_lr", group["lr"]),
            )
        )
        if self._auto_schedule_is_warmup(group):
            return scheduled_lr
        return self._auto_schedule_effective_lr(group, scheduled_lr)

    @staticmethod
    def _auto_schedule_new_stats(parameter: torch.Tensor) -> dict:
        scalar = torch.zeros((), device=parameter.device, dtype=torch.float32)
        return {
            "update_norm_sq": scalar.clone(),
            "parameter_norm_sq": scalar.clone(),
            "noise_norm_sq": scalar.clone(),
            "moment_norm_sq": scalar.clone(),
            "scale_norm_sq": scalar.clone(),
            "scale_count": 0,
            "scale_seen": False,
            "limiter_active": scalar.clone(),
            "limiter_count": 0,
            "projection_refresh": False,
        }

    @staticmethod
    def _auto_schedule_add_norm(
        stats: dict,
        key: str,
        value: torch.Tensor,
    ) -> None:
        stats[key].add_(value.float().square().sum())

    def _auto_schedule_finish_group(
        self,
        group: dict,
        stats: Optional[dict],
    ) -> None:
        if not getattr(self, "auto_schedule_kind", None) or stats is None:
            return
        options = self._auto_schedule_options
        group["_auto_schedule_step"] = int(group["_auto_schedule_step"]) + 1

        (
            parameter_norm_sq,
            update_norm_sq,
            noise_norm_sq,
            moment_norm_sq,
            scale_norm_sq,
            limiter_active,
        ) = torch.stack(
            (
                stats["parameter_norm_sq"],
                stats["update_norm_sq"],
                stats["noise_norm_sq"],
                stats["moment_norm_sq"],
                stats["scale_norm_sq"],
                stats["limiter_active"],
            )
        ).detach().cpu().tolist()
        if parameter_norm_sq > 0.0 and update_norm_sq > 0.0:
            ratio = math.sqrt(update_norm_sq / parameter_norm_sq)
            if group["_auto_schedule_has_multiscale_ema"]:
                fast_beta = options["fast_beta"]
                slow_beta = options["slow_beta"]
                group["_auto_schedule_fast_ema"] = (
                    fast_beta * float(group["_auto_schedule_fast_ema"])
                    + (1.0 - fast_beta) * ratio
                )
                group["_auto_schedule_slow_ema"] = (
                    slow_beta * float(group["_auto_schedule_slow_ema"])
                    + (1.0 - slow_beta) * ratio
                )
            else:
                group["_auto_schedule_fast_ema"] = ratio
                group["_auto_schedule_slow_ema"] = ratio
                group["_auto_schedule_has_multiscale_ema"] = True
            group["_auto_schedule_ema_ratio"] = group["_auto_schedule_slow_ema"]
            group["_auto_schedule_has_ratio"] = True

            if (
                int(group["_auto_schedule_step"]) > 1
                and not self._auto_schedule_is_warmup(group)
            ):
                ema_ratio = max(float(group["_auto_schedule_ema_ratio"]), 1e-30)
                error = options["gain"] * (
                    math.log(options["target_update_ratio"]) - math.log(
                        ema_ratio
                    )
                )
                target_correction = math.exp(options["trust_alpha"] * error)
                correction = 1.0 + options["controller_rate"] * (
                    target_correction - 1.0
                )
                correction = min(
                    max(correction, options["max_decrease"]),
                    options["max_increase"],
                )
                multiplier = float(group["_auto_schedule_multiplier"]) * correction
                group["_auto_schedule_multiplier"] = min(
                    max(multiplier, options["min_factor"]),
                    options["max_factor"],
                )

        if moment_norm_sq > 0.0:
            confidence = 1.0 / (
                1.0 + math.sqrt(noise_norm_sq / moment_norm_sq)
            )
            confidence = min(max(confidence, 0.0), 1.0)
            if group["_auto_schedule_has_confidence"]:
                beta = options["ema_beta"]
                previous = float(group["_auto_schedule_ema_confidence"])
                group["_auto_schedule_ema_confidence"] = (
                    beta * previous + (1.0 - beta) * confidence
                )
            else:
                group["_auto_schedule_ema_confidence"] = confidence
                group["_auto_schedule_has_confidence"] = True

        if stats["scale_seen"]:
            scale_count = max(int(stats["scale_count"]), 1)
            scale_rms = math.sqrt(
                max(scale_norm_sq, 0.0) / scale_count
            )
            if group["_auto_schedule_has_scale"]:
                previous = max(
                    float(group["_auto_schedule_previous_scale_rms"]),
                    1e-30,
                )
                delta = abs(math.log(max(scale_rms, 1e-30)) - math.log(previous))
                beta = options["ema_beta"]
                group["_auto_schedule_ema_scale_delta"] = (
                    beta * float(group["_auto_schedule_ema_scale_delta"])
                    + (1.0 - beta) * delta
                )
            group["_auto_schedule_previous_scale_rms"] = scale_rms
            group["_auto_schedule_has_scale"] = True

        if int(stats["limiter_count"]) > 0:
            limiter_rate = limiter_active / stats["limiter_count"]
            beta = options["ema_beta"]
            group["_auto_schedule_ema_limiter"] = (
                beta * float(group["_auto_schedule_ema_limiter"])
                + (1.0 - beta) * limiter_rate
            )

        if stats["projection_refresh"]:
            group["_auto_schedule_cooldown"] = options["cooldown_steps"]
        elif int(group["_auto_schedule_cooldown"]) > 0:
            group["_auto_schedule_cooldown"] -= 1

    def _auto_schedule_cap(self, group: dict) -> float:
        """Return an optimizer-specific LR cap in units of scheduled_lr."""
        kind = getattr(self, "auto_schedule_kind", None)
        if kind is None:
            return 1.0
        if self._auto_schedule_is_warmup(group):
            return 1.0
        options = self._auto_schedule_options
        cap = 1.0
        if kind in {"came", "apollo-came"} and group["_auto_schedule_has_confidence"]:
            confidence_cap = (
                options["confidence_floor"]
                + (1.0 - options["confidence_floor"])
                * float(group["_auto_schedule_ema_confidence"])
            )
            cap = min(cap, confidence_cap)
        if kind in {"apollo", "apollo-came"}:
            if group["_auto_schedule_cooldown"] <= 0:
                cap = min(
                    cap,
                    math.exp(
                        -options["stability_gain"]
                        * float(group["_auto_schedule_ema_scale_delta"])
                    ),
                )
            cap = min(
                cap,
                math.exp(
                    -options["limiter_gain"]
                    * float(group["_auto_schedule_ema_limiter"])
                ),
            )
        return min(max(cap, options["min_factor"]), 1.0)

    def _auto_schedule_status(self, group: dict) -> dict | None:
        """Return scalar controller diagnostics for logging."""
        if not getattr(self, "auto_schedule_kind", None):
            return None
        group = self._auto_schedule_group_view(group)
        return {
            "multiplier": float(group["_auto_schedule_multiplier"]),
            "cap": float(self._auto_schedule_cap(group)),
            "warmup": self._auto_schedule_is_warmup(group),
        }

    def _auto_schedule_effective_lr(self, group: dict, scheduled_lr: float) -> float:
        cap = self._auto_schedule_cap(group)
        effective_lr = scheduled_lr * min(
            float(group["_auto_schedule_multiplier"]), cap
        )
        return min(
            max(effective_lr, scheduled_lr * self._auto_schedule_options["min_factor"]),
            scheduled_lr * self._auto_schedule_options["max_factor"],
        )
