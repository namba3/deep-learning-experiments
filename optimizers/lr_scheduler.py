"""External learning-rate schedules shared by the training scripts."""

import argparse
import math


LR_SCHEDULER_CHOICES = (
    "auto",
    "constant",
    "linear",
    "cosine",
    "cosine-restarts",
    "polynomial",
    "inverse-sqrt",
    "step",
    "multistep",
    "exponential",
)


def parse_lr_milestones(value: str) -> tuple[int, ...]:
    """Parse comma-separated positive optimizer-step milestones."""
    try:
        milestones = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "LR milestones must be comma-separated integers"
        ) from error
    if not milestones or any(milestone <= 0 for milestone in milestones):
        raise argparse.ArgumentTypeError(
            "LR milestones must contain positive integers"
        )
    return milestones


def add_lr_scheduler_arguments(
    parser: argparse.ArgumentParser,
    *,
    default: str = "cosine",
    include_force_scheduler: bool = False,
) -> None:
    """Add the common step-based LR scheduler CLI contract."""
    parser.add_argument(
        "--lr-scheduler",
        choices=LR_SCHEDULER_CHOICES,
        default=default,
        help=(
            "External LR schedule: auto, constant, linear, cosine "
            "(cosine annealing), cosine-restarts, polynomial, inverse-sqrt, "
            "step, multistep, or exponential."
        ),
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=None,
        help="Linear warmup steps before the selected LR schedule. Default: 0.",
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=None,
        help=(
            "Warmup fraction of optimizer steps; cannot be combined with "
            "--warmup-steps. Default: 0."
        ),
    )
    parser.add_argument(
        "--min-lr-ratio", type=float, default=0.0,
        help="Final LR/base-LR floor for decaying schedules. Default: 0.",
    )
    parser.add_argument(
        "--lr-step-size", type=int, default=1000,
        help="Step interval for the step scheduler. Default: 1000.",
    )
    parser.add_argument(
        "--lr-gamma", type=float, default=0.1,
        help="Multiplicative factor for step/multistep/exponential. Default: 0.1.",
    )
    parser.add_argument(
        "--lr-milestones", type=parse_lr_milestones, default=(1000, 2000),
        metavar="STEP[,STEP...]",
        help="Optimizer steps for multistep decay. Default: 1000,2000.",
    )
    parser.add_argument(
        "--lr-num-cycles", type=int, default=1,
        help="Number of cycles for cosine-restarts. Default: 1.",
    )
    parser.add_argument(
        "--lr-power", type=float, default=1.0,
        help="Power for polynomial decay. Default: 1.0.",
    )
    if include_force_scheduler:
        parser.add_argument(
            "--force-scheduler",
            action="store_true",
            help="ScheduleFree互換optimizerでも外部schedulerを使用する",
        )


def resolve_lr_scheduler_name(name: str, optimizer_name: str | None = None) -> str:
    """Resolve the compatibility ``auto`` choice to a concrete schedule."""
    if name != "auto":
        return name
    # Preserve the historical behavior of the small image classifiers:
    # CAME used constant-with-warmup while the other optimizers used cosine.
    return "constant" if optimizer_name == "CAME" else "cosine"


def resolve_warmup_steps(
    total_steps: int,
    *,
    warmup_steps: int | None = 0,
    warmup_ratio: float | None = 0.0,
) -> int:
    """Resolve mutually exclusive absolute and fractional warmup settings."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup_steps = 0 if warmup_steps is None else warmup_steps
    warmup_ratio = 0.0 if warmup_ratio is None else warmup_ratio
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if warmup_steps and warmup_ratio:
        raise ValueError("use only one of warmup_steps and warmup_ratio")
    resolved = warmup_steps or (
        max(1, round(total_steps * warmup_ratio))
        if warmup_ratio > 0.0
        else 0
    )
    if resolved > total_steps:
        raise ValueError("warmup_steps must be smaller than total_steps")
    return resolved


def build_lr_scheduler(optimizer, args, total_steps: int):
    """Build a common scheduler from an argparse namespace."""
    warmup_steps = resolve_warmup_steps(
        total_steps,
        warmup_steps=getattr(args, "warmup_steps", 0),
        warmup_ratio=getattr(args, "warmup_ratio", 0.0),
    )
    return LearningRateSchedule(
        optimizer,
        resolve_lr_scheduler_name(
            getattr(args, "lr_scheduler", "cosine"),
            getattr(args, "optimizer", None),
        ),
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=getattr(args, "min_lr_ratio", 0.0),
        step_size=getattr(args, "lr_step_size", 1000),
        gamma=getattr(args, "lr_gamma", 0.1),
        milestones=getattr(args, "lr_milestones", (1000, 2000)),
        num_cycles=getattr(args, "lr_num_cycles", 1),
        power=getattr(args, "lr_power", 1.0),
    )


class LearningRateSchedule:
    """Apply a schedule to every optimizer parameter group.

    The schedule writes ``_external_scheduled_lr`` so optimizer-specific
    AutoSchedule controllers can apply their group multiplier without
    compounding it with the external schedule.
    """

    def __init__(
        self,
        optimizer,
        name: str,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        step_size: int = 1000,
        gamma: float = 0.1,
        milestones: tuple[int, ...] = (1000, 2000),
        num_cycles: int = 1,
        power: float = 1.0,
    ):
        if name not in LR_SCHEDULER_CHOICES[1:]:
            raise ValueError(f"unknown learning-rate schedule: {name}")
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be between 0 and 1")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if not milestones or any(milestone <= 0 for milestone in milestones):
            raise ValueError("milestones must contain positive integers")
        if num_cycles <= 0:
            raise ValueError("num_cycles must be positive")
        if power <= 0.0:
            raise ValueError("power must be positive")
        self.optimizer = optimizer
        self.name = name
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        self.milestones = tuple(sorted(int(milestone) for milestone in milestones))
        self.num_cycles = int(num_cycles)
        self.power = float(power)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_step = 0

    def state_dict(self):
        """Return scheduler state required for an exact resume."""
        return {
            "name": self.name,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "step_size": self.step_size,
            "gamma": self.gamma,
            "milestones": self.milestones,
            "num_cycles": self.num_cycles,
            "power": self.power,
            "base_lrs": self.base_lrs,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state):
        """Restore scheduler state and reapply the saved LR."""
        for key in (
            "name", "total_steps", "warmup_steps", "min_lr_ratio",
            "step_size", "gamma", "milestones", "num_cycles", "power",
        ):
            if key in state and getattr(self, key) != state[key]:
                raise ValueError(
                    f"LR scheduler configuration mismatch for {key}: "
                    f"current={getattr(self, key)!r}, saved={state[key]!r}"
                )
        if "base_lrs" in state:
            if len(state["base_lrs"]) != len(self.base_lrs):
                raise ValueError("LR scheduler parameter-group count mismatch")
            self.base_lrs = [float(value) for value in state["base_lrs"]]
        self.last_step = int(state.get("last_step", 0))
        self.step(self.last_step)

    def _decay_factor(self, step: int) -> float:
        if self.name == "constant":
            return 1.0
        decay_steps = max(self.total_steps - self.warmup_steps, 1)
        decay_step = max(step - self.warmup_steps, 0)
        progress = min(decay_step / decay_steps, 1.0)
        if self.name == "linear":
            return 1.0 - progress * (1.0 - self.min_lr_ratio)
        if self.name == "cosine":
            curve = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * curve
        if self.name == "cosine-restarts":
            cycle_progress = (progress * self.num_cycles) % 1.0
            curve = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * curve
        if self.name == "polynomial":
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * (
                (1.0 - progress) ** self.power
            )
        if self.name == "inverse-sqrt":
            reference_step = max(self.warmup_steps, 1)
            curve = math.sqrt(reference_step / max(step, reference_step))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * curve
        if self.name == "step":
            return max(self.min_lr_ratio, self.gamma ** (decay_step // self.step_size))
        if self.name == "multistep":
            decay_count = sum(decay_step >= milestone for milestone in self.milestones)
            return max(self.min_lr_ratio, self.gamma ** decay_count)
        if self.name == "exponential":
            return max(self.min_lr_ratio, self.gamma ** decay_step)
        raise AssertionError(f"unhandled learning-rate schedule: {self.name}")

    def step(self, step: int) -> float:
        """Set the LR for the given 1-based optimizer step."""
        step = max(int(step), 0)
        if self.warmup_steps > 0 and step < self.warmup_steps:
            factor = step / self.warmup_steps
        else:
            factor = self._decay_factor(step)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            scheduled_lr = base_lr * factor
            group["_external_scheduled_lr"] = scheduled_lr
            group["lr"] = scheduled_lr
        self.last_step = step
        return factor

    @property
    def current_lrs(self):
        return tuple(group["lr"] for group in self.optimizer.param_groups)

    @staticmethod
    def _scheduled_group_lr(group) -> float:
        return float(group.get("_external_scheduled_lr", group["lr"]))

    @staticmethod
    def _effective_group_lr(optimizer, group) -> float:
        preview = getattr(optimizer, "_auto_schedule_preview_lr", None)
        if preview is not None:
            applied = group.get("_auto_schedule_applied_lr")
            if applied is not None:
                return float(applied)
            return float(preview(group))
        return float(group.get("_external_scheduled_lr", group["lr"]))

    def _format_lrs(self, *, effective: bool) -> str:
        optimizers = getattr(self.optimizer, "optimizers", None)
        if optimizers is not None:
            parts = []
            for name, optimizer in optimizers.items():
                values = (
                    self._effective_group_lr(optimizer, group)
                    if effective
                    else self._scheduled_group_lr(group)
                    for group in optimizer.param_groups
                )
                parts.append(f"{name}=" + ",".join(f"{value:.3e}" for value in values))
            return " ".join(parts)
        values = (
            self._effective_group_lr(self.optimizer, group)
            if effective
            else self._scheduled_group_lr(group)
            for group in self.optimizer.param_groups
        )
        return "main=" + ",".join(f"{value:.3e}" for value in values)

    def format_scheduled_lrs(self) -> str:
        """Format the LR set by the external schedule."""
        return self._format_lrs(effective=False)

    def format_effective_lrs(self) -> str:
        """Format the last applied LR, or the next LR before the first step."""
        return self._format_lrs(effective=True)

    @staticmethod
    def _format_controller_group(optimizer, group) -> str | None:
        status = getattr(optimizer, "_auto_schedule_status", None)
        if status is None:
            return None
        state = status(group)
        if state is None:
            return None
        prefix = "warmup " if state["warmup"] else ""
        return (
            f"{prefix}m={state['multiplier']:.3f} "
            f"cap={state['cap']:.3f}"
        )

    def format_auto_schedule_state(self) -> str:
        """Format AutoSchedule multiplier/cap diagnostics, if enabled."""
        optimizers = getattr(self.optimizer, "optimizers", None)
        if optimizers is None:
            optimizers = {"main": self.optimizer}
        parts = []
        for name, optimizer in optimizers.items():
            values = [
                self._format_controller_group(optimizer, group)
                for group in optimizer.param_groups
            ]
            values = [value for value in values if value is not None]
            if values:
                parts.append(f"{name}=" + ",".join(values))
        return " ".join(parts)

    def format_current_lrs(self) -> str:
        # Compatibility alias.  New logs should use the explicit names above.
        return self.format_scheduled_lrs()
