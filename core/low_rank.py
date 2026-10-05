"""Low-rank trainable adapters for shared model layers.

The adapters here operate on ``torch.nn.Linear`` weights.  The base layer is
kept frozen and the adapter adds a trainable low-rank residual.  This module
contains the parameterization only; optimizer and training-loop policy remain
in the individual experiment scripts.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class _AdapterLinear(nn.Module):
    """Common implementation for Linear-based low-rank adapters."""

    adapter_type = "adapter"

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("adapter base must be an nn.Linear")
        if rank <= 0:
            raise ValueError("adapter rank must be positive")
        if alpha is None:
            alpha = float(rank)
        if alpha <= 0:
            raise ValueError("adapter alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("adapter dropout must be in [0, 1)")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.merged = False
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def _input(self, input: torch.Tensor) -> torch.Tensor:
        return self.dropout(input.to(dtype=self._adapter_dtype()))

    def _adapter_dtype(self) -> torch.dtype:
        raise NotImplementedError

    def _delta_weight(self) -> torch.Tensor:
        raise NotImplementedError

    def _effective_weight(self) -> torch.Tensor:
        return self.base.weight.to(dtype=self._adapter_dtype()) + self._delta_weight()

    def _forward_unmerged(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base(input)
        return self._forward_unmerged(input)

    @torch.no_grad()
    def merge(self) -> None:
        """Materialize the adapter update into the base weight."""
        if self.merged:
            return
        delta = self._effective_weight().to(dtype=self.base.weight.dtype)
        delta = delta - self.base.weight
        self.base.weight.add_(delta)
        self._merged_delta = delta.detach()
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Remove a previously materialized adapter update."""
        if not self.merged:
            return
        self.base.weight.sub_(self._merged_delta.to(dtype=self.base.weight.dtype))
        self.merged = False
        del self._merged_delta

    def extra_repr(self) -> str:
        return (
            f"in_features={self.base.in_features}, "
            f"out_features={self.base.out_features}, "
            f"rank={self.rank}, alpha={self.alpha:g}"
        )


class LoRALinear(_AdapterLinear):
    """A Linear layer with a frozen base weight and LoRA factors."""

    adapter_type = "lora"

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(base, rank, alpha, dropout)
        self.lora_A = nn.Parameter(
            base.weight.new_empty(self.rank, base.in_features),
        )
        self.lora_B = nn.Parameter(
            base.weight.new_zeros(base.out_features, self.rank),
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def _delta_weight(self) -> torch.Tensor:
        return self.scaling * (self.lora_B @ self.lora_A)

    def _adapter_dtype(self) -> torch.dtype:
        return self.lora_A.dtype

    def _forward_unmerged(self, input: torch.Tensor) -> torch.Tensor:
        result = self.base(input)
        adapter_input = self._input(input)
        update = F.linear(adapter_input, self.lora_A)
        update = F.linear(update, self.lora_B)
        return result + update.to(dtype=result.dtype) * self.scaling


class DoRALinear(_AdapterLinear):
    """Weight-decomposed low-rank adaptation for a Linear layer.

    The trainable magnitude is stored per output row.  The low-rank update
    changes the direction and the magnitude parameter independently:

        W' = m[:, None] * (W + delta W) / ||W + delta W||_2
    """

    adapter_type = "dora"

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float | None = None,
        dropout: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(base, rank, alpha, dropout)
        if eps <= 0:
            raise ValueError("DoRA eps must be positive")
        self.eps = float(eps)
        self.lora_A = nn.Parameter(
            base.weight.new_empty(self.rank, base.in_features),
        )
        self.lora_B = nn.Parameter(
            base.weight.new_zeros(base.out_features, self.rank),
        )
        self.magnitude = nn.Parameter(
            base.weight.detach().norm(dim=1).clone(),
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def _delta_weight(self) -> torch.Tensor:
        return self.scaling * (self.lora_B @ self.lora_A)

    def _adapter_dtype(self) -> torch.dtype:
        return self.lora_A.dtype

    def _effective_weight(self) -> torch.Tensor:
        direction = super()._effective_weight()
        norm = direction.norm(dim=1, keepdim=True).clamp_min(self.eps)
        return self.magnitude.to(dtype=direction.dtype)[:, None] * direction / norm

    def _forward_unmerged(self, input: torch.Tensor) -> torch.Tensor:
        weight = self._effective_weight()
        bias = None if self.base.bias is None else self.base.bias.to(weight.dtype)
        return F.linear(self._input(input), weight, bias)


class LoHALinear(_AdapterLinear):
    """Hadamard-product low-rank adaptation for a Linear layer."""

    adapter_type = "loha"

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(base, rank, alpha, dropout)
        self.lora_A1 = nn.Parameter(
            base.weight.new_empty(self.rank, base.in_features),
        )
        self.lora_B1 = nn.Parameter(
            base.weight.new_empty(base.out_features, self.rank),
        )
        self.lora_A2 = nn.Parameter(
            base.weight.new_empty(self.rank, base.in_features),
        )
        self.lora_B2 = nn.Parameter(
            base.weight.new_zeros(base.out_features, self.rank),
        )
        nn.init.kaiming_uniform_(self.lora_A1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A2, a=math.sqrt(5))

    def _delta_weight(self) -> torch.Tensor:
        first = self.lora_B1 @ self.lora_A1
        second = self.lora_B2 @ self.lora_A2
        return self.scaling * (first * second)

    def _adapter_dtype(self) -> torch.dtype:
        return self.lora_A1.dtype

    def _forward_unmerged(self, input: torch.Tensor) -> torch.Tensor:
        result = self.base(input)
        adapter_input = self._input(input)
        # LoHA takes the Hadamard product of the two reconstructed weight
        # matrices.  Multiplying the two activation outputs would be a
        # different function and would not be merge-equivalent.
        update = F.linear(adapter_input, self._delta_weight())
        return result + update.to(dtype=result.dtype)


class RGLULoRALinear(_AdapterLinear):
    """LoRA with a weight-space Residual GLU correction.

    The default identity mode initializes the value branch to zero so adapter
    injection preserves the base Linear output.  Both modes start the gate at
    one because ``B2`` is zero:

        delta W = scale * (B1 @ A1) * (1 + SiLU(B2 @ A2))

    ``init_mode="lora_warm"`` initializes ``B1`` like a LoRA factor while
    keeping the gate at one, producing a nonzero LoRA-like warm start.  The
    Hadamard product is formed on reconstructed weight matrices, keeping the
    adapter merge-equivalent and independent of the input activation.
    """

    adapter_type = "rglu_lora"

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float | None = None,
        dropout: float = 0.0,
        init_mode: str = "identity",
    ) -> None:
        super().__init__(base, rank, alpha, dropout)
        if init_mode not in {"identity", "lora_warm"}:
            raise ValueError(
                "GLU-LoRA init_mode must be 'identity' or "
                "'lora_warm'"
            )
        self.init_mode = init_mode
        self.lora_A1 = nn.Parameter(
            base.weight.new_empty(self.rank, base.in_features),
        )
        self.lora_B1 = nn.Parameter(
            base.weight.new_empty(base.out_features, self.rank),
        )
        self.lora_A2 = nn.Parameter(
            base.weight.new_empty(self.rank, base.in_features),
        )
        self.lora_B2 = nn.Parameter(
            base.weight.new_zeros(base.out_features, self.rank),
        )
        nn.init.kaiming_uniform_(self.lora_A1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A2, a=math.sqrt(5))
        if init_mode == "identity":
            nn.init.zeros_(self.lora_B1)
        else:
            nn.init.kaiming_uniform_(self.lora_B1, a=math.sqrt(5))

    def _delta_weight(self) -> torch.Tensor:
        value = self.lora_B1 @ self.lora_A1
        gate_logits = self.lora_B2 @ self.lora_A2
        gate = 1.0 + F.silu(gate_logits)
        return self.scaling * (value * gate)

    def _adapter_dtype(self) -> torch.dtype:
        return self.lora_A1.dtype

    def _forward_unmerged(self, input: torch.Tensor) -> torch.Tensor:
        result = self.base(input)
        adapter_input = self._input(input)
        update = F.linear(adapter_input, self._delta_weight())
        return result + update.to(dtype=result.dtype)


class GLULoRALinear(RGLULoRALinear):
    """LoRA with a non-residual weight-space GLU correction.

    The value and gate branches use the same parameter shapes as
    ``RGLULoRALinear``; only the residual gate offset differs:

        delta W = scale * (B1 @ A1) * SiLU(B2 @ A2)

    This is the direct GLU-like extension of LoHA.  It remains mergeable
    because the gate is computed from adapter parameters, not activations.
    """

    adapter_type = "glu_lora"

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float | None = None,
        dropout: float = 0.0,
        init_mode: str = "identity",
    ) -> None:
        super().__init__(base, rank, alpha, dropout, init_mode)
        if init_mode == "identity":
            # B2=0 makes the gate zero, so a zero B1 would also make every
            # first-order gradient zero. Keep the output at the base model
            # while activating the value branch for GLU training.
            nn.init.kaiming_uniform_(self.lora_B1, a=math.sqrt(5))

    def _delta_weight(self) -> torch.Tensor:
        value = self.lora_B1 @ self.lora_A1
        gate_logits = self.lora_B2 @ self.lora_A2
        gate = F.silu(gate_logits)
        return self.scaling * (value * gate)


ADAPTER_TYPES = {
    "lora": LoRALinear,
    "dora": DoRALinear,
    "loha": LoHALinear,
    "glu_lora": GLULoRALinear,
    "rglu_lora": RGLULoRALinear,
}

# Keep loading old checkpoints and old experiment commands possible.  The
# state-dict parameter names are unchanged, so only the metadata/CLI value
# needs normalization.
ADAPTER_ALIASES = {"residual_swiglu_loha": "rglu_lora"}

# Backward-compatible Python import for existing probes and downstream code.
ResidualSwiGLULoHALinear = RGLULoRALinear


def canonicalize_adapter_type(adapter: str) -> str:
    """Return the canonical adapter identifier, accepting legacy aliases."""
    return ADAPTER_ALIASES.get(adapter, adapter)


def resolve_adapter_type(adapter: str, rank: int) -> str:
    """Resolve the legacy rank-only LoRA option to the named adapter API."""
    adapter = canonicalize_adapter_type(adapter)
    if adapter not in {"none", *ADAPTER_TYPES}:
        raise ValueError(f"unknown adapter type: {adapter}")
    if rank < 0:
        raise ValueError("adapter rank must be >= 0")
    if adapter == "none" and rank > 0:
        return "lora"
    if adapter != "none" and rank == 0:
        raise ValueError("a positive adapter rank is required")
    return adapter


def _compile_patterns(patterns: Iterable[str]) -> tuple[re.Pattern[str], ...]:
    try:
        return tuple(re.compile(pattern) for pattern in patterns)
    except re.error as error:
        raise ValueError(f"invalid adapter target pattern: {error}") from error


def inject_adapter(
    model: nn.Module,
    adapter: str,
    target_patterns: Iterable[str],
    rank: int,
    alpha: float | None = None,
    dropout: float = 0.0,
    init_mode: str = "identity",
) -> list[str]:
    """Replace matching Linear children with the selected adapter."""
    adapter = resolve_adapter_type(adapter, rank)
    if adapter == "none":
        return []
    if init_mode != "identity" and adapter not in {"glu_lora", "rglu_lora"}:
        raise ValueError(
            "non-identity init_mode is only supported by "
            "glu_lora and rglu_lora"
        )
    patterns = _compile_patterns(target_patterns)
    if not patterns:
        raise ValueError("at least one adapter target pattern is required")
    adapter_class = ADAPTER_TYPES[adapter]
    matched: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, _AdapterLinear):
                continue
            if isinstance(child, nn.Linear) and any(
                pattern.search(path) for pattern in patterns
            ):
                setattr(
                    parent,
                    name,
                    adapter_class(
                        child,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                        **(
                            {"init_mode": init_mode}
                            if adapter in {"glu_lora", "rglu_lora"}
                            else {}
                        ),
                    ),
                )
                matched.append(path)
            else:
                visit(child, path)

    visit(model, "")
    if not matched:
        raise ValueError("adapter target patterns matched no nn.Linear modules")
    return matched


def inject_lora(
    model: nn.Module,
    target_patterns: Iterable[str],
    rank: int,
    alpha: float | None = None,
    dropout: float = 0.0,
) -> list[str]:
    """Backward-compatible LoRA-specific injection wrapper."""
    return inject_adapter(model, "lora", target_patterns, rank, alpha, dropout)


def mark_only_adapter_trainable(model: nn.Module) -> int:
    """Freeze the model and enable only parameters owned by adapters."""
    trainable = 0
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, _AdapterLinear):
            for name, parameter in module.named_parameters(recurse=False):
                if name.startswith("lora_") or name == "magnitude":
                    parameter.requires_grad_(True)
                    trainable += parameter.numel()
    if trainable == 0:
        raise ValueError("model does not contain adapter parameters")
    return trainable


def mark_only_lora_trainable(model: nn.Module) -> int:
    """Backward-compatible alias for the LoRA training policy."""
    return mark_only_adapter_trainable(model)


def iter_adapter_modules(model: nn.Module):
    """Yield all low-rank adapter modules in model traversal order."""
    return (
        module for module in model.modules()
        if isinstance(module, _AdapterLinear)
    )


def iter_lora_modules(model: nn.Module):
    """Backward-compatible iterator for all adapter modules."""
    return iter_adapter_modules(model)


@torch.no_grad()
def merge_adapter(model: nn.Module) -> None:
    for module in iter_adapter_modules(model):
        module.merge()


@torch.no_grad()
def unmerge_adapter(model: nn.Module) -> None:
    for module in iter_adapter_modules(model):
        module.unmerge()


@torch.no_grad()
def materialize_adapter(model: nn.Module) -> None:
    """Merge adapters and replace them with their plain base Linear layers."""
    def visit(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, _AdapterLinear):
                child.merge()
                setattr(parent, name, child.base)
            else:
                visit(child)

    visit(model)


@torch.no_grad()
def merge_lora(model: nn.Module) -> None:
    """Backward-compatible merge wrapper."""
    merge_adapter(model)


@torch.no_grad()
def unmerge_lora(model: nn.Module) -> None:
    """Backward-compatible unmerge wrapper."""
    unmerge_adapter(model)
