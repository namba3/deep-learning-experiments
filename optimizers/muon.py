"""Single-device Muon with parameter-dtype momentum state."""

from __future__ import annotations

from typing import Callable, Optional

import torch
from muon import SingleDeviceMuon as _SingleDeviceMuon
from muon import muon_update

from ._matrix_common import apply_muon_update, ensure_momentum_buffer


class SingleDeviceMuon(_SingleDeviceMuon):
    """Single-device Muon with parameter-dtype momentum state."""

    def __init__(self, params, lr=0.02, weight_decay: float = 0.0, momentum=0.95, *, backend="auto"):
        if backend not in {"auto", "torch", "triton"}:
            raise ValueError("backend must be 'auto', 'torch', or 'triton'")
        super().__init__(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
        self.backend = backend

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach().clone()
                if grad.is_sparse:
                    raise RuntimeError(
                        "SingleDeviceMuon does not support sparse gradients"
                    )

                state = self.state[parameter]
                momentum = ensure_momentum_buffer(
                    state, grad, dtype=parameter.dtype,
                )

                update = muon_update(
                    grad,
                    momentum,
                    beta=group["momentum"],
                )
                apply_muon_update(
                    parameter, update, lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    backend=self.backend,
                )

        return loss
