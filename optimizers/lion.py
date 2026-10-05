"""Lion optimizer with parameter-dtype momentum state."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .lion_triton import apply as _triton_apply


class Lion(torch.optim.Optimizer):
    """The plain Lion optimizer for parameters of any shape.

    Lion uses two momentum coefficients: ``beta1`` controls the update
    direction and ``beta2`` controls the momentum state.  It does not use a
    second-moment buffer or bias correction.
    """

    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        *,
        backend="auto",
    ):
        if lr < 0.0:
            raise ValueError("lr must be non-negative")
        if len(betas) != 2 or not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if backend not in {"auto", "torch", "triton"}:
            raise ValueError("backend must be 'auto', 'torch', or 'triton'")
        defaults = dict(
            lr=float(lr), betas=tuple(betas), weight_decay=float(weight_decay),
        )
        super().__init__(params, defaults)
        self.backend = backend

    @torch.no_grad()
    def step(
        self, closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[parameter]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                elif state["exp_avg"].dtype != parameter.dtype:
                    state["exp_avg"] = state["exp_avg"].to(dtype=parameter.dtype)
                exp_avg = state["exp_avg"]
                if self.backend in {"auto", "triton"} and _triton_apply(
                    parameter,
                    parameter.grad,
                    exp_avg,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                ):
                    continue
                update = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1).sign()
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

                if group["weight_decay"]:
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])

        return loss
