"""Muon variants: NorMuon and AdaMuon."""

from __future__ import annotations

import torch
from muon import muon_update

from ._matrix_common import (
    AdamFallback,
    apply_muon_update,
    ensure_momentum_buffer,
    loss_from_closure,
    muon_matrix,
)


class NorMuon(torch.optim.Optimizer):
    """Muon with neuron-wise (row-wise) adaptive normalization."""

    def __init__(self, params, lr=1e-2, momentum=0.95, beta2=0.99,
                 eps=1e-8, weight_decay=0.0, *, backend="auto"):
        if backend not in {"auto", "torch", "triton"}:
            raise ValueError("backend must be 'auto', 'torch', or 'triton'")
        defaults = dict(lr=lr, momentum=momentum, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, betas=(momentum, beta2))
        super().__init__(params, defaults)
        self.backend = backend

    @torch.no_grad()
    def step(self, closure=None):
        loss = loss_from_closure(closure)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                raw_grad = parameter.grad
                if raw_grad.is_sparse:
                    raise RuntimeError("NorMuon does not support sparse gradients")
                state = self.state[parameter]
                if parameter.ndim < 2:
                    AdamFallback.step(parameter, raw_grad.float(), state, group)
                    continue
                matrix = muon_matrix(parameter, raw_grad.detach().clone())
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        matrix, dtype=parameter.dtype,
                    )
                    state["row_norm_sq"] = torch.zeros(matrix.shape[0], device=matrix.device)
                momentum = ensure_momentum_buffer(
                    state, matrix, dtype=parameter.dtype,
                )
                update = muon_update(matrix, momentum, beta=group["momentum"])
                row_norm_sq = update.square().mean(dim=1)
                beta2, eps = group["beta2"], group["eps"]
                state["row_norm_sq"].mul_(beta2).add_(row_norm_sq, alpha=1 - beta2)
                target = state["row_norm_sq"].mean().sqrt()
                update = update * target / state["row_norm_sq"].add(eps).sqrt().unsqueeze(1)
                apply_muon_update(
                    parameter, update, lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    backend=self.backend,
                )
        return loss


class AdaMuon(torch.optim.Optimizer):
    """Adaptive Muon with an element-wise second-moment estimate."""

    def __init__(self, params, lr=1e-2, momentum=0.95, beta2=0.99,
                 eps=1e-8, weight_decay=0.0, *, backend="auto"):
        if backend not in {"auto", "torch", "triton"}:
            raise ValueError("backend must be 'auto', 'torch', or 'triton'")
        defaults = dict(lr=lr, momentum=momentum, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, betas=(momentum, beta2))
        super().__init__(params, defaults)
        self.backend = backend

    @torch.no_grad()
    def step(self, closure=None):
        loss = loss_from_closure(closure)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                raw_grad = parameter.grad
                if raw_grad.is_sparse:
                    raise RuntimeError("AdaMuon does not support sparse gradients")
                state = self.state[parameter]
                if parameter.ndim < 2:
                    AdamFallback.step(parameter, raw_grad.float(), state, group)
                    continue
                matrix = muon_matrix(parameter, raw_grad.detach().clone())
                if "exp_avg_sq" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        matrix, dtype=parameter.dtype,
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        matrix, dtype=parameter.dtype,
                    )
                    state["step"] = 0
                state["step"] += 1
                momentum = ensure_momentum_buffer(
                    state, matrix, dtype=parameter.dtype,
                )
                if state["exp_avg_sq"].dtype != parameter.dtype:
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(
                        dtype=parameter.dtype,
                    )
                update = muon_update(matrix, momentum, beta=group["momentum"])
                beta2, eps = group["beta2"], group["eps"]
                state["exp_avg_sq"].mul_(beta2).addcmul_(update, update, value=1 - beta2)
                adaptive = update / state["exp_avg_sq"].add(eps).sqrt()
                adaptive = adaptive * update.square().mean().sqrt() / adaptive.square().mean().sqrt().clamp_min(eps)
                apply_muon_update(
                    parameter, adaptive, lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    backend=self.backend,
                )
        return loss
