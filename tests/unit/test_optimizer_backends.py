from typing import Any

import pytest

import torch

from optimizers.adamw import AdamWAutoSchedule, AdamWFP32State

from optimizers.came import CAME

from optimizers.lion import Lion

from optimizers.muon_variants import AdaMuon, NorMuon

from optimizers.schedulefree import AdamWScheduleFree, RAdamScheduleFree

def _assign_gradient(parameter, value=1.0):
    parameter.grad = torch.full_like(parameter, value)

def test_came_triton_backend_falls_back_without_cuda():
    parameter = torch.nn.Parameter(torch.randn(4, 3))
    optimizer = CAME([parameter], lr=0.01, backend="triton")
    _assign_gradient(parameter)
    optimizer.step()

    assert torch.isfinite(parameter).all()

@pytest.mark.parametrize("optimizer_type", [AdamWFP32State, AdamWAutoSchedule])
def test_adamw_triton_backend_falls_back_without_cuda(optimizer_type):
    parameter = torch.nn.Parameter(torch.randn(4, 3))
    kwargs: dict[str, Any] = dict(lr=0.01, weight_decay=0.01, backend="triton")
    if optimizer_type is AdamWAutoSchedule:
        kwargs["auto_schedule_warmup_steps"] = 0
    optimizer = optimizer_type([parameter], **kwargs)
    _assign_gradient(parameter)
    optimizer.step()

    assert torch.isfinite(parameter).all()

def test_lion_triton_backend_falls_back_without_cuda():
    parameter = torch.nn.Parameter(torch.randn(4, 3))
    optimizer = Lion([parameter], lr=0.01, weight_decay=0.01, backend="triton")
    _assign_gradient(parameter)
    optimizer.step()

    assert torch.isfinite(parameter).all()

def test_schedulefree_adamw_triton_backend_falls_back_without_cuda():
    parameter = torch.nn.Parameter(torch.randn(4, 3))
    optimizer = AdamWScheduleFree(
        [parameter], lr=0.01, weight_decay=0.01,
        warmup_steps=0, backend="triton",
    )
    optimizer.train()
    _assign_gradient(parameter)
    optimizer.step()

    assert torch.isfinite(parameter).all()

def test_schedulefree_radam_triton_backend_falls_back_without_cuda():
    parameter = torch.nn.Parameter(torch.randn(4, 3))
    optimizer = RAdamScheduleFree(
        [parameter], lr=0.01, weight_decay=0.01, backend="triton",
    )
    optimizer.train()
    _assign_gradient(parameter)
    optimizer.step()

    assert torch.isfinite(parameter).all()

@pytest.mark.parametrize("optimizer_type", [NorMuon, AdaMuon])
def test_muon_triton_backend_falls_back_without_cuda(optimizer_type):
    parameter = torch.nn.Parameter(torch.randn(4, 3))
    optimizer = optimizer_type(
        [parameter], lr=0.01, weight_decay=0.01, backend="triton",
    )
    _assign_gradient(parameter)
    optimizer.step()

    assert torch.isfinite(parameter).all()
