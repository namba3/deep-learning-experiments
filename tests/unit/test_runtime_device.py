import argparse

import pytest
import torch

from runtime.device import add_device_argument, resolve_device


def test_resolve_device_auto_uses_default(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device("auto", default=torch.device("cpu")) == torch.device("cpu")


def test_resolve_device_auto_detects_available_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device("auto") == torch.device("cpu")


def test_resolve_device_cpu_is_always_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_cuda_fails_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")


def test_resolve_device_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device("cuda") == torch.device("cuda")


def test_add_device_argument_uses_common_default():
    parser = argparse.ArgumentParser()
    add_device_argument(parser)

    assert parser.parse_args([]).device == "auto"
    assert parser.parse_args(["--device", "cpu"]).device == "cpu"
