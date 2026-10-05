import json

import torch
from torch import nn

from cifar10 import train as cifar10_train
from cifar10.train_adapter import main as train_adapter_main
from safetensors import safe_open


class TinyCIFAR10:
    def __init__(self, *, train, transform=None, **_kwargs):
        self.train = train
        self.transform = transform

    def __len__(self):
        return 4 if self.train else 2

    def __getitem__(self, index):
        image = torch.full((3, 32, 32), float(index) / 4.0)
        label = index % 10
        return image, label


class TinyCIFAR10Model(nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()
        self.linear = nn.Linear(3, 10)

    def forward(self, images, depth=None):
        del depth
        return self.linear(images.mean(dim=(2, 3)))


def _events(run_dir):
    return [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]


def test_adapter_checkpoint_resume_restores_adapter_configuration(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(cifar10_train.datasets, "CIFAR10", TinyCIFAR10)
    monkeypatch.setattr(cifar10_train, "CIFAR10ViT", TinyCIFAR10Model)
    monkeypatch.setattr(cifar10_train, "DEVICE", torch.device("cpu"))

    output_dir = str(tmp_path)
    train_adapter_main([
        "--output-dir", output_dir,
        "--base-init", "random",
        "--adapter", "lora",
        "--lora-rank", "1",
        "--lora-target", "^linear$",
        "--epochs", "1",
        "--batch-size", "2",
        "--num-workers", "0",
        "--seed", "7",
        "--lr-scheduler", "constant",
        "--warmup-steps", "0",
    ])

    first_run = next((tmp_path / "runs").iterdir())
    checkpoint = next(
        first_run.joinpath("checkpoints").glob("*_epoch1*.safetensors")
    )
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        config = json.loads(handle.metadata()["cifar10.config"])
    assert config["args"]["adapter"] == "lora"
    assert config["args"]["lora_rank"] == 1

    train_adapter_main([
        "--output-dir", output_dir,
        "--resume", str(checkpoint),
        "--epochs", "2",
        "--batch-size", "2",
        "--num-workers", "0",
        "--lr-scheduler", "constant",
        "--warmup-steps", "0",
    ])

    run_dirs = list((tmp_path / "runs").iterdir())
    resumed_run = next(path for path in run_dirs if path != first_run)
    assert _events(resumed_run)[-1]["status"] == "completed"
    resumed_config = json.loads(
        (resumed_run / "config.json").read_text(encoding="utf-8")
    )
    assert resumed_config["args"]["adapter"] == "lora"
    assert resumed_config["args"]["lora_rank"] == 1
