import json
import os
import signal
import sys

import torch
from torch import nn
from safetensors import safe_open

from mnist import train as mnist_train
from runtime.checkpoint import load_training_state


class InterruptingMNIST:
    armed = True

    def __init__(self, *, train, **_kwargs):
        self.train = train
        self.sent_signal = False

    def __len__(self):
        return 4 if self.train else 2

    def __getitem__(self, index):
        if self.train and self.armed and not self.sent_signal:
            self.sent_signal = True
            os.kill(os.getpid(), signal.SIGINT)
        image = torch.full((1, 28, 28), float(index) / 2.0)
        return image, index % 10


class TinyMNISTModel(nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()
        self.linear = nn.Linear(1, 10)

    def forward(self, images):
        pooled = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return self.linear(pooled)


def test_mnist_main_saves_resumable_checkpoint_on_sigint(tmp_path, monkeypatch):
    InterruptingMNIST.armed = True
    monkeypatch.setattr(mnist_train.datasets, "MNIST", InterruptingMNIST)
    monkeypatch.setattr(mnist_train, "MNISTViT", TinyMNISTModel)
    monkeypatch.setattr(mnist_train, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mnist.train",
            "--output-dir",
            str(tmp_path),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--seed",
            "7",
            "--lr-scheduler",
            "constant",
            "--warmup-steps",
            "0",
        ],
    )

    previous_handler = signal.getsignal(signal.SIGINT)
    mnist_train.main()

    assert signal.getsignal(signal.SIGINT) == previous_handler
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    checkpoints = list((run_dir / "checkpoints").glob("*_latest.safetensors"))
    assert len(checkpoints) == 1
    with safe_open(checkpoints[0], framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
    assert "mnist.config" in metadata
    assert json.loads(metadata["mnist.config"])["args"]["optimizer"] == "AdamW"

    resume_state = load_training_state(checkpoints[0])
    assert resume_state is not None
    assert resume_state["global_step"] == 1
    assert resume_state["extra"]["sampler"]["position"] == 2

    events = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        event["event"] == "checkpoint" and event["kind"] == "interrupted"
        for event in events
    )
    assert events[-1]["event"] == "run_finished"
    assert events[-1]["status"] == "interrupted"

    InterruptingMNIST.armed = False
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mnist.train",
            "--output-dir",
            str(tmp_path),
            "--resume",
            str(checkpoints[0]),
        ],
    )
    mnist_train.main()

    resumed_run_dir = next(
        path for path in (tmp_path / "runs").iterdir() if path != run_dir
    )
    resumed_events = [
        json.loads(line)
        for line in (resumed_run_dir / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert resumed_events[-1]["event"] == "run_finished"
    assert resumed_events[-1]["status"] == "completed"
    resumed_config = json.loads(
        (resumed_run_dir / "config.json").read_text(encoding="utf-8")
    )
    assert resumed_config["args"]["optimizer"] == "AdamW"
    assert resumed_config["args"]["batch_size"] == 2

    init_output_dir = tmp_path / "init-only"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mnist.train",
            "--output-dir",
            str(init_output_dir),
            "--init-checkpoint",
            str(checkpoints[0]),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--seed",
            "99",
            "--lr-scheduler",
            "constant",
            "--warmup-steps",
            "0",
        ],
    )
    mnist_train.main()

    init_run_dir = next((init_output_dir / "runs").iterdir())
    init_events = [
        json.loads(line)
        for line in (init_run_dir / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert init_events[-1]["event"] == "run_finished"
    assert init_events[-1]["status"] == "completed"
    init_config = json.loads(
        (init_run_dir / "config.json").read_text(encoding="utf-8")
    )
    assert init_config["args"]["resume"] is None
    assert init_config["args"]["init_checkpoint"] == str(checkpoints[0])
