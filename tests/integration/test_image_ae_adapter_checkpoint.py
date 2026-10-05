import json
import sys

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from image_ae import export_adapter, train as image_ae_train
from image_ae.train_adapter import main as train_adapter_main


class TinyCIFAR10:
    def __init__(self, *, train, transform=None, **_kwargs):
        self.train = train
        self.transform = transform

    def __len__(self):
        return 4 if self.train else 2

    def __getitem__(self, index):
        value = np.full((32, 32, 3), 32 + index * 16, dtype=np.uint8)
        image = Image.fromarray(value, mode="RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, index % 10


class TinyImageAE(nn.Module):
    def __init__(self, latent_channels, **_kwargs):
        super().__init__()
        self.latent_channels = latent_channels
        self.vae = False
        self.encoder = nn.Linear(3, latent_channels)
        self.decoder = nn.Linear(latent_channels, 3)

    def encode(self, images):
        channels_last = images.movedim(1, -1)
        return self.encoder(channels_last).movedim(-1, 1)

    def decode(self, latents):
        channels_last = latents.movedim(1, -1)
        return torch.sigmoid(self.decoder(channels_last)).movedim(-1, 1)

    def forward(self, images):
        latents = self.encode(images)
        return self.decode(latents), latents


def _common_argv(output_dir, *, epochs, checkpoint=None, base=None):
    argv = [
        "image_ae.train",
        "--output-dir", str(output_dir),
        "--dataset", "cifar10",
        "--data-dir", str(output_dir),
        "--image-size", "32",
        "--latent-channels", "2",
        "--bottleneck-channels", "16",
        "--encoder", "basic_cnn",
        "--decoder", "basic_cnn",
        "--epochs", str(epochs),
        "--batch-size", "2",
        "--cifar10-train-samples", "4",
        "--cifar10-val-samples", "2",
        "--num-workers", "0",
        "--seed", "7",
        "--optimizer", "AdamW",
        "--lr-scheduler", "constant",
        "--warmup-steps", "0",
    ]
    if checkpoint is not None:
        argv.extend(["--resume", str(checkpoint)])
    if base is not None:
        argv.extend([
            "--lora-base-checkpoint", str(base),
            "--adapter", "lora",
            "--lora-rank", "1",
            "--lora-target", "^encoder$",
        ])
    return argv


def _events(run_dir):
    return [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_image_ae_adapter_base_checkpoint_resume_and_merge(tmp_path, monkeypatch):
    monkeypatch.setattr(image_ae_train.datasets, "CIFAR10", TinyCIFAR10)
    monkeypatch.setattr(image_ae_train, "ImageAE", TinyImageAE)

    base_model = TinyImageAE(latent_channels=2)
    base_checkpoint = tmp_path / "base.safetensors"
    monkeypatch.setattr(sys, "argv", _common_argv(tmp_path, epochs=1))
    base_args = image_ae_train.parse_args()
    base_args.adapter = "none"
    base_args.lora_rank = 0
    save_file(
        base_model.state_dict(),
        base_checkpoint,
        metadata=image_ae_train.checkpoint_metadata(base_args),
    )

    monkeypatch.setattr(
        sys, "argv", _common_argv(tmp_path, epochs=1, base=base_checkpoint),
    )
    train_adapter_main()
    first_run = next(path for path in (tmp_path / "runs").iterdir())
    adapter_checkpoint = next(
        first_run.joinpath("checkpoints").glob("*_latest.safetensors"),
    )
    with safe_open(adapter_checkpoint, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = set(handle.keys())
    training_config = json.loads(metadata[image_ae_train.TRAINING_CONFIG_METADATA_KEY])
    assert training_config["adapter"] == "lora"
    assert any("lora_" in key for key in keys)

    monkeypatch.setattr(
        sys, "argv", _common_argv(tmp_path, epochs=2, checkpoint=adapter_checkpoint),
    )
    train_adapter_main()
    run_dirs = list((tmp_path / "runs").iterdir())
    resumed_run = next(path for path in run_dirs if path != first_run)
    assert _events(resumed_run)[-1]["status"] == "completed"

    merged_checkpoint = tmp_path / "merged.safetensors"
    monkeypatch.setattr(export_adapter, "_build_model", lambda *_args: TinyImageAE(2))
    export_adapter.export_checkpoint(str(adapter_checkpoint), str(merged_checkpoint))
    with safe_open(merged_checkpoint, framework="pt", device="cpu") as handle:
        merged_keys = set(handle.keys())
        merged_metadata = handle.metadata() or {}
    assert not any("lora_" in key for key in merged_keys)
    merged_config = json.loads(
        merged_metadata[image_ae_train.TRAINING_CONFIG_METADATA_KEY],
    )
    assert merged_config["adapter"] == "none"
    assert merged_config["merged_adapter"] is True
