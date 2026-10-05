import json
import os
import signal
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from safetensors import safe_open

from image_ae import train as image_ae_train
from image_gen import train as image_gen_train
from runtime.checkpoint import load_training_state
from runtime.signal import GracefulStop


class InterruptingCIFAR10:
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
        image = torch.full((3, 32, 32), float(index) / 2.0)
        return image, index % 10


class TinyImageAE(nn.Module):
    def __init__(self, latent_channels, **_kwargs):
        super().__init__()
        self.latent_channels = latent_channels
        self.vae = False
        self.encoder = nn.Conv2d(3, latent_channels, kernel_size=1)
        self.decoder = nn.Conv2d(latent_channels, 3, kernel_size=1)

    def encode(self, images):
        return self.encoder(images)

    def decode(self, latents):
        return torch.sigmoid(self.decoder(latents))

    def forward(self, images):
        latents = self.encode(images)
        return self.decode(latents), latents


class InterruptingRecordsDataset:
    armed = False

    def __init__(self, _records_path, image_size, _bucket_step):
        self.bucket_shapes = ((image_size, image_size),)
        self.bucket_ids = [0, 0]
        self.sent_signal = False

    def __len__(self):
        return 2

    def __getitem__(self, _index):
        if self.armed and not self.sent_signal:
            self.sent_signal = True
            os.kill(os.getpid(), signal.SIGINT)
        return torch.zeros(3, 32, 32), "synthetic caption"


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"


class FakeTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 4)
        self.config = SimpleNamespace(hidden_size=4, max_position_embeddings=8)


class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(scaling_factor=1.0)


class TinyTextAdapter(nn.Module):
    def __init__(self, text_encoder_dim, adapter_dim, *_args):
        super().__init__()
        self.projection = nn.Linear(text_encoder_dim, adapter_dim)

    def forward(self, hidden_states, _attention_mask=None):
        return self.projection(hidden_states)


class TinyDiT(nn.Module):
    def __init__(self, latent_channels, *_args):
        super().__init__()
        self.image_input_dim = latent_channels
        self.scale = nn.Parameter(torch.ones(()))

    def forward(
        self, latents, _timesteps, _text_context, _text_context_mask=None,
        timing=None,
    ):
        del timing
        return latents * self.scale


class ArmingGracefulStop(GracefulStop):
    arm_on_install = True

    def install(self):
        super().install()
        if self.arm_on_install:
            InterruptingRecordsDataset.armed = True


def _events(run_dir):
    return [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _assert_interrupted_run(run_dir, checkpoint_path, sampler_position):
    assert checkpoint_path.is_file()
    resume_state = load_training_state(checkpoint_path)
    assert resume_state is not None
    assert resume_state["global_step"] == 1
    assert resume_state["extra"]["sampler"]["position"] == sampler_position
    assert _events(run_dir)[-1]["status"] == "interrupted"


def _checkpoint_metadata(checkpoint_path):
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        return checkpoint.metadata() or {}


def test_image_ae_main_saves_resumable_checkpoint_on_sigint(tmp_path, monkeypatch):
    InterruptingCIFAR10.armed = True
    monkeypatch.setattr(image_ae_train.datasets, "CIFAR10", InterruptingCIFAR10)
    monkeypatch.setattr(image_ae_train, "ImageAE", TinyImageAE)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "image_ae.train",
            "--output-dir",
            str(tmp_path),
            "--dataset",
            "cifar10",
            "--image-size",
            "32",
            "--latent-channels",
            "2",
            "--bottleneck-channels",
            "16",
            "--encoder",
            "basic_cnn",
            "--decoder",
            "basic_cnn",
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
    image_ae_train.main()

    assert signal.getsignal(signal.SIGINT) == previous_handler
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    checkpoints = list((run_dir / "checkpoints").glob("*_latest.safetensors"))
    assert len(checkpoints) == 1
    _assert_interrupted_run(run_dir, checkpoints[0], sampler_position=2)
    metadata = _checkpoint_metadata(checkpoints[0])
    network_config = json.loads(metadata["image_ae.network_config"])
    training_config = json.loads(metadata["image_ae.training_config"])
    assert network_config["dataset"] == "cifar10"
    assert network_config["latent_channels"] == 2
    assert training_config["optimizer"] == "AdamW"
    assert training_config["lr_scheduler"] == "constant"
    assert training_config["warmup_steps"] == 0

    InterruptingCIFAR10.armed = False
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "image_ae.train",
            "--output-dir",
            str(tmp_path),
            "--dataset",
            "cifar10",
            "--image-size",
            "32",
            "--latent-channels",
            "2",
            "--bottleneck-channels",
            "16",
            "--encoder",
            "basic_cnn",
            "--decoder",
            "basic_cnn",
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
            "--resume",
            str(checkpoints[0]),
        ],
    )
    image_ae_train.main()

    resumed_run_dir = next(
        path for path in (tmp_path / "runs").iterdir() if path != run_dir
    )
    assert _events(resumed_run_dir)[-1]["status"] == "completed"


def test_image_gen_main_saves_resumable_checkpoint_on_sigint(tmp_path, monkeypatch):
    from diffusers import AutoencoderKL
    from transformers import AutoModel, AutoTokenizer
    from image_gen import training_data, training_model

    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: FakeTokenizer()),
    )
    monkeypatch.setattr(
        AutoModel,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: FakeTextEncoder()),
    )
    monkeypatch.setattr(
        AutoencoderKL,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: FakeVAE()),
    )
    monkeypatch.setattr(training_data, "RecordsDataset", InterruptingRecordsDataset)
    monkeypatch.setattr(training_model, "TextConditioningAdapter", TinyTextAdapter)
    monkeypatch.setattr(training_model, "DiT", TinyDiT)
    monkeypatch.setattr(
        training_data,
        "validate_bucket_shapes_with_vae",
        lambda *_args, **_kwargs: (2, 2),
    )
    def fake_encode_images(_vae, images, _latent_scale=None):
        return F.avg_pool2d(images, 2)[:, :2]

    monkeypatch.setattr(
        image_gen_train,
        "encode_images",
        fake_encode_images,
    )
    monkeypatch.setattr(training_data, "encode_images", fake_encode_images)
    monkeypatch.setattr(
        image_gen_train,
        "encode_text",
        lambda _tokenizer, _text_encoder, captions, device, _max_length: (
            torch.zeros(len(captions), 2, 4, device=device),
            torch.ones(len(captions), 2, dtype=torch.bool, device=device),
        ),
    )
    monkeypatch.setattr(image_gen_train, "GracefulStop", ArmingGracefulStop)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "image_gen.train",
            "--output-dir",
            str(tmp_path),
            "--records",
            "synthetic.jsonl",
            "--vae-model",
            "dummy",
            "--text-model",
            "dummy",
            "--image-size",
            "32",
            "--bucket-step",
            "32",
            "--latent-channels",
            "2",
            "--text-adapter-dim",
            "8",
            "--text-adapter-transformer-dims",
            "32",
            "--text-adapter-transformer-heads",
            "4",
            "--text-adapter-transformer-kv-heads",
            "2",
            "--model-dim",
            "16",
            "--depth",
            "1",
            "--heads",
            "4",
            "--kv-heads",
            "2",
            "--context-depth",
            "1",
            "--context-heads",
            "2",
            "--context-kv-heads",
            "1",
            "--attention-pattern",
            "full",
            "--attention-gate",
            "none",
            "--patch-size",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--seed",
            "7",
            "--amp",
            "no",
            "--trainable-dtype",
            "fp32",
            "--timestep-repeats",
            "1",
            "--observe-interval",
            "100",
            "--sample-steps",
            "1",
            "--gc-interval",
            "100",
            "--optimizer",
            "AdamW",
            "--lr-scheduler",
            "constant",
            "--warmup-steps",
            "0",
        ],
    )

    previous_handler = signal.getsignal(signal.SIGINT)
    image_gen_train.main()

    assert signal.getsignal(signal.SIGINT) == previous_handler
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    checkpoint_path = run_dir / "checkpoints" / "checkpoint_latest.safetensors"
    _assert_interrupted_run(run_dir, checkpoint_path, sampler_position=1)
    metadata = _checkpoint_metadata(checkpoint_path)
    checkpoint_config = json.loads(metadata["image_gen.config"])
    assert checkpoint_config["network_version"] == image_gen_train.NETWORK_CONFIG_VERSION
    assert checkpoint_config["latent_channels"] == 2
    assert checkpoint_config["optimizer"] == "AdamW"
    assert checkpoint_config["lr_scheduler"] == "constant"
    assert checkpoint_config["warmup_steps"] == 0

    InterruptingRecordsDataset.armed = False
    ArmingGracefulStop.arm_on_install = False
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "image_gen.train",
            "--output-dir",
            str(tmp_path),
            "--records",
            "synthetic.jsonl",
            "--vae-model",
            "dummy",
            "--resume",
            str(checkpoint_path),
        ],
    )
    image_gen_train.main()

    resumed_run_dir = next(
        path for path in (tmp_path / "runs").iterdir() if path != run_dir
    )
    assert _events(resumed_run_dir)[-1]["status"] == "completed"
    resumed_config = json.loads(
        (resumed_run_dir / "config.json").read_text(encoding="utf-8")
    )
    assert resumed_config["args"]["optimizer"] == "AdamW"
    assert resumed_config["args"]["model_dim"] == 16
    assert resumed_config["args"]["trainable_dtype"] == "fp32"
