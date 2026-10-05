import json
import os
import signal
import sys

import torch
from torch import nn
from safetensors import safe_open

from text_lm import train as text_lm_train
from cifar10 import train as cifar10_train
from runtime.checkpoint import load_training_state


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


class TinyCIFAR10Model(nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()
        self.linear = nn.Linear(3, 10)

    def forward(self, images, depth=None):
        del depth
        pooled = images.mean(dim=(2, 3))
        return self.linear(pooled)


def _read_events(run_dir):
    return [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _assert_interrupted_checkpoint(run_dir, checkpoint_path):
    resume_state = load_training_state(checkpoint_path)
    assert resume_state is not None
    assert resume_state["global_step"] == 1
    assert resume_state["extra"]["sampler"]["position"] == 2

    events = _read_events(run_dir)
    assert any(
        event["event"] == "checkpoint" and event["kind"] == "interrupted"
        for event in events
    )
    assert events[-1]["event"] == "run_finished"
    assert events[-1]["status"] == "interrupted"


def test_cifar10_main_saves_resumable_checkpoint_on_sigint(tmp_path, monkeypatch):
    InterruptingCIFAR10.armed = True
    monkeypatch.setattr(cifar10_train.datasets, "CIFAR10", InterruptingCIFAR10)
    monkeypatch.setattr(cifar10_train, "CIFAR10ViT", TinyCIFAR10Model)
    monkeypatch.setattr(cifar10_train, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cifar10.train",
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
    cifar10_train.main()

    assert signal.getsignal(signal.SIGINT) == previous_handler
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    checkpoints = list((run_dir / "checkpoints").glob("*_latest.safetensors"))
    assert len(checkpoints) == 1
    _assert_interrupted_checkpoint(run_dir, checkpoints[0])
    with safe_open(checkpoints[0], framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
    assert "cifar10.config" in metadata
    assert json.loads(metadata["cifar10.config"])["args"]["optimizer"] == "AdamW"

    InterruptingCIFAR10.armed = False
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cifar10.train",
            "--output-dir",
            str(tmp_path),
            "--resume",
            str(checkpoints[0]),
        ],
    )
    cifar10_train.main()

    resumed_run_dir = next(
        path for path in (tmp_path / "runs").iterdir() if path != run_dir
    )
    assert _read_events(resumed_run_dir)[-1]["status"] == "completed"
    resumed_config = json.loads(
        (resumed_run_dir / "config.json").read_text(encoding="utf-8")
    )
    assert resumed_config["args"]["optimizer"] == "AdamW"
    assert resumed_config["args"]["attention_type"] == "full"


class InterruptingTokenDataset:
    armed = True
    column_names = ["input_ids", "attention_mask"]

    def __init__(self, *, train):
        self.train = train
        self.sent_signal = False
        self.items = [
            {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]},
            {"input_ids": [2, 3, 4, 5], "attention_mask": [1, 1, 1, 1]},
            {"input_ids": [3, 4, 5, 6], "attention_mask": [1, 1, 1, 1]},
            {"input_ids": [1, 3, 5, 7], "attention_mask": [1, 1, 1, 1]},
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        if self.train and self.armed and not self.sent_signal:
            self.sent_signal = True
            os.kill(os.getpid(), signal.SIGINT)
        return self.items[index]

    def __iter__(self):
        return iter(self.items)

    def map(self, _function, remove_columns):
        del remove_columns
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 5
    eos_token = "<eos>"
    pad_token = "<pad>"
    chat_template = None

    def __len__(self):
        return 8

    def save_pretrained(self, _output_dir):
        return None

    def pad(self, features, padding=True, return_tensors=None):
        del padding
        if return_tensors != "pt":
            raise ValueError("the integration test expects tensor padding")
        input_ids = torch.tensor(
            [feature["input_ids"] for feature in features], dtype=torch.long
        )
        attention_mask = torch.tensor(
            [feature["attention_mask"] for feature in features], dtype=torch.long
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(*_args, **_kwargs):
        return FakeTokenizer()


class TinyTextModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len, embed_dim, **_kwargs):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(
        self, input_ids, attention_mask=None, depth=None, return_hidden=False,
    ):
        del attention_mask, depth
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        hidden = self.token_embedding(input_ids)
        hidden = hidden + self.position_embedding(positions)[None, :, :]
        if return_hidden:
            return hidden
        return self.output(hidden)


def test_text_lm_main_saves_resumable_checkpoint_on_sigint(tmp_path, monkeypatch):
    InterruptingTokenDataset.armed = True
    monkeypatch.setattr(text_lm_train, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(text_lm_train, "TinyTextLM", TinyTextModel)
    monkeypatch.setattr(
        text_lm_train,
        "load_instruction_dataset",
        lambda _dataset_name, _dataset_path, _eval_ratio, _seed: (
            InterruptingTokenDataset(train=True),
            InterruptingTokenDataset(train=False),
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "text_lm.train",
            "--output-dir",
            str(tmp_path),
            "--data-mode",
            "instruction",
            "--max-seq-len",
            "4",
            "--embed-dim",
            "4",
            "--num-layers",
            "8",
            "--num-heads",
            "1",
            "--kv-heads",
            "1",
            "--architecture",
            "mhla3-gqa-looped-hybrid",
            "--mhla-looped-prefix-cycles",
            "0",
            "--mhla-looped-repeats",
            "2",
            "--mhla-looped-suffix-cycles",
            "0",
            "--condition-dim",
            "2",
            "--transform-rank",
            "1",
            "--epochs",
            "1",
            "--steps-per-epoch",
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
    text_lm_train.main()

    assert signal.getsignal(signal.SIGINT) == previous_handler
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    checkpoint_path = run_dir / "checkpoints" / "latest" / "model.safetensors"
    assert checkpoint_path.is_file()
    _assert_interrupted_checkpoint(run_dir, checkpoint_path)
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
    assert text_lm_train.TEXT_LM_CONFIG_METADATA_KEY in metadata
    saved_args = json.loads(
        metadata[text_lm_train.TEXT_LM_CONFIG_METADATA_KEY]
    )["args"]
    assert saved_args["embed_dim"] == 4
    assert saved_args["architecture"] == "mhla3-gqa-looped-hybrid"
    assert saved_args["mhla_looped_prefix_cycles"] == 0
    assert saved_args["mhla_looped_repeats"] == 2
    assert saved_args["mhla_looped_suffix_cycles"] == 0

    InterruptingTokenDataset.armed = False
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "text_lm.train",
            "--output-dir",
            str(tmp_path),
            "--resume",
            str(checkpoint_path),
        ],
    )
    text_lm_train.main()

    resumed_run_dir = next(
        path for path in (tmp_path / "runs").iterdir() if path != run_dir
    )
    assert _read_events(resumed_run_dir)[-1]["status"] == "completed"
    resumed_config = json.loads(
        (resumed_run_dir / "config.json").read_text(encoding="utf-8")
    )
    assert resumed_config["args"]["embed_dim"] == 4
    assert resumed_config["args"]["architecture"] == (
        "mhla3-gqa-looped-hybrid"
    )
    assert resumed_config["args"]["mhla_looped_repeats"] == 2
    assert resumed_config["args"]["steps_per_epoch"] == 1
