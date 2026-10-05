import json

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from text_lm import train as text_lm_train
from text_lm.adapter_training import DEFAULT_ADAPTER_TARGETS
from text_lm.export_adapter import export_checkpoint
from text_lm.train import TinyTextLM
from text_lm.train_adapter import main as train_adapter_main
from core.low_rank import inject_adapter


class TinyTokenDataset:
    column_names = ["input_ids", "attention_mask"]

    def __init__(self, *, train):
        self.train = train
        self.items = [
            {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]},
            {"input_ids": [2, 3, 4, 5], "attention_mask": [1, 1, 1, 1]},
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __iter__(self):
        return iter(self.items)

    def map(self, _function, remove_columns, **_kwargs):
        del remove_columns
        return self


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 5
    pad_token = "<pad>"
    eos_token = "<eos>"
    chat_template = None

    def __len__(self):
        return 8

    def __call__(self, *_args, **_kwargs):
        return {"input_ids": [1, 2, 3, 5], "attention_mask": [1, 1, 1, 1]}

    def pad(self, features, padding=True, return_tensors=None):
        del padding
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([item["input_ids"] for item in features]),
            "attention_mask": torch.tensor(
                [item["attention_mask"] for item in features]
            ),
        }

    def save_pretrained(self, _output_dir):
        return None


class TinyTextModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, **_kwargs):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(
        self, input_ids, attention_mask=None, depth=None, return_hidden=False,
    ):
        del attention_mask, depth
        hidden = self.token_embedding(input_ids)
        if return_hidden:
            return hidden
        return self.output(hidden)


def test_text_lm_adapter_entrypoint_trains(tmp_path, monkeypatch):
    monkeypatch.setattr(text_lm_train, "AutoTokenizer", type(
        "FakeAutoTokenizer", (), {
            "from_pretrained": staticmethod(lambda *_args, **_kwargs: TinyTokenizer()),
        },
    ))
    monkeypatch.setattr(text_lm_train, "TinyTextLM", TinyTextModel)
    monkeypatch.setattr(
        text_lm_train,
        "load_instruction_dataset",
        lambda *_args: (TinyTokenDataset(train=True), TinyTokenDataset(train=False)),
    )

    base_model = TinyTextModel(vocab_size=8, embed_dim=4)
    base_checkpoint = tmp_path / "base.safetensors"
    save_file(base_model.state_dict(), base_checkpoint)

    train_adapter_main([
        "--output-dir", str(tmp_path),
        "--data-mode", "instruction",
        "--lora-base-checkpoint", str(base_checkpoint),
        "--adapter", "lora", "--lora-rank", "1",
        "--lora-target", "^output$",
        "--architecture", "naive", "--embed-dim", "4",
        "--max-seq-len", "4", "--num-layers", "1", "--num-heads", "1",
        "--kv-heads", "1", "--batch-size", "2", "--epochs", "1",
        "--steps-per-epoch", "1", "--eval-max-batches", "1",
        "--num-workers", "0", "--lr-scheduler", "constant",
        "--warmup-steps", "0", "--dataset-path", "ignored.json",
    ])

    run_dir = next(path for path in (tmp_path / "runs").iterdir())
    checkpoint = next(run_dir.joinpath("checkpoints").rglob("model.safetensors"))
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = set(handle.keys())
    config = json.loads(metadata[text_lm_train.TEXT_LM_CONFIG_METADATA_KEY])
    assert config["args"]["adapter"] == "lora"
    assert "output.lora_A" in keys


def test_text_lm_export_materializes_adapter(tmp_path):
    model = TinyTextLM(
        vocab_size=31, max_seq_len=8, embed_dim=16, num_layers=2,
        num_heads=4, kv_heads=2, architecture="naive",
    )
    inject_adapter(
        model, "lora", DEFAULT_ADAPTER_TARGETS, rank=2, alpha=2,
    )
    config = {
        "max_seq_len": 8,
        "embed_dim": 16,
        "num_layers": 2,
        "num_heads": 4,
        "kv_heads": 2,
        "condition_dim": 64,
        "transform_rank": 10,
        "architecture": "naive",
        "bf16": False,
        "adapter": "lora",
        "lora_rank": 2,
        "lora_alpha": 2.0,
        "lora_dropout": 0.0,
        "lora_target": list(DEFAULT_ADAPTER_TARGETS),
        "adapter_init": "identity",
    }
    checkpoint = tmp_path / "adapter.safetensors"
    save_file(
        model.state_dict(), checkpoint,
        metadata={
            text_lm_train.TEXT_LM_CONFIG_METADATA_KEY: json.dumps(
                {"schema_version": 1, "args": config}, sort_keys=True,
            )
        },
    )
    output = tmp_path / "merged.safetensors"

    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        wrapped_output = model(input_ids)

    export_checkpoint(str(checkpoint), str(output))

    merged_model = TinyTextLM(
        vocab_size=31, max_seq_len=8, embed_dim=16, num_layers=2,
        num_heads=4, kv_heads=2, architecture="naive",
    )
    merged_model.load_state_dict(load_file(str(output)), strict=True)
    merged_model.eval()
    with torch.no_grad():
        merged_output = merged_model(input_ids)
    assert torch.allclose(wrapped_output, merged_output, atol=5e-5, rtol=1e-5)

    with safe_open(output, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        metadata = handle.metadata() or {}
    assert not any("lora_" in key for key in keys)
    saved = json.loads(
        metadata[text_lm_train.TEXT_LM_CONFIG_METADATA_KEY],
    )["args"]
    assert saved["adapter"] == "none"
    assert saved["merged_adapter"] is True
