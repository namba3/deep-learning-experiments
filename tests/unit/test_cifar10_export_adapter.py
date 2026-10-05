from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from cifar10 import export_adapter
from cifar10.train import CIFAR10_CONFIG_METADATA_KEY
from core.low_rank import inject_lora
from runtime.config import checkpoint_config_metadata


class TinyModel(torch.nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, images):
        return self.linear(images)


def test_export_checkpoint_materializes_adapter_and_writes_plain_metadata(
    tmp_path, monkeypatch,
):
    torch.manual_seed(0)
    base = TinyModel()
    inject_lora(base, [r"^linear$"], rank=1)
    with torch.no_grad():
        base.linear.lora_A.fill_(0.5)
        base.linear.lora_B.fill_(0.25)
    config = SimpleNamespace(
        adapter="lora",
        lora_rank=1,
        lora_alpha=1.0,
        lora_dropout=0.0,
        lora_target=[r"^linear$"],
        adapter_init="identity",
        bf16=False,
        attention_type="full",
        window_size=4,
        mhla_block_size=None,
        mhla_backend="auto",
    )
    checkpoint = tmp_path / "adapter.safetensors"
    save_file(
        base.state_dict(), checkpoint,
        metadata=checkpoint_config_metadata(config, CIFAR10_CONFIG_METADATA_KEY),
    )
    output = tmp_path / "merged.safetensors"
    monkeypatch.setattr(export_adapter, "CIFAR10ViT", TinyModel)
    monkeypatch.setattr(export_adapter, "PATCH_SIZE", 2)
    monkeypatch.setattr(export_adapter, "EMBED_DIM", 128)
    monkeypatch.setattr(export_adapter, "NUM_LAYERS", 3)
    monkeypatch.setattr(export_adapter, "NUM_HEADS", 8)

    export_adapter.export_checkpoint(str(checkpoint), str(output))

    with safe_open(output, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        keys = set(handle.keys())
    assert "linear.weight" in keys
    assert not any("lora_" in key for key in keys)
    saved_config = export_adapter.read_checkpoint_config(
        str(output), CIFAR10_CONFIG_METADATA_KEY,
    )
    assert saved_config is not None
    assert saved_config["adapter"] == "none"
    assert saved_config["merged_adapter"] is True
    assert metadata is not None


def test_export_rejects_plain_checkpoint(tmp_path):
    path = tmp_path / "plain.safetensors"
    save_file(
        {"weight": torch.ones(1)}, path,
        metadata=checkpoint_config_metadata(
            SimpleNamespace(adapter="none", lora_rank=0),
            CIFAR10_CONFIG_METADATA_KEY,
        ),
    )
    with pytest.raises(ValueError, match="does not contain"):
        export_adapter.export_checkpoint(str(path), str(tmp_path / "out.safetensors"))
