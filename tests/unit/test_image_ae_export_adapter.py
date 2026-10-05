import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from core.low_rank import inject_lora
from image_ae import export_adapter
from image_ae.train import (
    TRAINING_CONFIG_METADATA_KEY,
    checkpoint_metadata,
    parse_args,
)


class TinyModel(torch.nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)


def _metadata(*, adapter: str = "lora", rank: int = 1):
    previous_argv = sys.argv
    sys.argv = ["image_ae.train"]
    try:
        args = parse_args()
    finally:
        sys.argv = previous_argv
    args.adapter = adapter
    args.lora_rank = rank
    args.lora_alpha = 1.0 if rank else None
    args.lora_target = [r"^linear$"] if rank else None
    return checkpoint_metadata(args)


def test_export_checkpoint_materializes_image_ae_adapter(tmp_path, monkeypatch):
    torch.manual_seed(0)
    model = TinyModel()
    inject_lora(model, [r"^linear$"], rank=1)
    with torch.no_grad():
        model.linear.lora_A.fill_(0.5)
        model.linear.lora_B.fill_(0.25)
    checkpoint = tmp_path / "adapter.safetensors"
    save_file(model.state_dict(), checkpoint, metadata=_metadata())
    output = tmp_path / "merged.safetensors"

    monkeypatch.setattr(export_adapter, "_build_model", lambda *_args: TinyModel())
    export_adapter.export_checkpoint(str(checkpoint), str(output))

    with safe_open(output, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        metadata = handle.metadata() or {}
    assert "linear.weight" in keys
    assert not any("lora_" in key for key in keys)
    assert metadata[TRAINING_CONFIG_METADATA_KEY]
    saved = export_adapter.read_checkpoint_config(
        str(output), TRAINING_CONFIG_METADATA_KEY,
    )
    assert saved is not None
    assert saved["adapter"] == "none"
    assert saved["merged_adapter"] is True


def test_export_rejects_plain_image_ae_checkpoint(tmp_path):
    checkpoint = tmp_path / "plain.safetensors"
    save_file(
        {"weight": torch.ones(1)},
        checkpoint,
        metadata=_metadata(adapter="none", rank=0),
    )
    with pytest.raises(ValueError, match="does not contain"):
        export_adapter.export_checkpoint(
            str(checkpoint), str(tmp_path / "out.safetensors"),
        )
