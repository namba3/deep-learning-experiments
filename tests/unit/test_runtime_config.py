import json
from argparse import Namespace

import pytest
import torch
from safetensors.torch import save_file

from runtime.config import (
    CONFIG_SCHEMA_VERSION,
    apply_saved_config,
    checkpoint_config_metadata,
    cli_option_provided,
    read_checkpoint_config,
)


def test_cli_option_provided_supports_separate_and_equals_forms():
    assert cli_option_provided(["--lr", "1e-4"], "--lr")
    assert cli_option_provided(["--lr=1e-4"], "--lr")
    assert not cli_option_provided(["--learning-rate", "1e-4"], "--lr")


def test_apply_saved_config_restores_only_non_explicit_values():
    args = Namespace(optimizer="AdamW", lr=1e-4, amp="bf16")
    overridden = []
    restored = apply_saved_config(
        args,
        {"optimizer": "CAME", "lr": 2e-4, "amp": "no"},
        {"amp": ("--amp",)},
        argv=["--optimizer", "AdamW", "--amp=no"],
        overridden_keys=overridden,
    )

    assert restored == ["lr"]
    assert overridden == ["optimizer", "amp"]
    assert args.optimizer == "AdamW"
    assert args.lr == 2e-4
    assert args.amp == "bf16"


def test_apply_saved_config_recognizes_boolean_negative_alias():
    args = Namespace(compile=False)
    restored = apply_saved_config(
        args,
        {"compile": True},
        {"compile": ("--compile", "--no-compile")},
        argv=["--no-compile"],
    )

    assert restored == []
    assert args.compile is False


def test_checkpoint_config_metadata_serializes_namespace():
    metadata = checkpoint_config_metadata(
        Namespace(optimizer="AdamW", values=(1, 2)),
        "train.config",
    )

    payload = json.loads(metadata["train.config"])
    assert payload["schema_version"] == CONFIG_SCHEMA_VERSION
    assert payload["args"] == {"optimizer": "AdamW", "values": [1, 2]}


def test_read_checkpoint_config_accepts_versioned_and_legacy_metadata(tmp_path):
    versioned_path = tmp_path / "versioned.safetensors"
    save_file(
        {"weight": torch.ones(1)},
        versioned_path,
        metadata=checkpoint_config_metadata(
            Namespace(optimizer="AdamW"), "train.config",
        ),
    )
    assert read_checkpoint_config(versioned_path, "train.config") == {
        "optimizer": "AdamW",
    }

    legacy_path = tmp_path / "legacy.safetensors"
    save_file(
        {"weight": torch.ones(1)},
        legacy_path,
        metadata={"train.config": '{"optimizer": "CAME"}'},
    )
    assert read_checkpoint_config(legacy_path, "train.config") == {
        "optimizer": "CAME",
    }


def test_read_checkpoint_config_rejects_unknown_future_schema(tmp_path):
    path = tmp_path / "future.safetensors"
    save_file(
        {"weight": torch.ones(1)},
        path,
        metadata={
            "train.config": json.dumps({
                "schema_version": CONFIG_SCHEMA_VERSION + 1,
                "args": {},
            }),
        },
    )

    with pytest.raises(ValueError, match="unsupported schema version"):
        read_checkpoint_config(path, "train.config")
