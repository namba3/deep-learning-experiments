import sys
import json

import pytest

from text_lm import train as text_lm_train
from cifar10 import train as cifar10_train
from image_ae import train as image_ae_train
from image_gen import train as image_gen_train
from mnist import train as mnist_train


@pytest.mark.parametrize(
    ("module", "entrypoint_args"),
    [
        (mnist_train, []),
        (cifar10_train, []),
        (text_lm_train, []),
        (image_ae_train, []),
        (image_gen_train, ["--vae-model", "dummy"]),
    ],
    ids=["mnist", "cifar10", "text_lm", "image_ae", "image_gen"],
)
def test_train_entrypoint_accepts_common_cli_and_dry_runs(
    module, entrypoint_args, tmp_path, monkeypatch
):
    output_dir = tmp_path / module.__name__.split(".")[0]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            f"{module.__name__}.train",
            "--output-dir",
            str(output_dir),
            "--run-name",
            "smoke-test",
            "--device",
            "cpu",
            "--dry-run",
            "--optimizer",
            "AdamW",
            "--lr-scheduler",
            "cosine",
            "--warmup-steps",
            "3",
            *entrypoint_args,
        ],
    )

    module.main()

    run_dirs = list((output_dir / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert "smoke-test" in run_dir.name
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["script"] == module.__name__
    assert config["run_name"] == "smoke-test"
    assert config["args"]["optimizer"] == "AdamW"
    assert config["args"]["lr_scheduler"] == "cosine"
    assert config["args"]["warmup_steps"] == 3
    assert config["args"]["device"] == "cpu"
    if module is text_lm_train:
        assert config["args"]["data_mode"] == "text"
        assert config["args"]["dataset_name"] == "roneneldan/TinyStories"
        assert config["args"]["tokenizer"] == "Qwen/Qwen3.5-0.8B"


@pytest.mark.parametrize(
    ("module", "entrypoint_args"),
    [
        (mnist_train, []),
        (cifar10_train, []),
        (text_lm_train, []),
        (image_ae_train, []),
        (image_gen_train, ["--vae-model", "dummy"]),
    ],
    ids=["mnist", "cifar10", "text_lm", "image_ae", "image_gen"],
)
def test_train_rejects_resume_and_init_checkpoint_together(
    module, entrypoint_args, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            f"{module.__name__}.train",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "resume-conflict",
            "--resume",
            "resume.safetensors",
            "--init-checkpoint",
            "init.safetensors",
            *entrypoint_args,
        ],
    )

    with pytest.raises(ValueError, match="--resume and --init-checkpoint"):
        module.main()
