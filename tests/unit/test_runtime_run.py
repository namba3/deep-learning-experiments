import json
import sys
from pathlib import Path

import pytest

from runtime.run import RunRecorder
from runtime.preflight import build_training_preflight
from runtime.validation import ValidationTimer, build_validation_report


def test_run_recorder_writes_resolved_config_and_jsonl_events(tmp_path: Path):
    recorder = RunRecorder(
        tmp_path / "output",
        script="mnist.train",
        config={"output_dir": tmp_path / "output", "shape": (28, 28)},
        run_id="test-run",
        command=["python3", "-m", "mnist.train"],
    )
    recorder.record("epoch", epoch=1, loss=0.25)
    recorder.finish(checkpoint="output/model.safetensors")

    config = json.loads(recorder.config_path.read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in recorder.events_path.read_text(encoding="utf-8").splitlines()
    ]

    assert config["run_id"] == "test-run"
    assert config["args"]["output_dir"] == str(tmp_path / "output")
    assert config["args"]["shape"] == [28, 28]
    assert recorder.checkpoints_dir.is_dir()
    assert recorder.artifacts_dir.is_dir()
    assert recorder.tensorboard_dir.is_dir()
    assert [event["event"] for event in events] == [
        "run_started",
        "epoch",
        "run_finished",
    ]
    assert events[-1]["status"] == "completed"


def test_run_recorder_writes_canonical_training_step_event(tmp_path: Path):
    recorder = RunRecorder(
        tmp_path,
        script="test.train",
        config={"epochs": 2},
        run_id="step-event",
    )

    payload = recorder.record_training_step(
        global_step=12,
        epoch=2,
        train_loss=0.5,
        eval_loss=0.4,
        effective_lr=1e-4,
        scheduled_lr=2e-4,
        step_time_sec=0.25,
        steps_per_second=4.0,
        samples_per_second=128.0,
        metrics={"train_accuracy": 0.9},
    )

    assert payload["event"] == "step"
    assert payload["step"] == 12
    assert payload["global_step"] == 12
    assert payload["loss"] == 0.5
    assert payload["eval_loss"] == 0.4
    assert payload["lr"] == 1e-4
    assert payload["train_accuracy"] == 0.9


def test_run_recorder_creates_unique_run_directory_by_default(tmp_path: Path):
    first = RunRecorder(tmp_path, script="test", config={})
    second = RunRecorder(tmp_path, script="test", config={})

    assert first.run_id != second.run_id
    assert first.config_path.exists()
    assert second.events_path.exists()


def test_run_recorder_sanitizes_named_run_directory(tmp_path: Path):
    recorder = RunRecorder(
        tmp_path,
        script="test.train",
        config={},
        run_name="  ablation / rank=4  ",
    )

    assert "test.train_ablation-rank-4_" in recorder.run_id
    config = json.loads(recorder.config_path.read_text(encoding="utf-8"))
    assert config["run_name"] == "ablation-rank-4"


def test_run_recorder_rejects_empty_named_run(tmp_path: Path):
    with pytest.raises(ValueError, match="run_name"):
        RunRecorder(tmp_path, script="test.train", config={}, run_name=" / ")


def test_run_recorder_records_failure_before_delegating_exception_hook(
    tmp_path: Path, monkeypatch
):
    delegated = []

    def previous_hook(*args):
        delegated.append(args)

    monkeypatch.setattr(sys, "excepthook", previous_hook)
    recorder = RunRecorder(tmp_path, script="test", config={})
    recorder.install_exception_hook()
    error = RuntimeError("training failed")

    sys.excepthook(type(error), error, None)

    events = [
        json.loads(line)
        for line in recorder.events_path.read_text(encoding="utf-8").splitlines()
    ]
    failure = events[-1]
    assert failure["event"] == "run_failed"
    assert failure["error_type"] == "RuntimeError"
    assert failure["error_message"] == "training failed"
    assert delegated and delegated[0][1] is error

    recorder.finish(status="failed")
    assert sys.excepthook is previous_hook


def test_training_preflight_returns_serializable_common_summary():
    summary = build_training_preflight(
        script="test.train",
        device="cpu",
        dtype="torch.float32",
        output_dir="output",
        epochs=2,
        batch_size=4,
        num_workers=0,
        seed=7,
        extra={"optimizer": "AdamW"},
    )

    assert summary["script"] == "test.train"
    assert summary["optimizer"] == "AdamW"
    assert summary["seed"] == 7


def test_validation_report_returns_common_model_and_dataset_summary():
    report = build_validation_report(
        script="test.train",
        device="cpu",
        dtype="torch.float32",
        train_examples=10,
        eval_examples=4,
        model_parameters=100,
        trainable_parameters=90,
        steps_per_epoch=5,
        measurements={"validation_seconds": 1.25},
        extra={"input_shape": [1, 28, 28]},
    )

    assert report["train_examples"] == 10
    assert report["eval_examples"] == 4
    assert report["trainable_parameters"] == 90
    assert report["validation_seconds"] == 1.25
    assert report["input_shape"] == [1, 28, 28]


def test_validation_timer_reports_cpu_wall_time_without_cuda_metrics():
    measurements = ValidationTimer("cpu").finish()

    assert measurements["validation_seconds"] >= 0
    assert measurements["peak_allocated_bytes"] is None
    assert measurements["peak_reserved_bytes"] is None
