from runtime.progress import RichProgress
from runtime.metrics import (
    build_standard_progress_rows,
    write_standard_training_metrics,
)


def test_rich_progress_supports_desc_alias_and_multiline_status():
    progress = RichProgress(total=2, desc="test")
    try:
        progress.set_status(loss="waiting", metrics="-")
        with progress:
            progress.update(advance=1)
            progress.set_status({"loss": "0.5", "metrics": "ok"})

        assert progress.progress.tasks[0].completed == 1
        assert progress._status is not None
    finally:
        progress.close()


def test_rich_progress_iterable_advances_once_per_item():
    progress = RichProgress(["a", "b"], description="test")
    assert list(progress) == ["a", "b"]
    assert progress.progress.tasks[0].completed == 2


def test_rich_progress_set_postfix_keeps_one_line_compatibility():
    progress = RichProgress(total=1, description="test")
    try:
        progress.set_postfix(loss="0.25", acc="90%")
        assert progress.progress.tasks[0].fields["postfix"] == "loss=0.25 acc=90%"
    finally:
        progress.close()
class RecordingWriter:
    def __init__(self):
        self.values = []

    def add_scalar(self, name, value, step):
        self.values.append((name, value, step))


def test_standard_progress_rows_have_stable_common_keys():
    rows = build_standard_progress_rows(
        step=3,
        total_steps=10,
        global_step=17,
        loss="0.42",
        learning_rate="1e-4",
        step_seconds="0.2s",
        extra={"accuracy": "95%"},
    )

    assert rows == {
        "step": "3/10",
        "global_step": "17",
        "loss": "0.42",
        "lr": "1e-4",
        "step_time": "0.2s",
        "accuracy": "95%",
    }


def test_standard_training_metrics_write_canonical_tags():
    writer = RecordingWriter()
    write_standard_training_metrics(
        writer,
        step=2,
        train_loss=0.5,
        eval_loss=0.4,
        learning_rate=1e-4,
        scheduled_learning_rate=2e-4,
        steps_per_second=3.0,
        extra={"train/accuracy": 0.9},
    )

    assert writer.values == [
        ("train/loss/total", 0.5, 2),
        ("eval/loss/total", 0.4, 2),
        ("train/lr/effective", 1e-4, 2),
        ("train/lr/scheduled", 2e-4, 2),
        ("train/performance/steps_per_second", 3.0, 2),
        ("train/accuracy", 0.9, 2),
    ]
