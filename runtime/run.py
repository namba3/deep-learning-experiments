"""Small, dependency-free run metadata and event logging helpers."""

from __future__ import annotations

import json
import re
import sys
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    """Convert common CLI/config values into stable JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (AttributeError, TypeError, ValueError):
            pass
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_run_name(value: str) -> str:
    """Convert a user-provided run name into one safe path component."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    normalized = normalized.strip("-._")
    if not normalized:
        raise ValueError("run_name must contain at least one usable character")
    return normalized


class RunRecorder:
    """Write one resolved config and append-only JSONL events for a run.

    The recorder intentionally lives alongside an existing experiment output
    directory. It does not move model checkpoints or change their names.
    """

    schema_version = 1

    def __init__(
        self,
        output_dir: str | Path,
        *,
        script: str,
        config: Mapping[str, Any],
        run_id: str | None = None,
        run_name: str | None = None,
        command: list[str] | None = None,
    ) -> None:
        output_path = Path(output_dir)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_script = script.replace("/", ".").replace("\\", ".")
        safe_name = _safe_run_name(run_name) if run_name is not None else None
        generated_id = f"{safe_script}_{timestamp}_{uuid.uuid4().hex[:8]}"
        if safe_name is not None:
            generated_id = f"{safe_script}_{safe_name}_{timestamp}_{uuid.uuid4().hex[:8]}"
        self.run_id = run_id or generated_id
        self.run_dir = output_path / "runs" / self.run_id
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.artifacts_dir = self.run_dir / "artifacts"
        self.tensorboard_dir = self.run_dir / "tensorboard"
        self.config_path = self.run_dir / "config.json"
        self.events_path = self.run_dir / "metrics.jsonl"
        self.progress_path = self.run_dir / "progress.txt"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for directory in (
            self.checkpoints_dir,
            self.artifacts_dir,
            self.tensorboard_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._previous_excepthook = None
        self._exception_hook = None

        config_payload = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_name": safe_name,
            "script": script,
            "started_at": _utc_now(),
            "command": list(sys.argv) if command is None else command,
            "args": _jsonable(config),
        }
        self.config_path.write_text(
            json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.record(
            "run_started",
            config_path=str(self.config_path),
        )

    def record(self, event: str, **values: Any) -> dict[str, Any]:
        """Append one flushed event and return the serialized event payload."""
        payload = {
            "schema_version": self.schema_version,
            "event": event,
            "run_id": self.run_id,
            "timestamp": _utc_now(),
            **_jsonable(values),
        }
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            stream.flush()
        return payload

    def write_progress(self, global_step: int, **fields: Any) -> None:
        """Atomically replace the human-readable progress snapshot."""
        lines = [f"global_step={int(global_step)}"]
        for name, value in fields.items():
            if value is None:
                continue
            display_value = str(value).replace("\r", " ").replace("\n", " ")
            lines.append(f"{name}={display_value}")
        temporary_path = self.progress_path.with_name(f".{self.progress_path.name}.tmp")
        temporary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary_path.replace(self.progress_path)

    def record_training_step(
        self,
        *,
        global_step: int,
        epoch: int | None = None,
        train_loss: float | None = None,
        eval_loss: float | None = None,
        effective_lr: float | None = None,
        scheduled_lr: float | None = None,
        step_time_sec: float | None = None,
        steps_per_second: float | None = None,
        samples_per_second: float | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a common epoch-end training summary as a ``step`` event.

        ``global_step`` is the optimizer-step coordinate. The event is
        intentionally emitted at epoch boundaries so metrics.jsonl remains
        useful without adding one filesystem write per minibatch.
        """
        values: dict[str, Any] = {
            "step": int(global_step),
            "global_step": int(global_step),
        }
        optional_values = {
            "epoch": epoch,
            "loss": train_loss,
            "eval_loss": eval_loss,
            "lr": effective_lr,
            "scheduled_lr": scheduled_lr,
            "step_time_sec": step_time_sec,
            "steps_per_second": steps_per_second,
            "samples_per_second": samples_per_second,
        }
        values.update({
            key: value
            for key, value in optional_values.items()
            if value is not None
        })
        if metrics:
            values.update({str(key): value for key, value in metrics.items()})
        return self.record("step", **values)

    def record_failure(self, error: BaseException) -> None:
        """Record an uncaught training error without swallowing it."""
        self.record(
            "run_failed",
            status="failed",
            error_type=type(error).__name__,
            error_message=str(error),
        )

    def install_exception_hook(self) -> None:
        """Record uncaught exceptions before delegating to Python's hook."""
        if self._exception_hook is not None:
            return
        previous_hook = sys.excepthook

        def exception_hook(
            exception_type: type[BaseException],
            exception: BaseException,
            traceback,
        ) -> None:
            try:
                self.record_failure(exception)
            finally:
                previous_hook(exception_type, exception, traceback)

        self._previous_excepthook = previous_hook
        self._exception_hook = exception_hook
        sys.excepthook = exception_hook

    def uninstall_exception_hook(self) -> None:
        """Restore the previous exception hook after a terminal run event."""
        if self._exception_hook is None:
            return
        if sys.excepthook is self._exception_hook and self._previous_excepthook is not None:
            sys.excepthook = self._previous_excepthook
        self._previous_excepthook = None
        self._exception_hook = None

    def finish(self, *, status: str = "completed", **values: Any) -> None:
        """Record the terminal status of a normally completed run."""
        try:
            self.record("run_finished", status=status, **values)
        finally:
            self.uninstall_exception_hook()
