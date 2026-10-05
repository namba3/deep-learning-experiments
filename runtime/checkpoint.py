"""Operational checkpoint state shared by training entrypoints."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_STATE_VERSION = 1


def resume_state_path(checkpoint_path: str | Path) -> Path:
    """Return the sibling path used for non-model training state."""
    return Path(checkpoint_path).with_suffix(".resume.pt")


def capture_rng_state() -> dict[str, Any]:
    """Capture RNG state needed for a best-effort exact training resume."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        state["numpy"] = np.random.get_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`."""
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state:
        try:
            import numpy as np
        except ImportError:
            pass
        else:
            np.random.set_state(state["numpy"])


def make_training_state(
    *,
    optimizer,
    epoch: int,
    global_step: int,
    scheduler=None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a serializable state bundle excluding model weights."""
    state = {
        "state_version": CHECKPOINT_STATE_VERSION,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "rng": capture_rng_state(),
    }
    if extra:
        state["extra"] = extra
    return state


def save_training_state(checkpoint_path: str | Path, state: dict[str, Any]) -> Path:
    """Save training state next to a model-only checkpoint."""
    path = resume_state_path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return path


def load_training_state(checkpoint_path: str | Path) -> dict[str, Any] | None:
    """Load a sibling full-resume state, or return ``None`` for old exports."""
    path = resume_state_path(checkpoint_path)
    if not path.is_file():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid training state format: {path}")
    if state.get("state_version") != CHECKPOINT_STATE_VERSION:
        raise ValueError(
            f"Unsupported training state version {state.get('state_version')!r}: {path}"
        )
    return state
