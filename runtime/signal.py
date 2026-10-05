"""Shared graceful SIGINT handling for long-running training jobs."""

from __future__ import annotations

import signal
from types import FrameType


class GracefulStop:
    """Turn the first Ctrl-C into a cooperative stop request.

    A second Ctrl-C keeps the normal ``KeyboardInterrupt`` behavior so a
    stuck or unresponsive training step can still be aborted immediately.
    """

    def __init__(self, message: str = "Ctrl-C received; stopping safely") -> None:
        self.message = message
        self.requested = False
        self._previous_handler: signal.Handlers | None = None
        self._installed = False

    def install(self) -> None:
        """Install the controller as the process SIGINT handler."""
        if self._installed:
            return
        self._previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_signal)
        self._installed = True

    def request(self) -> None:
        """Request a cooperative stop without emitting a signal."""
        if not self.requested:
            print(f"\n{self.message}")
        self.requested = True

    def _handle_signal(
        self,
        _signum: int,
        _frame: FrameType | None,
    ) -> None:
        if self.requested:
            raise KeyboardInterrupt
        self.request()

    def restore(self) -> None:
        """Restore the SIGINT handler that was active before installation."""
        if not self._installed:
            return
        assert self._previous_handler is not None
        signal.signal(signal.SIGINT, self._previous_handler)
        self._previous_handler = None
        self._installed = False

    def __enter__(self) -> "GracefulStop":
        self.install()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: object,
    ) -> None:
        self.restore()
