import signal

import pytest

from runtime.signal import GracefulStop


def test_graceful_stop_requests_once_and_second_signal_is_immediate():
    controller = GracefulStop(message="stop requested")

    controller.request()
    assert controller.requested

    with pytest.raises(KeyboardInterrupt):
        controller._handle_signal(signal.SIGINT, None)


def test_graceful_stop_restores_previous_handler():
    previous = signal.getsignal(signal.SIGINT)
    controller = GracefulStop()
    controller.install()

    assert signal.getsignal(signal.SIGINT) == controller._handle_signal

    controller.restore()
    assert signal.getsignal(signal.SIGINT) == previous
