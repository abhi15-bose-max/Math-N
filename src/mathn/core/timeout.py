"""
A small SIGALRM-based timeout utility.

V1 runs everything in a single process on a single thread, so we use
``signal.alarm`` rather than pulling in multiprocessing. This only works
on Unix and only on the main thread -- both are true for the CLI /
experiment scripts this framework ships with. If ported to a context
without SIGALRM (e.g. Windows, or a worker thread), swap this for a
process- or thread-based watchdog; the call sites only depend on
``run_with_timeout`` raising ``TimeoutError``.
"""

from __future__ import annotations

import signal
from typing import Any, Callable


class TimeoutError_(TimeoutError):
    pass


def run_with_timeout(func: Callable, args: tuple = (), kwargs: dict = None, timeout_seconds: float = 30.0) -> Any:
    kwargs = kwargs or {}

    if timeout_seconds is None or timeout_seconds <= 0 or not hasattr(signal, "SIGALRM"):
        # No timeout enforcement available/requested -- just call directly.
        return func(*args, **kwargs)

    def _handler(signum, frame):
        raise TimeoutError(f"Operation exceeded {timeout_seconds}s timeout")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    old_alarm = signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return func(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
