"""macOS sleep prevention while a run is active.

Wraps Apple's ``caffeinate``. The guard prevents *idle* sleep so a long run
isn't interrupted when the machine is left alone. It cannot keep the machine
awake when the lid is closed unless macOS clamshell conditions are met (external
power + display/input); that limitation is intentional and documented, and the
orchestrator's resume path is what guarantees progress across a lid-close.

``caffeinate -w <pid>`` ties the assertion's lifetime to the watched process, so
if the orchestrator dies the assertion is released automatically -- no leaked
process keeping the Mac awake forever.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager


def is_supported() -> bool:
    """True only on macOS with ``caffeinate`` available."""
    return sys.platform == "darwin" and shutil.which("caffeinate") is not None


def start(watch_pid: int) -> subprocess.Popen | None:
    """Start ``caffeinate -dimsu -w <watch_pid>``; return the process or None.

    Returns ``None`` on unsupported platforms so callers can no-op cleanly. The
    ``-w`` flag makes caffeinate exit when the watched PID exits.
    """
    if not is_supported():
        return None
    return subprocess.Popen(
        ["caffeinate", "-dimsu", "-w", str(watch_pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@contextmanager
def keep_awake() -> Iterator[subprocess.Popen | None]:
    """Keep the machine awake for the duration of the block (best effort).

    Watches the current process, so the assertion is released when this process
    exits even if the context manager's ``finally`` is skipped by a hard kill.
    """
    import os

    guard = start(os.getpid())
    try:
        yield guard
    finally:
        if guard is not None and guard.poll() is None:
            guard.terminate()
            try:
                guard.wait(timeout=5)
            except subprocess.TimeoutExpired:
                guard.kill()
