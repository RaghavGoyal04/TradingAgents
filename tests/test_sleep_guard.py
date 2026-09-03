"""macOS sleep guard: platform gating and process-watch wiring."""


import pytest

from tradingagents.portfolio import sleep_guard


@pytest.mark.unit
def test_unsupported_platform_returns_none(monkeypatch):
    monkeypatch.setattr(sleep_guard.sys, "platform", "linux")
    assert sleep_guard.is_supported() is False
    assert sleep_guard.start(1234) is None


@pytest.mark.unit
def test_start_watches_pid_when_supported(monkeypatch):
    monkeypatch.setattr(sleep_guard, "is_supported", lambda: True)
    captured = {}

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd

        def poll(self):
            return 0

    monkeypatch.setattr(sleep_guard.subprocess, "Popen", FakePopen)
    sleep_guard.start(4321)
    assert captured["cmd"] == ["caffeinate", "-dimsu", "-w", "4321"]


@pytest.mark.unit
def test_keep_awake_noops_when_unsupported(monkeypatch):
    monkeypatch.setattr(sleep_guard, "is_supported", lambda: False)
    with sleep_guard.keep_awake() as guard:
        assert guard is None


@pytest.mark.unit
def test_keep_awake_terminates_guard(monkeypatch):
    monkeypatch.setattr(sleep_guard, "is_supported", lambda: True)
    events = []

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            pass

        def poll(self):
            return None  # still running

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(sleep_guard.subprocess, "Popen", FakePopen)
    with sleep_guard.keep_awake():
        pass
    assert events == ["terminate"]
