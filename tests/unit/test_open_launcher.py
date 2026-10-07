"""Unit tests for the OS file launcher (task 45).

The OS is faked by monkeypatching ``sys.platform``, so every branch runs on
every platform; ``subprocess.Popen`` and ``os.startfile`` are mocked and never
executed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from mynotes.platform import open as open_module
from mynotes.platform.open import open_command, open_in_default_app


def test_macos_uses_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    assert open_command(Path("/tmp/a.pdf")) == ["open", "/tmp/a.pdf"]


def test_linux_uses_xdg_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert open_command(Path("/tmp/a.pdf")) == ["xdg-open", "/tmp/a.pdf"]


def test_windows_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    assert open_command(Path("C:/tmp/a.pdf")) is None


def test_posix_popen_is_detached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    seen: dict[str, object] = {}

    def fake_popen(command: list[str], **kwargs: object) -> None:
        seen["command"] = command
        seen["kwargs"] = kwargs

    monkeypatch.setattr(open_module.subprocess, "Popen", fake_popen)
    path = Path("/tmp/a.pdf")
    open_in_default_app(path)
    assert seen["command"] == ["xdg-open", str(path)]
    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["start_new_session"] is True
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL


def test_macos_popen_uses_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    seen: list[list[str]] = []
    monkeypatch.setattr(open_module.subprocess, "Popen", lambda command, **_: seen.append(command))
    open_in_default_app(Path("/tmp/a.pdf"))
    assert seen == [["open", "/tmp/a.pdf"]]


def test_windows_calls_startfile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    seen: list[str] = []
    monkeypatch.setattr(
        open_module.os, "startfile", lambda target: seen.append(target), raising=False
    )
    path = Path("C:/tmp/relatorio.docx")
    open_in_default_app(path)
    assert seen == [str(path)]


def test_missing_launcher_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    def raise_file_not_found(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(open_module.subprocess, "Popen", raise_file_not_found)
    with pytest.raises(RuntimeError, match="xdg-open"):
        open_in_default_app(Path("/tmp/a.pdf"))


def test_startfile_oserror_becomes_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    def raise_os_error(target: str) -> None:
        raise OSError("no association")

    monkeypatch.setattr(open_module.os, "startfile", raise_os_error, raising=False)
    with pytest.raises(RuntimeError, match="no association"):
        open_in_default_app(Path("C:/tmp/a.bin"))
