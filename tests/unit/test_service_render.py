"""Unit tests for the service installers (task 49).

Every backend runs against a fixed :class:`ServiceSpec` built from fake
paths; ``subprocess.run`` is mocked everywhere and ``Path.home()`` is
monkeypatched, so nothing touches the real system and the golden files stay
machine-independent.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from mynotes.platform import service as service_module
from mynotes.platform.service import ServiceError, ServiceSpec, backend_for_platform, default_spec
from mynotes.platform.service.launchd import LaunchdBackend, tcc_protected_roots, tcc_steps
from mynotes.platform.service.systemd import SystemdBackend, render_unit
from mynotes.platform.service.windows_task import WindowsTaskBackend, render_script

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "service"
FAKE_HOME = Path("/home/test")

POSIX_SPEC = ServiceSpec(
    name="mynotes-worker",
    exec_args=[
        "/home/test/.local/bin/uv",
        "run",
        "--directory",
        "/home/test/project",
        "mynotes",
        "worker",
    ],
    working_dir=Path("/home/test/project"),
    log_dir=Path("/home/test/Library/Logs/mynotes"),
    env={"MYNOTES_CONFIG": "/home/test/mynotes.yaml"},
)

WINDOWS_SPEC = ServiceSpec(
    name="mynotes-worker",
    exec_args=["C:\\proj\\.venv\\Scripts\\pythonw.exe", "-m", "mynotes", "worker"],
    working_dir=Path("C:\\proj"),
    log_dir=Path("C:\\Users\\test\\AppData\\Local\\mynotes\\Logs"),
    env={"MYNOTES_CONFIG": "C:\\Users\\test\\mynotes.yaml"},
)


@pytest.fixture(autouse=True)
def _fake_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin ``Path.home()`` so rendered paths match the golden files."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: FAKE_HOME))


def _fake_run(monkeypatch: pytest.MonkeyPatch, fail: bool = False) -> list[list[str]]:
    """Replace ``subprocess.run`` with a recorder; return the call list."""
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1 if fail else 0, "", "boom" if fail else "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _golden(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# --- systemd -----------------------------------------------------------------


def test_systemd_render_matches_golden() -> None:
    rendered = SystemdBackend().render(POSIX_SPEC)
    unit = "/home/test/.config/systemd/user/mynotes-worker.service"
    assert list(rendered) == [unit]
    assert rendered[unit] == _golden("systemd.service")


def test_systemd_render_quotes_paths_with_spaces() -> None:
    spec = replace(
        POSIX_SPEC,
        working_dir=Path("/home/test/my project"),
        exec_args=["/home/test/my uv/uv", "run", "--directory", "/home/test/my project"],
    )
    text = render_unit(spec)
    assert 'WorkingDirectory="/home/test/my project"' in text
    assert 'ExecStart="/home/test/my uv/uv" run --directory "/home/test/my project"' in text


def test_systemd_install_writes_unit_and_enables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    calls = _fake_run(monkeypatch)
    backend = SystemdBackend()
    backend.install(POSIX_SPEC)

    unit = tmp_path / ".config" / "systemd" / "user" / "mynotes-worker.service"
    assert unit.is_file()
    assert unit.read_text(encoding="utf-8") == _golden("systemd.service")
    assert calls == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", "mynotes-worker"],
    ]


def test_systemd_install_fails_loudly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    _fake_run(monkeypatch, fail=True)
    with pytest.raises(ServiceError, match="daemon-reload failed"):
        SystemdBackend().install(POSIX_SPEC)


def test_systemd_uninstall_disables_and_removes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    unit = tmp_path / ".config" / "systemd" / "user" / "mynotes-worker.service"
    unit.parent.mkdir(parents=True)
    unit.write_text("[Unit]\n", encoding="utf-8")
    calls = _fake_run(monkeypatch)

    SystemdBackend().uninstall(POSIX_SPEC)

    assert not unit.exists()
    assert calls == [
        ["systemctl", "--user", "disable", "--now", "mynotes-worker"],
        ["systemctl", "--user", "daemon-reload"],
    ]


def test_systemd_status_and_logs_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, "active\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = SystemdBackend()
    assert backend.status(POSIX_SPEC) == "active"
    assert backend.logs_hint(POSIX_SPEC) == "journalctl --user -u mynotes-worker -f"
    assert any("loginctl enable-linger" in hint for hint in backend.hints(POSIX_SPEC))


# --- launchd -----------------------------------------------------------------


def test_launchd_render_matches_golden() -> None:
    rendered = LaunchdBackend().render(POSIX_SPEC)
    plist = "/home/test/Library/LaunchAgents/com.mynotes.worker.plist"
    assert list(rendered) == [plist]
    assert rendered[plist] == _golden("launchd.plist")
    payload = plistlib.loads(rendered[plist].encode("utf-8"))
    assert payload["Label"] == "com.mynotes.worker"
    assert payload["RunAtLoad"] is True
    assert payload["KeepAlive"] is True


def test_tcc_flags_root_under_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: FAKE_HOME))
    protected = tcc_protected_roots(
        [FAKE_HOME / "Documents" / "notes", FAKE_HOME / "work" / "notes"]
    )
    assert protected == [FAKE_HOME / "Documents" / "notes"]


def test_tcc_flags_icloud_drive_root() -> None:
    protected = tcc_protected_roots(
        [FAKE_HOME / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Notes"]
    )
    assert protected == [
        FAKE_HOME / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Notes"
    ]


def test_tcc_steps_mention_uv_and_python() -> None:
    lines = tcc_steps([FAKE_HOME / "Documents"], Path("/usr/local/bin/uv"), "/py/bin/python")
    joined = "\n".join(lines)
    assert "/usr/local/bin/uv" in joined
    assert "/py/bin/python" in joined
    assert "Full Disk Access" in joined


def test_launchd_install_bootstraps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    calls = _fake_run(monkeypatch)
    backend = LaunchdBackend()
    backend.install(POSIX_SPEC)

    plist = tmp_path / "Library" / "LaunchAgents" / "com.mynotes.worker.plist"
    assert plist.is_file()
    assert calls == [
        ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist)],
    ]


def test_launchd_install_falls_back_to_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        code = 1 if cmd[1] == "bootstrap" else 0
        return subprocess.CompletedProcess(cmd, code, "", "Bootstrap failed")

    monkeypatch.setattr(subprocess, "run", fake_run)
    LaunchdBackend().install(POSIX_SPEC)

    assert calls[0][1] == "bootstrap"
    assert calls[1] == ["launchctl", "load", "-w", str(calls[0][3])]


def test_launchd_install_raises_when_both_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    _fake_run(monkeypatch, fail=True)
    with pytest.raises(ServiceError, match="bootstrap failed"):
        LaunchdBackend().install(POSIX_SPEC)


def test_launchd_uninstall_bootouts_and_deletes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    plist = tmp_path / "Library" / "LaunchAgents" / "com.mynotes.worker.plist"
    plist.parent.mkdir(parents=True)
    plist.write_text("<plist/>", encoding="utf-8")
    calls = _fake_run(monkeypatch)

    LaunchdBackend().uninstall(POSIX_SPEC)

    assert not plist.exists()
    assert calls == [
        ["launchctl", "bootout", f"gui/{os.getuid()}/com.mynotes.worker"],
    ]


def test_launchd_uninstall_tolerates_failed_bootout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, "", "not loaded")

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = LaunchdBackend()
    backend.plist_path(POSIX_SPEC).parent.mkdir(parents=True)
    backend.plist_path(POSIX_SPEC).write_text("<plist/>", encoding="utf-8")

    backend.uninstall(POSIX_SPEC)

    assert not backend.plist_path(POSIX_SPEC).exists()
    assert len(calls) == 1


def test_launchd_status_and_logs_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        state = "\t\tstate = running\n" if cmd[1] == "print" else ""
        return subprocess.CompletedProcess(cmd, 0, state, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = LaunchdBackend()
    assert backend.status(POSIX_SPEC) == "running"
    assert backend.logs_hint(POSIX_SPEC) == "tail -f /home/test/Library/Logs/mynotes/worker.log"


def test_launchd_status_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 3, "", "Could not find service")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert LaunchdBackend().status(POSIX_SPEC) == "not installed"


# --- Windows Task Scheduler --------------------------------------------------


def test_windows_render_matches_golden() -> None:
    rendered = WindowsTaskBackend().render(WINDOWS_SPEC)
    key, content = next(iter(rendered.items()))
    assert key == "powershell -NoProfile -Command <script>"
    assert content == _golden("windows_task.ps1")


def test_windows_script_defaults_the_log_file() -> None:
    spec = replace(WINDOWS_SPEC, env={})
    script = render_script(spec)
    assert "MYNOTES_LOG_FILE" in script
    assert "worker.log" in script


def test_windows_install_runs_powershell(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_run(monkeypatch)
    WindowsTaskBackend().install(WINDOWS_SPEC)

    assert len(calls) == 1
    assert calls[0][:3] == ["powershell", "-NoProfile", "-Command"]
    assert "Register-ScheduledTask -TaskName 'mynotes-worker'" in calls[0][3]
    assert "New-ScheduledTaskAction -Execute 'C:\\proj\\.venv\\Scripts\\pythonw.exe'" in calls[0][3]
    assert "-RestartCount 3" in calls[0][3]


def test_windows_install_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_run(monkeypatch, fail=True)
    with pytest.raises(ServiceError, match="Register-ScheduledTask failed"):
        WindowsTaskBackend().install(WINDOWS_SPEC)


def test_windows_uninstall_unregisters(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_run(monkeypatch)
    WindowsTaskBackend().uninstall(WINDOWS_SPEC)

    assert calls == [
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Unregister-ScheduledTask -Confirm:$false -TaskName 'mynotes-worker'",
        ],
    ]


def test_windows_status_reports_state(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, "\nState\n-----\nRunning\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = WindowsTaskBackend()
    assert "Running" in backend.status(WINDOWS_SPEC)
    assert "Get-Content -Wait" in backend.logs_hint(WINDOWS_SPEC)
    assert backend.hints(WINDOWS_SPEC) == []


# --- common interface --------------------------------------------------------


@pytest.mark.parametrize(
    ("platform_name", "backend_type"),
    [
        ("linux", "SystemdBackend"),
        ("darwin", "LaunchdBackend"),
        ("win32", "WindowsTaskBackend"),
    ],
)
def test_backend_for_platform(
    monkeypatch: pytest.MonkeyPatch, platform_name: str, backend_type: str
) -> None:
    monkeypatch.setattr(sys, "platform", platform_name)
    assert type(backend_for_platform()).__name__ == backend_type


def test_backend_for_platform_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "sunos5")
    with pytest.raises(ServiceError, match="unsupported platform"):
        backend_for_platform()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows uses pythonw.exe instead of uv")
def test_default_spec_requires_uv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(service_module, "_project_dir", lambda: tmp_path)
    with pytest.raises(ServiceError, match="uv was not found on PATH"):
        default_spec()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows uses pythonw.exe instead of uv")
def test_default_spec_posix_builds_uv_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(service_module, "_project_dir", lambda: tmp_path)
    monkeypatch.delenv("MYNOTES_CONFIG", raising=False)

    spec = default_spec()

    assert spec.name == "mynotes-worker"
    assert spec.exec_args == [
        "/usr/bin/uv",
        "run",
        "--directory",
        str(tmp_path),
        "mynotes",
        "worker",
    ]
    assert spec.env == {}
    assert Path(spec.exec_args[0]).is_absolute()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows uses pythonw.exe instead of uv")
def test_default_spec_carries_config_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(service_module, "_project_dir", lambda: tmp_path)
    monkeypatch.setenv("MYNOTES_CONFIG", "/home/test/mynotes.yaml")

    spec = default_spec()

    assert spec.env == {"MYNOTES_CONFIG": "/home/test/mynotes.yaml"}
    assert not any(key.lower().endswith(("key", "token", "password")) for key in spec.env)


def test_default_spec_windows_needs_pythonw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(service_module, "_project_dir", lambda: Path("C:/proj"))
    with pytest.raises(ServiceError, match=r"pythonw\.exe not found"):
        default_spec()


def test_default_spec_windows_uses_pythonw(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    venv_scripts = tmp_path / ".venv" / "Scripts"
    venv_scripts.mkdir(parents=True)
    (venv_scripts / "pythonw.exe").write_text("", encoding="utf-8")
    monkeypatch.setattr(service_module, "_project_dir", lambda: tmp_path)

    spec = default_spec()

    assert spec.exec_args == [str(venv_scripts / "pythonw.exe"), "-m", "mynotes", "worker"]
    assert spec.env["MYNOTES_LOG_FILE"] == str(spec.log_dir / "worker.log")


def test_project_dir_is_found_from_pyproject(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parent
    while not (repo_root / "pyproject.toml").is_file():
        repo_root = repo_root.parent
    assert service_module._project_dir() == repo_root


# --- components (worker / web) -----------------------------------------------


def test_default_spec_web_component_runs_serve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(service_module.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(service_module, "_project_dir", lambda: tmp_path)
    monkeypatch.delenv("MYNOTES_CONFIG", raising=False)
    spec = default_spec("web")
    assert spec.name == "mynotes-web"
    assert spec.component == "web"
    assert spec.exec_args[-2:] == ["mynotes", "serve"]
    assert spec.log_name == "web.log"
    assert "web interface" in spec.description
    with pytest.raises(ServiceError):
        default_spec("database")


def test_web_component_renders_its_own_unit_plist_and_task() -> None:
    web = replace(
        POSIX_SPEC,
        name="mynotes-web",
        exec_args=[*POSIX_SPEC.exec_args[:-1], "serve"],
        component="web",
        description="mynotes web interface and API (serve)",
    )
    unit = SystemdBackend().render(web)
    assert list(unit) == ["/home/test/.config/systemd/user/mynotes-web.service"]
    assert "Description=mynotes web interface and API (serve)" in next(iter(unit.values()))
    assert "mynotes serve" in next(iter(unit.values()))
    backend = LaunchdBackend()
    assert backend.label(web) == "com.mynotes.web"
    assert backend.log_file(web) == web.log_dir / "web.log"
    script = render_script(web)
    assert "runs the mynotes web at logon" in script
    assert "web.log" in script and "mynotes-web" in script
