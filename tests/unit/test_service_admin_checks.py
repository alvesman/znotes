"""Admin-only service entry points refuse non-admin principals before touching the database."""

from __future__ import annotations

import pytest

from mynotes.core import service
from mynotes.core.errors import Forbidden
from mynotes.core.principal import LOCAL, Principal

READER = Principal("alice", ["alice-notes"], can_open_files=False, is_admin=False)


def test_trigger_scan_requires_admin() -> None:
    with pytest.raises(Forbidden):
        service.trigger_scan(principal=READER)


def test_status_requires_admin_in_every_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "collect_status", lambda: {"ok": True})
    monkeypatch.setattr(service, "_status_extras", lambda: {})
    with pytest.raises(Forbidden):
        service.status(principal=READER)
    assert service.status(principal=LOCAL)["ok"] is True


def test_agent_rows_mark_silent_agents_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import nullcontext
    from datetime import UTC, datetime, timedelta

    from mynotes.config import Settings

    now = datetime.now(UTC)
    rows = [
        ("alice-notes", now - timedelta(seconds=10), {"hostname": "laptop"}),
        ("bob-notes", now - timedelta(minutes=10), {}),
        ("carol-notes", None, {}),
    ]

    class _Result:
        def all(self) -> list:
            return rows

    class _Session:
        def execute(self, statement: object) -> _Result:
            return _Result()

    monkeypatch.setattr(service, "session_scope", lambda: nullcontext(_Session()))
    monkeypatch.setattr(service, "get_settings", lambda: Settings())
    agents = {row["root"]: row for row in service._agent_rows()}
    assert agents["alice-notes"]["state"] == "agent" and agents["alice-notes"]["stale"] is False
    assert agents["alice-notes"]["backend"] == "agent on laptop"
    assert agents["bob-notes"]["state"] == "down" and agents["bob-notes"]["stale"] is True
    assert agents["carol-notes"]["since"] == "(never)" and agents["carol-notes"]["stale"] is True
