"""The storage checks of init and status."""

from __future__ import annotations

from pathlib import Path

from mynotes.config import Settings
from mynotes.core import setup


def test_storage_check_creates_and_probes_the_directory(tmp_path: Path) -> None:
    settings = Settings(storage={"dir": str(tmp_path / "blobs")})
    (result,) = setup._check_storage(settings)
    assert result.ok and "writable" in result.detail
    assert (tmp_path / "blobs" / "tmp").is_dir()
    assert setup._status_storage(settings) == {
        "configured": True,
        "dir": str(tmp_path / "blobs"),
        "exists": True,
        "bytes": 0,
    }


def test_storage_check_without_dir_and_without_database(monkeypatch) -> None:
    def no_db():
        raise RuntimeError("no database")

    monkeypatch.setattr(setup, "session_scope", no_db)
    (result,) = setup._check_storage(Settings())
    assert result.ok and "not configured" in result.detail
    assert setup._status_storage(Settings()) == {"configured": False}
