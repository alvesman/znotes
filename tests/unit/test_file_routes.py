"""Unit tests for raw file serving and open-in-app (task 45).

Route tests monkeypatch the facade functions with canned results (TestClient,
no database). :func:`mynotes.core.service.resolve_file` is also tested
directly against a temp root with a fake document lookup, including the real
``is_within`` check with a symlink escape.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mynotes.api.app import create_app
from mynotes.api.auth import get_principal
from mynotes.core import service
from mynotes.core.errors import Forbidden, NotFound
from mynotes.core.principal import LOCAL, Principal

DOCX_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


Row = tuple[object, ...]


class _FakeResult:
    def __init__(self, row: Row | None) -> None:
        self._row = row

    def first(self) -> Row | None:
        return self._row

    def scalar_one_or_none(self) -> object:
        return self._row[0] if self._row else None


class _FakeSession:
    def __init__(self, row: Row | None) -> None:
        self._row = row

    def execute(self, statement: object) -> _FakeResult:
        del statement
        return _FakeResult(self._row)


def _fake_db(
    row: tuple[str, str, str] | Row | None,
) -> Callable[[], AbstractContextManager[_FakeSession]]:
    """A session whose one query returns ``row``.

    A 3-tuple ``(rel_path, format, root_abs)`` describes a local root and is
    widened to the columns ``resolve_file`` selects; longer rows pass as is.
    """
    if row is not None and len(row) == 3:
        rel_path, file_format, root_abs = row
        row = (rel_path, file_format, None, "local", root_abs, 1, 1)
    return lambda: nullcontext(_FakeSession(row))


def _write_root(tmp_path: Path, name: str, content: bytes = b"content") -> Path:
    root = tmp_path / "root"
    root.mkdir()
    (root / name).write_bytes(content)
    return root


def _resolved(
    tmp_path: Path, filename: str, media_type: str, *, inline: bool
) -> service.ResolvedFile:
    path = tmp_path / filename
    path.write_bytes(b"%PDF-1.4 fake" if filename.endswith(".pdf") else b"payload")
    return service.ResolvedFile(
        document_id=7, path=path, filename=filename, media_type=media_type, inline=inline
    )


def test_raw_pdf_served_inline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    resolved = _resolved(tmp_path, "relatorio.pdf", "application/pdf", inline=True)
    seen: list[Principal] = []

    def fake_resolve(document_id: int, *, principal: Principal = LOCAL) -> service.ResolvedFile:
        seen.append(principal)
        assert document_id == 7
        return resolved

    monkeypatch.setattr(service, "resolve_file", fake_resolve)
    response = TestClient(create_app()).get("/v1/files/7/raw")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"] == "inline; filename*=UTF-8''relatorio.pdf"
    assert response.content == b"%PDF-1.4 fake"
    assert seen == [LOCAL]


def test_raw_docx_is_attachment_with_utf8_filename(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved = _resolved(tmp_path, "relatório.docx", DOCX_TYPE, inline=False)
    monkeypatch.setattr(service, "resolve_file", lambda document_id, *, principal=LOCAL: resolved)
    response = TestClient(create_app()).get("/v1/files/7/raw")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == DOCX_TYPE
    assert (
        response.headers["content-disposition"]
        == "attachment; filename*=UTF-8''relat%C3%B3rio.docx"
    )


def test_raw_html_gets_csp_sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    resolved = _resolved(tmp_path, "page.html", "text/html; charset=utf-8", inline=True)
    monkeypatch.setattr(service, "resolve_file", lambda document_id, *, principal=LOCAL: resolved)
    response = TestClient(create_app()).get("/v1/files/7/raw")
    assert response.status_code == 200, response.text
    assert response.headers["content-security-policy"] == "sandbox; script-src 'none'"


def test_symlink_escape_returns_404(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("secret", encoding="utf-8")
    link = root / "link.md"
    try:
        link.symlink_to(outside / "secret.md")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not allowed on this platform")
    monkeypatch.setattr(service, "session_scope", _fake_db(("link.md", "md", str(root))))
    with pytest.raises(service.NotFound):
        service.resolve_file(1)
    response = TestClient(create_app()).get("/v1/files/1/raw")
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_open_returns_204(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []
    monkeypatch.setattr(
        service, "open_original", lambda document_id, *, principal=LOCAL: seen.append(document_id)
    )
    response = TestClient(create_app()).post("/v1/files/7/open")
    assert response.status_code == 204, response.text
    assert response.content == b""
    assert seen == [7]


def test_open_denied_for_team_principal(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Principal] = []

    def fake_open(document_id: int, *, principal: Principal = LOCAL) -> None:
        seen.append(principal)
        if not principal.can_open_files:
            raise service.Forbidden("opening files in apps is only allowed in local mode")

    monkeypatch.setattr(service, "open_original", fake_open)
    app = create_app()
    app.dependency_overrides[get_principal] = lambda: Principal(
        user="team", acl_prefixes=[], can_open_files=False
    )
    response = TestClient(app).post("/v1/files/7/open")
    assert response.status_code == 403, response.text
    assert seen and seen[0].can_open_files is False


def test_open_unknown_document_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    def not_found(document_id: int, *, principal: Principal = LOCAL) -> None:
        raise service.NotFound(f"no accessible file for document {document_id}")

    monkeypatch.setattr(service, "open_original", not_found)
    response = TestClient(create_app()).post("/v1/files/999/open")
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_raw_unknown_document_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    def not_found(document_id: int, *, principal: Principal = LOCAL) -> service.ResolvedFile:
        raise service.NotFound(f"no accessible file for document {document_id}")

    monkeypatch.setattr(service, "resolve_file", not_found)
    response = TestClient(create_app()).get("/v1/files/999/raw")
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_resolve_file_returns_pdf_inline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = _write_root(tmp_path, "doc.pdf", b"%PDF-1.4")
    monkeypatch.setattr(service, "session_scope", _fake_db(("doc.pdf", "pdf", str(root))))
    resolved = service.resolve_file(42)
    assert resolved.document_id == 42
    assert resolved.path == (root / "doc.pdf").resolve()
    assert resolved.filename == "doc.pdf"
    assert resolved.media_type == "application/pdf"
    assert resolved.inline is True


@pytest.mark.parametrize(
    ("file_format", "media_type", "inline"),
    [
        ("pdf", "application/pdf", True),
        ("html", "text/html; charset=utf-8", True),
        ("md", "text/plain; charset=utf-8", True),
        ("txt", "text/plain; charset=utf-8", True),
        ("docx", DOCX_TYPE, False),
        (
            "pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            False,
        ),
        ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", False),
        ("epub", "application/epub+zip", False),
    ],
)
def test_media_type_per_format(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    file_format: str,
    media_type: str,
    inline: bool,
) -> None:
    root = _write_root(tmp_path, f"doc.{file_format}")
    monkeypatch.setattr(
        service, "session_scope", _fake_db((f"doc.{file_format}", file_format, str(root)))
    )
    resolved = service.resolve_file(1)
    assert resolved.media_type == media_type
    assert resolved.inline is inline


def test_resolve_file_without_active_file_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "session_scope", _fake_db(None))
    with pytest.raises(service.NotFound):
        service.resolve_file(1)


def test_resolve_file_missing_on_disk_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(service, "session_scope", _fake_db(("gone.md", "md", str(root))))
    with pytest.raises(service.NotFound):
        service.resolve_file(1)


def test_resolve_file_enforces_acl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = _write_root(tmp_path, "secret.md")
    monkeypatch.setattr(service, "session_scope", _fake_db(("secret.md", "md", str(root))))
    principal = Principal(user="team", acl_prefixes=["allowed"], can_open_files=False)
    with pytest.raises(service.Forbidden):
        service.resolve_file(1, principal=principal)


def test_resolve_file_serves_remote_roots_from_the_blob_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A remote root's bytes come from storage.dir, named by hash and format."""
    from mynotes.config import Settings
    from mynotes.storage.blobs import BlobStore

    store = BlobStore(tmp_path / "blobs")
    digest = "ab" * 32
    blob = store.path_for(1, 7, digest, "pdf")
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"%PDF-1.4 fake")
    settings = Settings(storage={"dir": str(tmp_path / "blobs")})
    monkeypatch.setattr("mynotes.storage.blobs.get_settings", lambda: settings)
    monkeypatch.setattr(
        service,
        "session_scope",
        _fake_db(("alice-notes/reports/q1.pdf", "pdf", digest, "remote", None, 1, 7)),
    )

    resolved = service.resolve_file(1)

    assert resolved.path == blob.resolve()
    assert resolved.filename == "q1.pdf"
    assert resolved.media_type == "application/pdf"


def test_resolve_file_remote_root_without_storage_dir_is_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mynotes.config import Settings

    monkeypatch.setattr("mynotes.storage.blobs.get_settings", lambda: Settings())
    monkeypatch.setattr(
        service, "session_scope", _fake_db(("r/a.md", "md", "cd" * 32, "remote", None, 1, 2))
    )
    with pytest.raises(NotFound):
        service.resolve_file(1)


def test_open_original_refuses_remote_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    """The original of an uploaded note is on the owner's machine, not here."""
    monkeypatch.setattr(service, "session_scope", _fake_db(("remote",)))
    with pytest.raises(Forbidden):
        service.open_original(1, principal=LOCAL)
