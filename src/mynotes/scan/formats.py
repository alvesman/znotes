"""Format detection for scanned files.

The scanner only yields files whose extension maps to a format mynotes can
extract (``settings.formats``). Legacy binary Office/OpenDocument files are
detected as ``"unsupported"`` so the scanner can record them (they show up in
``mynotes status``) instead of silently dropping them (PLAN.md §7.2). Any
other extension is ignored silently.
"""

from __future__ import annotations

from pathlib import Path

from mynotes import config

SUPPORTED: dict[str, str] = {
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".xlsx": "xlsx",
    ".html": "html",
    ".htm": "html",
    ".epub": "epub",
}
"""Extension (lower-case) to format name for every extractable format."""

LEGACY_UNSUPPORTED: frozenset[str] = frozenset(
    {".doc", ".ppt", ".xls", ".odt", ".odp", ".ods", ".rtf"}
)
"""Extensions of known document formats mynotes deliberately does not extract."""


def format_for(path: str | Path) -> str | None:
    """Return the scanner format name for a path's extension.

    Args:
        path: A file path (only the extension is inspected).

    Returns:
        The format name (e.g. ``"md"``, ``"pdf"``) for supported extensions
        that are enabled in ``settings.formats``, ``"unsupported"`` for the
        legacy formats in :data:`LEGACY_UNSUPPORTED`, and None for everything
        else (the caller ignores the file silently).
    """
    suffix = Path(path).suffix.lower()
    if suffix in LEGACY_UNSUPPORTED:
        return "unsupported"
    fmt = SUPPORTED.get(suffix)
    if fmt is None or fmt not in config.get_settings().formats:
        return None
    return fmt
