"""Document date resolution following the PLAN.md §7.2 precedence order."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta

_FILENAME_DATE_RE = re.compile(
    r"(?<!\d)"
    r"(?:"
    r"(?P<y>\d{4})(?P<sep>[-_.])(?P<m>\d{2})(?P=sep)(?P<d>\d{2})"
    r"|(?P<y2>\d{4})(?P<m2>\d{2})(?P<d2>\d{2})"
    r"|(?P<y3>\d{4})(?P<sep3>[-_])(?P<m3>\d{2})"
    r")"
    r"(?!\d)"
)
"""Filename date patterns, longest first: YYYY-MM-DD (any of -_. as the
separator, the same on both sides), YYYYMMDD and YYYY-MM / YYYY_MM."""

_FRONTMATTER_DATE_KEYS = ("updated", "date", "created")
_MIN_YEAR = 1990
_MAX_YEAR = 2100


def date_from_filename(name: str) -> datetime | None:
    """Find a date in a file name.

    Args:
        name: A file name such as ``"plan-2026-09-17.pdf"``.

    Returns:
        The first valid date in the name (leftmost occurrence; when several
        patterns match at the same position the most specific wins), as a
        timezone-aware UTC datetime; ``YYYY-MM`` resolves to the first day of
        the month. None when there is no match, the year is outside 1990-2100,
        or the month/day are invalid (8-digit runs that are not valid dates
        are not dates).
    """
    for match in _FILENAME_DATE_RE.finditer(name):
        candidate = _validated(match)
        if candidate is not None:
            return candidate
    return None


def resolve_doc_date(
    *,
    frontmatter: dict,
    filename: str,
    metadata_date: datetime | None,
    mtime_ns: int | None,
) -> datetime | None:
    """Resolve the document date.

    Args:
        frontmatter: Parsed frontmatter of the document.
        filename: File name, searched for an embedded date.
        metadata_date: Date from document metadata (e.g. PDF metadata).
        mtime_ns: File modification time in nanoseconds.

    Returns:
        The first available of: frontmatter ``updated``, ``date``, ``created``
        (a ``date``, ``datetime`` or ISO string); a date in the file name;
        ``metadata_date``; the mtime. Unparseable frontmatter values are
        skipped and the next source is used. Every result is timezone-aware
        UTC; naive values are assumed to be UTC.
    """
    for key in _FRONTMATTER_DATE_KEYS:
        parsed = _parse_frontmatter_value(frontmatter.get(key))
        if parsed is not None:
            return parsed
    from_filename = date_from_filename(filename)
    if from_filename is not None:
        return from_filename
    if metadata_date is not None:
        return _ensure_utc(metadata_date)
    if mtime_ns is not None:
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        return epoch + timedelta(microseconds=mtime_ns // 1000)
    return None


def _validated(match: re.Match[str]) -> datetime | None:
    """Build the UTC datetime for a filename match, or None when invalid."""
    if match.group("y") is not None:
        year, month, day = int(match["y"]), int(match["m"]), int(match["d"])
    elif match.group("y2") is not None:
        year, month, day = int(match["y2"]), int(match["m2"]), int(match["d2"])
    else:
        year, month, day = int(match["y3"]), int(match["m3"]), 1
    if not _MIN_YEAR <= year <= _MAX_YEAR:
        return None
    try:
        return datetime(year, month, day, tzinfo=UTC)
    except ValueError:
        return None


def _parse_frontmatter_value(value: object) -> datetime | None:
    """Parse one frontmatter value into an aware UTC datetime, or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if isinstance(value, str):
        try:
            return _ensure_utc(datetime.fromisoformat(value.strip()))
        except ValueError:
            return None
    return None


def _ensure_utc(value: datetime) -> datetime:
    """Return an aware datetime in UTC; naive values are assumed UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
