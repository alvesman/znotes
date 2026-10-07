"""File-name stems for successor detection (PLAN.md §7.5 step 2).

:func:`parse_stem` strips everything that varies between versions of the same
document — dates, version markers (``v2``, ``rev3``, ``(1)``, ``final`` ...),
case and accents — from a file name and returns the remaining stem together
with the captured version number and date. Two files whose stems are equal are
name-signal candidates for a supersedes relation; the captured version number
is the strongest ordering signal (which file is newer).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mynotes.extract.dates import date_from_filename
from mynotes.text.normalize import normalize_key

_SEPARATOR_RUN = re.compile(r"[_\-.]+")
_WHITESPACE_RUN = re.compile(r"\s+")

_FILENAME_DATE_RE = re.compile(
    r"(?<!\d)"
    r"(?:"
    r"(?P<y>\d{4})(?P<sep>[-_.])(?P<m>\d{2})(?P=sep)(?P<d>\d{2})"
    r"|(?P<y2>\d{4})(?P<m2>\d{2})(?P<d2>\d{2})"
    r"|(?P<y3>\d{4})(?P<sep3>[-_])(?P<m3>\d{2})"
    r")"
    r"(?!\d)"
)
"""The filename date patterns of :func:`mynotes.extract.dates`, repeated here
so the matched text can also be removed from the stem."""

_VERSION_MARKER_RE = re.compile(
    r"(?:^|[\s_\-.()])(?:(?:v|ver|rev|r)(?P<n>\d+)|versao\s*(?P<vn>\d+)|version\s*(?P<vin>\d+))"
)
_PAREN_NUMBER_RE = re.compile(r"\((?P<n>\d+)\)")
_MARKER_PATTERNS = (_VERSION_MARKER_RE, _PAREN_NUMBER_RE)

_NOISE_WORDS = frozenset(
    {
        "final",
        "draft",
        "rascunho",
        "copy",
        "copia",
        "old",
        "antigo",
        "novo",
        "new",
        "latest",
        "wip",
    }
)


@dataclass(frozen=True)
class StemInfo:
    """The version-relevant parts of one file name.

    Attributes:
        stem: Normalized stem used for matching (lowercase, accent-folded,
            dates/version markers/noise words removed, separators as spaces).
        version_number: The highest version marker found (``v2``, ``rev3``,
            ``(1)`` ...), or None. Four-digit numbers after ``v``/``r``-style
            markers are years (``v2026``) and are not captured.
        date: The date found in the name (``2026-09-17``, ``20260917``,
            ``2026-09``), or None.
        ext: Lowercase extension without the dot (``md``, ``pdf``); empty for
            files without an extension.
    """

    stem: str
    version_number: int | None
    date: datetime | None
    ext: str


def parse_stem(filename: str) -> StemInfo:
    """Parse a file name into its matching stem, version number and date.

    The name without extension is accent-folded and lowercased, the date
    (if any) is captured and removed, version markers are captured (keeping
    the highest number) and removed, noise words such as ``final``/``copy``
    are dropped, and separator runs collapse to single spaces.

    Args:
        filename: A file name (with or without a leading path).

    Returns:
        The parsed :class:`StemInfo`; an empty stem when nothing remains
        (e.g. a name that was only a date).
    """
    name = Path(filename).name
    ext = Path(name).suffix.lower().lstrip(".")
    stem = normalize_key(Path(name).stem)
    stem, date = _remove_dates(stem)
    stem, version = _remove_version_markers(stem)
    stem = _WHITESPACE_RUN.sub(" ", _SEPARATOR_RUN.sub(" ", stem)).strip()
    stem = " ".join(word for word in stem.split(" ") if word not in _NOISE_WORDS)
    return StemInfo(stem=stem, version_number=version, date=date, ext=ext)


def _remove_dates(stem: str) -> tuple[str, datetime | None]:
    """Capture the leftmost valid date in ``stem`` and remove every valid one.

    Invalid matches (year out of range, impossible month/day) are left in
    place: an 8-digit run that is not a date may be part of the name.

    Returns:
        The stem without valid dates and the captured date (or None).
    """
    found = date_from_filename(stem)
    valid = [match for match in _FILENAME_DATE_RE.finditer(stem) if date_from_filename(match[0])]
    for match in reversed(valid):
        stem = stem[: match.start()] + " " + stem[match.end() :]
    return stem, found


def _remove_version_markers(stem: str) -> int | None:
    """Remove version markers from ``stem`` and return the highest number.

    Markers are ``v2``/``_v2``, ``ver3``, ``rev3``, ``r2`` at a token
    boundary, ``versao 3``/``version 3`` and parenthesized numbers ``(1)``.
    A four-digit number after a ``v``/``ver``/``rev``/``r`` marker is a year
    (``v2026``): the marker is removed but the number is not captured.

    Returns:
        The stem without markers and the highest captured number (or None).
    """
    highest: int | None = None
    for pattern in _MARKER_PATTERNS:
        matches = list(pattern.finditer(stem))
        for match in reversed(matches):
            if pattern is _VERSION_MARKER_RE:
                bare = match.group("n")
                digits = bare or match.group("vn") or match.group("vin") or ""
                year_like = bare is not None and len(bare) == 4
            else:
                digits = match.group("n") or ""
                year_like = False
            if digits and not year_like:
                number = int(digits)
                if highest is None or number > highest:
                    highest = number
            stem = stem[: match.start()] + " " + stem[match.end() :]
    return stem, highest
