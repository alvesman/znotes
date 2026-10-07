"""Query analysis: inline filters and intent, rules only (PLAN.md §9 step 1).

:func:`analyze_query` strips the inline filter tokens (``#tag``, ``in:``,
``format:``, ``since:``/``until:``, ``history:`` and ``as-of:``) from the
query, merges them with the caller's base filters and detects the *history*
intent from whole-word keywords on the normalized text. The remaining text is
what the hybrid search embeds and matches; the detected language only feeds
the answer language (task 30).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Literal

from mynotes.config import get_settings
from mynotes.retrieval.types import SearchFilters
from mynotes.tags.normalize import normalize_tag
from mynotes.text.language import detect_language
from mynotes.text.normalize import normalize_key

logger = logging.getLogger(__name__)

HASHTAG_RE = re.compile(r"(?<![\w&/#])#([^\W\d_][\w\-/]*)")
"""Inline hashtags; the same pattern as the Markdown extractor's ``_HASHTAG_RE``."""

_IN_RE = re.compile(r'(?<!\w)in:(?:"(?P<quoted>[^"]*)"|(?P<bare>\S+))', re.IGNORECASE)
_FORMAT_RE = re.compile(r"(?<!\w)format:(?P<value>\S+)", re.IGNORECASE)
_DATE_VALUE = r"\s*(?P<value>\d{4}-\d{2}(?:-\d{2})?)(?!\d)"
"""A ``YYYY-MM`` or ``YYYY-MM-DD`` value, with optional spaces after the token's colon
(``as-of: 2026-06-05`` is how people write it)."""
_SINCE_RE = re.compile(r"(?<!\w)since:" + _DATE_VALUE, re.IGNORECASE)
_UNTIL_RE = re.compile(r"(?<!\w)until:" + _DATE_VALUE, re.IGNORECASE)
_AS_OF_RE = re.compile(r"(?<!\w)as-of:" + _DATE_VALUE, re.IGNORECASE)
_HISTORY_TOKEN_RE = re.compile(r"(?<!\w)history:(?=\s|$)", re.IGNORECASE)

_HISTORY_KEYWORDS: tuple[str, ...] = (
    "antes",
    "anterior",
    "versao anterior",
    "versoes",
    "historico",
    "evolucao",
    "mudou",
    "alteracoes",
    "previous",
    "earlier",
    "history",
    "changed",
    "changes",
    "evolution",
    "used to",
)
"""Matched as whole words/phrases on ``normalize_key(text)`` (already folded)."""

_HISTORY_KEYWORD_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(keyword) for keyword in _HISTORY_KEYWORDS) + r")(?!\w)"
)


@dataclass
class QueryAnalysis:
    """The rule-based analysis of one user query.

    Attributes:
        raw: The query exactly as typed.
        text: The query with the filter tokens removed, whitespace collapsed.
        filters: The merged filters (inline tokens over the base filters).
        intent: ``"history"`` when the query asks about past versions.
        as_of: The point in time of an ``as-of:`` token, or None.
        language: ``"pt"``/``"en"`` when the cleaned text is confidently
            detected, else None.
        quoted: The phrases the user put in quotes (straight or curly), in
            order, without the quotes; a quoted phrase that equals a note's
            title locks that note to the top (:mod:`mynotes.retrieval.title_lock`).
        named_dates: Full dates written in the question text ("2026-01-12",
            "12/01/2026", "12 de janeiro de 2026", "January 12, 2026"), for
            current-intent questions only: in an as-of or history question a
            date is a cutoff, not the identity of a note.
        asks_change: The question asks when something changed to a value
            ("em que reunião X passou a Y?", "when did X change to Y?").
    """

    raw: str
    text: str
    filters: SearchFilters
    intent: Literal["current", "history"]
    as_of: datetime | None
    language: str | None
    quoted: tuple[str, ...] = ()
    named_dates: tuple[date, ...] = ()
    asks_change: bool = False


_QUOTED_RE = re.compile(r'"([^"\n]{2,120})"|“([^”\n]{2,120})”|«([^»\n]{2,120})»')


def quoted_phrases(text: str) -> tuple[str, ...]:
    """The phrases in straight, curly or angle quotes, stripped, in order of appearance.

    Args:
        text: The query text (filter tokens already removed; ``in:"..."``
            folders are therefore never returned).
    """
    phrases: list[str] = []
    for match in _QUOTED_RE.finditer(text):
        phrase = next(group for group in match.groups() if group is not None).strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    return tuple(phrases)


_MONTHS = {
    name: number
    for number, names in enumerate(
        (
            ("janeiro", "january", "jan"),
            ("fevereiro", "february", "fev", "feb"),
            ("marco", "march", "mar"),
            ("abril", "april", "abr", "apr"),
            ("maio", "may", "mai"),
            ("junho", "june", "jun"),
            ("julho", "july", "jul"),
            ("agosto", "august", "ago", "aug"),
            ("setembro", "september", "set", "sep"),
            ("outubro", "october", "out", "oct"),
            ("novembro", "november", "nov"),
            ("dezembro", "december", "dez", "dec"),
        ),
        start=1,
    )
    for name in names
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ISO_DATE_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})-(\d{2})-(\d{2})(?!\d)")
_DMY_DATE_RE = re.compile(r"(?<![\w/.-])(\d{1,2})[/.](\d{1,2})[/.]((?:19|20)\d{2})(?!\d)")
_PT_DATE_RE = re.compile(rf"\b(\d{{1,2}}) de ({_MONTH_ALT})\.? de ((?:19|20)\d{{2}})\b")
_EN_DATE_RE = re.compile(rf"\b({_MONTH_ALT})\.? (\d{{1,2}})(?:st|nd|rd|th)?,? ((?:19|20)\d{{2}})\b")
_EN_DMY_RE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)? ({_MONTH_ALT})\.?,? ((?:19|20)\d{{2}})\b")
_CHANGE_RE = re.compile(
    r"\b(passou a|passou para|passaram a|mudou para|mudou de|foi alterad[oa] para|"
    r"desde quando|a partir de quando|changed to|switched to|moved to|became|since when)\b"
)


def named_dates(text: str) -> tuple[date, ...]:
    """Full calendar dates written in ``text``, in order of appearance, without repeats.

    ISO (``2026-01-12``), day/month/year with ``/`` or ``.`` (``12/01/2026``,
    Portuguese order), ``12 de janeiro de 2026`` and ``January 12, 2026`` /
    ``12 January 2026``. Month-only or year-only mentions are not dates of a note.
    """
    folded = normalize_key(text)
    found: list[tuple[int, date]] = []

    def add(position: int, year: str, month: int, day: str) -> None:
        try:
            found.append((position, date(int(year), month, int(day))))
        except ValueError:
            return

    for m in _ISO_DATE_RE.finditer(folded):
        add(m.start(), m.group(1), int(m.group(2)), m.group(3))
    for m in _DMY_DATE_RE.finditer(folded):
        add(m.start(), m.group(3), int(m.group(2)), m.group(1))
    for m in _PT_DATE_RE.finditer(folded):
        add(m.start(), m.group(3), _MONTHS[m.group(2)], m.group(1))
    for m in _EN_DATE_RE.finditer(folded):
        add(m.start(), m.group(3), _MONTHS[m.group(1)], m.group(2))
    for m in _EN_DMY_RE.finditer(folded):
        add(m.start(), m.group(3), _MONTHS[m.group(2)], m.group(1))
    ordered: list[date] = []
    for _position, value in sorted(found, key=lambda item: item[0]):
        if value not in ordered:
            ordered.append(value)
    return tuple(ordered)


def is_change_question(text: str) -> bool:
    """Whether the question asks when something changed to a value."""
    return bool(_CHANGE_RE.search(normalize_key(text)))


def _parse_date_value(value: str) -> datetime | None:
    """Parse ``YYYY-MM`` (first day) or ``YYYY-MM-DD`` as UTC, else None."""
    parts = value.split("-")
    try:
        if len(parts) == 2:
            return datetime(int(parts[0]), int(parts[1]), 1, tzinfo=UTC)
        if len(parts) == 3:
            return datetime(int(parts[0]), int(parts[1]), int(parts[2]), tzinfo=UTC)
    except ValueError:
        return None
    return None


def _end_of_period(value: str) -> datetime | None:
    """The last moment of the given day or month as UTC, or None."""
    start = _parse_date_value(value)
    if start is None:
        return None
    if len(value.split("-")) == 2:
        next_month = datetime(start.year + start.month // 12, start.month % 12 + 1, 1, tzinfo=UTC)
        return next_month - timedelta(microseconds=1)
    return start + timedelta(days=1) - timedelta(microseconds=1)


def _without_spans(raw: str, spans: list[tuple[int, int]]) -> str:
    """Drop the given (possibly overlapping) spans and collapse whitespace."""
    dropped = [False] * len(raw)
    for start, end in spans:
        for index in range(max(0, start), min(end, len(raw))):
            dropped[index] = True
    kept = "".join(char for char, skip in zip(raw, dropped, strict=True) if not skip)
    return " ".join(kept.split())


def _dedupe(values: list[str]) -> list[str]:
    """First occurrence wins, order preserved."""
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def analyze_query(raw: str, base_filters: SearchFilters | None = None) -> QueryAnalysis:
    """Analyze a query: strip inline filters, merge filters, detect intent.

    Args:
        raw: The raw user query, possibly with inline filter tokens.
        base_filters: Filters from CLI/API options; lists are concatenated
            with the inline ones and scalars from the query win.

    Returns:
        The :class:`QueryAnalysis` for the query. History intent (keywords or
        ``history:``/``as-of:``) sets ``include_superseded`` and turns
        ``include_old_versions`` into ``versions.index_old_versions``.
    """
    base = base_filters if base_filters is not None else SearchFilters()
    tags = list(base.tags)
    folders = list(base.folders)
    formats = list(base.formats)
    since = base.since
    until = base.until
    intent: Literal["current", "history"] = "current"
    as_of: datetime | None = None
    spans: list[tuple[int, int]] = []

    for match in HASHTAG_RE.finditer(raw):
        tag = normalize_tag(match.group(1))
        if tag:
            tags.append(tag)
        spans.append(match.span())
    for match in _IN_RE.finditer(raw):
        quoted = match.group("quoted")
        folder = (quoted if quoted is not None else match.group("bare") or "").strip()
        if folder:
            folders.append(folder)
        spans.append(match.span())
    for match in _FORMAT_RE.finditer(raw):
        value = match.group("value").strip()
        if value:
            formats.append(value)
        spans.append(match.span())
    for match in _SINCE_RE.finditer(raw):
        parsed = _parse_date_value(match.group("value"))
        if parsed is not None:
            since = parsed
            spans.append(match.span())
        else:
            logger.debug("ignoring unparseable since token %r", match.group(0))
    for match in _UNTIL_RE.finditer(raw):
        parsed = _end_of_period(match.group("value"))
        if parsed is not None:
            until = parsed
            spans.append(match.span())
        else:
            logger.debug("ignoring unparseable until token %r", match.group(0))
    for match in _AS_OF_RE.finditer(raw):
        parsed = _parse_date_value(match.group("value"))
        if parsed is not None:
            as_of = parsed
            intent = "history"
            # The state "as of" a date: nothing written after that day can describe it, so
            # the date also bounds doc_date (the version valid on that date then ranks first).
            bound = _end_of_period(match.group("value"))
            if bound is not None:
                until = bound if until is None else min(until, bound)
            spans.append(match.span())
        else:
            logger.debug("ignoring unparseable as-of token %r", match.group(0))
    for match in _HISTORY_TOKEN_RE.finditer(raw):
        intent = "history"
        spans.append(match.span())

    text = _without_spans(raw, spans)
    if intent == "current" and _HISTORY_KEYWORD_RE.search(normalize_key(text)):
        intent = "history"

    filters = SearchFilters(
        tags=_dedupe(tags),
        folders=_dedupe(folders),
        formats=_dedupe(formats),
        since=since,
        until=until,
        include_superseded=True if intent == "history" else base.include_superseded,
        include_old_versions=(
            get_settings().versions.index_old_versions
            if intent == "history"
            else base.include_old_versions
        ),
        acl_prefixes=base.acl_prefixes,
        workspace_id=base.workspace_id,
    )
    return QueryAnalysis(
        raw=raw,
        text=text,
        filters=filters,
        intent=intent,
        as_of=as_of,
        language=detect_language(text),
        quoted=quoted_phrases(text),
        named_dates=named_dates(text) if intent == "current" and as_of is None else (),
        asks_change=is_change_question(text),
    )
