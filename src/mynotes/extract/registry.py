"""Extractor registry mapping format names to extractor instances.

Extractors register themselves at import time of their module;
:func:`get_extractor` imports the known extractor modules lazily so that
importing the registry alone stays cheap and free of import cycles
(PLAN.md §7.2).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Protocol

from mynotes.extract.ir import IRDocument

_KNOWN_MODULES: tuple[str, ...] = (
    "mynotes.extract.markdown",
    "mynotes.extract.pdf",
    "mynotes.extract.office",
)
"""Extractor modules imported (once) on the first :func:`get_extractor` call."""

_EXTRACTORS: dict[str, Extractor] = {}
_LOADED_MODULES: set[str] = set()


class Extractor(Protocol):
    """Structural interface every format extractor implements."""

    name: str
    version: str
    formats: tuple[str, ...]

    def extract(self, path: Path) -> IRDocument:
        """Extract one file into the common IR."""
        ...


def register(extractor: Extractor) -> None:
    """Add an extractor for each of its formats.

    Args:
        extractor: The extractor instance; later registrations for the same
            format replace earlier ones.
    """
    for fmt in extractor.formats:
        _EXTRACTORS[fmt] = extractor


def get_extractor(fmt: str) -> Extractor:
    """Return the registered extractor for a format name.

    Args:
        fmt: A scanner format name (e.g. ``"md"``, ``"pdf"``).

    Returns:
        The extractor registered for that format.

    Raises:
        KeyError: When no known extractor handles the format. The message is
            ``"no extractor for format '<fmt>'"``.
    """
    for module_name in _KNOWN_MODULES:
        if module_name not in _LOADED_MODULES:
            importlib.import_module(module_name)
            _LOADED_MODULES.add(module_name)
    try:
        return _EXTRACTORS[fmt]
    except KeyError:
        raise KeyError(f"no extractor for format '{fmt}'") from None
