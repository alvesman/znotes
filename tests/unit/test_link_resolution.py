"""Unit tests for the deterministic link resolution (task 39).

An in-memory :class:`DocIndexEntry` list stands in for the workspace, so no
database or filesystem is involved. Covers the task's five cases plus the
matching-order and self-link edges.
"""

from __future__ import annotations

from mynotes.extract.ir import Link
from mynotes.graph.links import DocIndexEntry, file_stem, resolve_link, wikilink_stem

SOURCE = "a/b/src.md"
A = DocIndexEntry(
    document_id=1,
    rel_path=SOURCE,
    folder="a/b",
    stem_norm="src",
    title_norm="fonte",
)
OTHER = DocIndexEntry(
    document_id=2,
    rel_path="a/other/note.md",
    folder="a/other",
    stem_norm="note",
    title_norm="nota",
)
PLANO_SAME_FOLDER = DocIndexEntry(
    document_id=3,
    rel_path="a/b/Plano Beta.md",
    folder="a/b",
    stem_norm="plano beta",
    title_norm=None,
)
PLANO_FAR = DocIndexEntry(
    document_id=4,
    rel_path="docs/Plano Beta.md",
    folder="docs",
    stem_norm="plano beta",
    title_norm=None,
)
BY_TITLE = DocIndexEntry(
    document_id=5,
    rel_path="outro.md",
    folder="",
    stem_norm="outro",
    title_norm="o plano beta",
)

INDEX = [A, OTHER, PLANO_SAME_FOLDER, PLANO_FAR, BY_TITLE]


def test_markdown_link_with_anchor_resolves_relative() -> None:
    """`[x](../other/note.md#sec)` from `a/b/src.md` lands on `a/other/note.md`."""
    link = Link(target="../other/note.md#sec", kind="markdown", text="x")
    assert resolve_link(link, SOURCE, INDEX) == 2


def test_markdown_link_url_encoded_and_query() -> None:
    """`Plano%20Beta.md` decodes to a space and matches; query parts drop."""
    link = Link(target="Plano%20Beta.md?download=1", kind="markdown")
    assert resolve_link(link, "a/b/readme.md", INDEX) == 3


def test_markdown_link_case_and_accent_insensitive() -> None:
    """A folded match backs the exact one up."""
    link = Link(target="../other/Nóte.md", kind="markdown")
    assert resolve_link(link, SOURCE, INDEX) == 2


def test_markdown_link_unresolvable_and_anchor_only() -> None:
    """Missing targets and `#section`-only links resolve to None."""
    assert resolve_link(Link(target="missing.md", kind="markdown"), SOURCE, INDEX) is None
    assert resolve_link(Link(target="#sec", kind="markdown"), SOURCE, INDEX) is None


def test_wikilink_by_stem_prefers_same_folder() -> None:
    """`[[Plano Beta]]` has two stem matches; the source's folder wins."""
    link = Link(target="Plano Beta", kind="wikilink")
    assert resolve_link(link, SOURCE, INDEX) == 3


def test_wikilink_by_stem_prefers_shortest_distance_then_recency() -> None:
    """Without a same-folder match, the closer folder wins (docs/ over a/b/)."""
    link = Link(target="Plano Beta", kind="wikilink")
    assert resolve_link(link, "readme.md", INDEX) == 4


def test_wikilink_by_title_when_no_stem_matches() -> None:
    """A target that is no rel_path and no stem falls back to the title."""
    link = Link(target="O Plano Beta", kind="wikilink")
    assert resolve_link(link, SOURCE, INDEX) == 5


def test_wikilink_accent_and_case_insensitive_stem() -> None:
    """`[[plano beta]]` matches `Plano Beta.md` by folded stem."""
    link = Link(target="plano beta", kind="wikilink")
    assert resolve_link(link, SOURCE, INDEX) == 3


def test_wikilink_exact_rel_path_with_and_without_extension() -> None:
    """Step 1 matches the exact rel_path, also when `T + ".md"` is needed."""
    assert resolve_link(Link(target="docs/Plano Beta.md", kind="wikilink"), SOURCE, INDEX) == 4
    assert resolve_link(Link(target="a/other/note", kind="wikilink"), SOURCE, INDEX) == 2


def test_wikilink_unresolvable() -> None:
    """No rel_path, stem or title match → None."""
    assert resolve_link(Link(target="Nada", kind="wikilink"), SOURCE, INDEX) is None


def test_external_urls_and_self_links_are_none() -> None:
    """URLs never resolve and a link to the source file itself is dropped."""
    assert resolve_link(Link(target="https://ex.com/a.md", kind="url"), SOURCE, INDEX) is None
    assert resolve_link(Link(target="a/b/src.md", kind="wikilink"), SOURCE, INDEX) is None
    assert resolve_link(Link(target="src.md#top", kind="markdown"), SOURCE, INDEX) is None


def test_stem_helpers() -> None:
    """File stems drop the last extension; wikilink stems drop only `.md`."""
    assert file_stem("notes/idea.md") == "idea"
    assert file_stem("notes/.hidden") == ".hidden"
    assert wikilink_stem("a/b/Plano Beta.md") == "Plano Beta"
    assert wikilink_stem("Plano") == "Plano"
