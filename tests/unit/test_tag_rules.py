"""Unit tests for tag normalization, folder rules and manual tag sources (task 37)."""

from __future__ import annotations

from mynotes.config import FolderRule, TagsConfig
from mynotes.tags.folder_rules import folder_tags
from mynotes.tags.manual import frontmatter_tags, hashtag_tags, manual_tags
from mynotes.tags.normalize import normalize_tag, tag_kind


def rules(*pairs: tuple[str, list[str]]) -> TagsConfig:
    """A TagsConfig from ``(pattern, tags)`` pairs."""
    return TagsConfig(
        folder_rules=[FolderRule(pattern=pattern, tags=tags) for pattern, tags in pairs]
    )


def test_normalize_tag_folds_case_accents_and_spaces() -> None:
    assert normalize_tag("Área Técnica") == "area-tecnica"
    assert normalize_tag("Project: Alpha") == "project:alpha"
    assert normalize_tag("#ml") == "ml"


def test_normalize_tag_keeps_the_prefix_separator() -> None:
    assert normalize_tag("Projeto::: Alfa  ") == "projeto:alfa"
    assert normalize_tag("Status : Arquivado") == "status:arquivado"


def test_normalize_tag_drops_other_characters_and_collapses_dashes() -> None:
    assert normalize_tag("a --- b") == "a-b"
    assert normalize_tag("Coração & Ação!") == "coracao-acao"
    assert normalize_tag("my_tag") == "my_tag"
    assert normalize_tag("ml/ai") == "ml/ai"


def test_normalize_tag_empty_results() -> None:
    assert normalize_tag("") == ""
    assert normalize_tag("  ") == ""
    assert normalize_tag("#!?") == ""
    assert normalize_tag("---") == ""


def test_tag_kind_from_prefix() -> None:
    assert tag_kind("project:alpha") == "project"
    assert tag_kind("status:archived") == "status"
    assert tag_kind("folder:projetos") == "folder"
    assert tag_kind("topic:ml") == "topic"
    assert tag_kind("ml") == "other"
    assert tag_kind("") == "other"


def test_folder_rule_example_from_the_plan() -> None:
    cfg = rules(
        ("projects/{project}/**", ["project:{project}"]),
        ("**/archive/**", ["status:archived"]),
    )
    assert folder_tags("Projects/Alpha Beta/archive/idea.md", cfg) == [
        "project:alpha-beta",
        "status:archived",
    ]


def test_folder_rule_matches_case_and_accent_insensitively() -> None:
    cfg = rules(("projetos/{project}/**", ["project:{project}"]))
    assert folder_tags("PROJETOS/Ação/x.md", cfg) == ["project:acao"]


def test_folder_rule_star_matches_exactly_one_segment() -> None:
    cfg = rules(("*/notas/**", ["status:notas"]))
    assert folder_tags("A/notas/x.md", cfg) == ["status:notas", "folder:a"]
    assert folder_tags("A/B/notas/x.md", cfg) == ["folder:a", "folder:b", "folder:notas"]


def test_folder_rule_without_glob_matches_only_the_exact_folder() -> None:
    cfg = rules(("clients/2026", ["status:q1"]))
    assert folder_tags("clients/2026/relatorio.md", cfg) == ["status:q1"]
    assert folder_tags("clients/2027/relatorio.md", cfg) == ["folder:clients", "folder:2027"]


def test_folder_segments_are_tagged_when_no_rule_matches() -> None:
    assert folder_tags("Docs/Reuniões/nota.md", rules()) == [
        "folder:docs",
        "folder:reunioes",
    ]


def test_folder_segments_skip_ignore_segments_and_archive_folders() -> None:
    cfg = TagsConfig()
    assert folder_tags("misc/keep/tmp/archive/nota.md", cfg, ["archive", "old"]) == ["folder:keep"]


def test_folder_segments_respect_configured_ignore_segments() -> None:
    cfg = TagsConfig(ignore_segments=["rascunhos"])
    assert folder_tags("rascunhos/nota.md", cfg) == []


def test_folder_segments_can_be_disabled() -> None:
    cfg = TagsConfig(folder_segments_as_tags=False)
    assert folder_tags("docs/nota.md", cfg) == []


def test_folder_rule_templates_with_missing_captures_are_skipped() -> None:
    cfg = rules(("projects/{project}/**", ["project:{project}", "topic:{missing}"]))
    assert folder_tags("projects/alpha/deep/x.md", cfg) == ["project:alpha", "folder:deep"]


def test_folder_rules_capture_values_are_normalized() -> None:
    cfg = rules(("projetos/{project}/**", ["project:{project}"]))
    assert folder_tags("Projetos/Área Técnica 2/x.md", cfg) == ["project:area-tecnica-2"]


def test_folder_tags_of_a_root_file_are_empty() -> None:
    assert folder_tags("nota.md", rules()) == []


def test_frontmatter_and_hashtag_helpers() -> None:
    assert frontmatter_tags({"tags": ["Ideia Alfa", "ml"], "keywords": "a, b"}) == [
        "ideia-alfa",
        "ml",
        "a",
        "b",
    ]
    assert hashtag_tags(["Projeto-X", "#dup", "dup"]) == ["projeto-x", "dup"]


def test_manual_tags_include_folder_tags_after_rel_path() -> None:
    cfg = rules(("projects/{project}/**", ["project:{project}"]))
    tags = manual_tags({"tags": ["ML"]}, ["ideia"], rel_path="projects/alpha/note.md", cfg=cfg)
    assert tags == ["ml", "ideia", "project:alpha"]


def test_manual_tags_without_rel_path_ignore_folders() -> None:
    assert manual_tags({"tags": ["ml"]}, [], rel_path=None) == ["ml"]
