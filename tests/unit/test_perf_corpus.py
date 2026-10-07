"""Smoke test for the synthetic corpus generator (task 51).

The generator lives in ``scripts/perf/generate_corpus.py``; loading it via
importlib keeps the deliverable a script while still covering the CI-time
``--files 50`` run: 50 files, the 80/10/5/3/2 format mix, a folder tree and
deterministic output for a fixed seed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


def _load_generator() -> Any:
    """Import ``scripts/perf/generate_corpus.py`` as a module."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "perf" / "generate_corpus.py"
    spec = importlib.util.spec_from_file_location("perf_generate_corpus", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def generator() -> Any:
    return _load_generator()


def test_generates_fifty_files_in_the_format_mix(generator: Any, tmp_path: Path) -> None:
    """50 files land on disk with the planned per-format counts."""
    summary = generator.generate(tmp_path / "corpus", files=50, seed=42)
    counts = summary["counts"]
    assert summary["files"] == 50
    assert sum(counts.values()) == 50
    assert counts == generator.plan_counts(50)
    assert counts["md"] == 41 and counts["txt"] == 5 and counts["pdf"] == 2
    assert counts["docx"] == 1 and counts["pptx"] == 1

    on_disk: dict[str, int] = {}
    for path in (tmp_path / "corpus").rglob("*"):
        if path.is_file():
            on_disk[path.suffix.lstrip(".")] = on_disk.get(path.suffix.lstrip("."), 0) + 1
    assert on_disk == counts
    directories = [folder for folder in (tmp_path / "corpus").rglob("*") if folder.is_dir()]
    assert summary["folders"] == len(directories) - 3  # projects/, areas/, archive/ roots


def test_same_seed_is_deterministic(generator: Any, tmp_path: Path) -> None:
    """Two runs with the same seed write the same text-file bytes."""
    first = tmp_path / "a"
    second = tmp_path / "b"
    summary_a = generator.generate(first, files=50, seed=7)
    summary_b = generator.generate(second, files=50, seed=7)
    assert summary_a == summary_b
    for extension in ("md", "txt"):
        names_a = sorted(path.relative_to(first) for path in first.rglob(f"*.{extension}"))
        names_b = sorted(path.relative_to(second) for path in second.rglob(f"*.{extension}"))
        assert names_a == names_b
        for name in names_a:
            assert (first / name).read_bytes() == (second / name).read_bytes()


def test_plan_counts_sum_to_total(generator: Any) -> None:
    """The mix plan sums exactly to the requested file count."""
    for total in (1, 7, 50, 1999, 20000):
        counts = generator.plan_counts(total)
        assert sum(counts.values()) == total
        assert counts["md"] >= counts["txt"] >= counts["pdf"]
