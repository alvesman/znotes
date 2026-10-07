#!/usr/bin/env python3
"""M7 perf: synthetic corpus generator (task 51).

Builds a deterministic folder tree that mimics the planned upper scale
(20k files, roughly 300k chunks; PLAN.md §1): ``projects/<p>/…`` (4 levels
deep), ``areas/…`` and ``archive/…`` with about 200 folders. The mix is
80% Markdown (frontmatter with tags/dates, 3-15 sections, PT, EN or mixed,
built from a word list and sentence templates - no LLM), 10% TXT, 5% PDF
(PyMuPDF, 2-10 pages) and 5% DOCX/PPTX (python-docx/python-pptx).
5% of the Markdown files are ``_v2`` successors with 70-90% overlapping
text, 2% of all files are byte-identical duplicates and 1% of the Markdown
files carry a ``supersedes:`` frontmatter link to an earlier file.

Output is deterministic for a given ``--seed`` (the Office containers get
fresh container timestamps per run; their text content is identical).

    uv run python scripts/perf/generate_corpus.py --out /tmp/mynotes-perf \
        --files 20000 --seed 42

Prints a JSON summary (per-format counts and the version/duplicate counts)
on stdout.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

try:
    import docx
    import fitz
    import pptx
except ImportError as exc:  # pragma: no cover - dev extras must be installed
    print(f"missing dependency: {exc}; run 'uv sync'", file=sys.stderr)
    raise SystemExit(2) from exc

PT_WORDS = [
    "projeto",
    "orçamento",
    "pesquisa",
    "reunião",
    "calendário",
    "servidor",
    "base de dados",
    "segurança",
    "infraestrutura",
    "protótipo",
    "apresentação",
    "cliente",
    "contrato",
    "relatório",
    "análise",
    "requisitos",
    "arquitetura",
    "testes",
    "migração",
    "filtro",
    "exaustor",
    "cozinha",
    "laboratório",
    "equipamento",
    "fornecedor",
    "cronograma",
    "marco",
    "entrega",
    "equipe",
    "minuta",
    "estratégia",
    "aprendizado",
    "conferência",
    "artigo",
    "experimento",
    "resultado",
    "métrica",
    "latência",
    "índice",
    "vetor",
    "embeddings",
    "recuperação",
    "busca",
    "semântico",
    "grafo",
    "entidade",
    "relação",
    "versão",
    "histórico",
    "arquivo",
    "pasta",
    "etiqueta",
    "aquisição",
    "avaliação",
    "backup",
    "capacidade",
    "comissão",
    "custo",
    "desempenho",
    "desenho",
    "documentação",
    "exemplo",
    "formação",
    "gestão",
    "implementação",
    "integração",
    "inventário",
    "manutenção",
    "metodologia",
    "monitorização",
    "normalização",
    "objeto",
    "opinião",
    "parceria",
    "planeamento",
    "política",
    "prioridade",
    "processo",
    "produção",
    "programa",
    "qualidade",
    "questão",
    "refrigeração",
    "reparação",
    "resolução",
    "responsável",
    "revisão",
    "sensores",
    "serviço",
    "solução",
    "supervisão",
    "tecnologia",
    "utilizador",
    "validação",
    "valor",
]

EN_WORDS = [
    "project",
    "budget",
    "research",
    "meeting",
    "calendar",
    "server",
    "database",
    "security",
    "infrastructure",
    "prototype",
    "presentation",
    "client",
    "contract",
    "report",
    "analysis",
    "requirements",
    "architecture",
    "testing",
    "migration",
    "filter",
    "exhaust hood",
    "kitchen",
    "laboratory",
    "equipment",
    "supplier",
    "schedule",
    "milestone",
    "delivery",
    "team",
    "draft",
    "strategy",
    "learning",
    "conference",
    "paper",
    "experiment",
    "result",
    "metric",
    "latency",
    "index",
    "vector",
    "embeddings",
    "retrieval",
    "search",
    "semantic",
    "graph",
    "entity",
    "relation",
    "version",
    "history",
    "archive",
    "folder",
    "tag",
    "acquisition",
    "assessment",
    "backup",
    "capacity",
    "committee",
    "competition",
    "cost",
    "performance",
    "design",
    "documentation",
    "example",
    "training",
    "management",
    "implementation",
    "integration",
    "inventory",
    "maintenance",
    "methodology",
    "monitoring",
    "normalization",
    "object",
    "feedback",
    "partner",
    "planning",
    "policy",
    "priority",
    "process",
    "production",
    "program",
    "quality",
    "question",
    "refrigeration",
    "repair",
    "resolution",
    "owner",
    "review",
    "sensor",
    "service",
    "solution",
    "supervision",
    "technology",
    "user",
    "validation",
    "value",
]

PROJECTS = [
    "alfa",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
]

AREAS = [
    "pesquisa",
    "financas",
    "reunioes",
    "infraestrutura",
    "laboratorio",
    "formacao",
    "administracao",
    "arquivo",
]

PT_SECTIONS = [
    "Contexto e motivação",
    "Decisões tomadas",
    "Próximos passos",
    "Riscos e mitigação",
    "Resultados do experimento",
    "Notas técnicas",
    "Perguntas em aberto",
    "Orçamento e recursos",
    "Cronograma e marcos",
    "Reunião com a equipe",
    "Avaliação de fornecedores",
    "Lições aprendidas",
    "Requisitos e restrições",
    "Testes e validação",
    "Documentação e arquivos",
]

EN_SECTIONS = [
    "Background and motivation",
    "Decisions",
    "Next steps",
    "Risks and mitigations",
    "Experiment results",
    "Technical notes",
    "Open questions",
    "Budget and resources",
    "Schedule and milestones",
    "Team meeting",
    "Supplier assessment",
    "Lessons learned",
    "Requirements and constraints",
    "Testing and validation",
    "Documentation and archive",
]

PT_SENTENCES = [
    "O {a} do {b} precisa de revisão antes do próximo marco.",
    "A equipe decidiu adiar o {a} para a segunda fase do {b}.",
    "Discutimos o {a} com o fornecedor e o {b} fica para setembro.",
    "O {a} apresentado na reunião depende do resultado do {b}.",
    "Depois de medir o {a}, o {b} melhorou dez por cento.",
    "O {a} e o {b} foram aprovados pela comissão sem reservas.",
    "Falta documentar o {a} e fechar o orçamento do {b}.",
    "O novo {a} substitui o anterior; o {b} permanece em produção.",
    "Precisamos de mais dois meses para o {a} e o {b}.",
    "A avaliação do {a} ficou positiva; o {b} precisa de outro protótipo.",
    "O {a} foi integrado com o {b} no ambiente de testes.",
    "Registei as métricas do {a} no relatório do {b}.",
    "O cliente pediu uma demonstração do {a} e do {b}.",
    "A versão atual do {a} corrige os problemas do {b}.",
    "O cronograma do {a} coincide com a migração do {b}.",
    "Para o {a}, reutilizámos a arquitetura do {b}.",
]

EN_SENTENCES = [
    "The {a} of the {b} needs another review before the next milestone.",
    "The team decided to postpone the {a} to the second phase of the {b}.",
    "We discussed the {a} with the supplier; the {b} moves to September.",
    "The {a} shown at the meeting depends on the outcome of the {b}.",
    "After measuring the {a}, the {b} improved by ten percent.",
    "The {a} and the {b} were approved by the committee without remarks.",
    "We still need to document the {a} and close the {b} budget.",
    "The new {a} replaces the old one; the {b} stays in production.",
    "We need two more months for the {a} and the {b}.",
    "The assessment of the {a} was positive; the {b} needs another prototype.",
    "The {a} was integrated with the {b} in the staging environment.",
    "I recorded the {a} metrics in the {b} report.",
    "The client asked for a demo of the {a} and the {b}.",
    "The current version of the {a} fixes the problems of the {b}.",
    "The {a} schedule coincides with the {b} migration.",
    "For the {a} we reused the architecture of the {b}.",
]

MONTHS = [
    "janeiro",
    "fevereiro",
    "março",
    "abril",
    "maio",
    "junho",
    "julho",
    "agosto",
    "setembro",
    "outubro",
    "novembro",
    "dezembro",
]

TOPIC_TAGS = [
    "planeamento",
    "testes",
    "compras",
    "infraestrutura",
    "pesquisa",
    "laboratorio",
    "documentacao",
    "reunioes",
    "financas",
    "arquivo",
]

_DOC_TIMES = "D:20260101000000"
_OFFICE_EPOCH = datetime(2026, 1, 1)


def plan_counts(total: int) -> dict[str, int]:
    """Split ``total`` files into the per-format counts (80/10/5/3/2 mix).

    Args:
        total: Number of files to generate.

    Returns:
        ``{"md": ..., "txt": ..., "pdf": ..., "docx": ..., "pptx": ...}``
        summing to ``total`` (the rounding rest lands on the md bucket).
    """
    shares = {"md": 0.80, "txt": 0.10, "pdf": 0.05, "docx": 0.03, "pptx": 0.02}
    counts = {name: int(total * share) for name, share in shares.items()}
    counts["md"] += total - sum(counts.values())
    return counts


def build_folders(out: Path) -> list[Path]:
    """Create the folder tree (projects, areas, archive) and return it.

    Args:
        out: Corpus root directory (created when missing).

    Returns:
        The list of folders that receive files, roughly 200 entries and up
        to four levels deep (``projects/<p>/<mid>/<leaf>``).
    """
    out.mkdir(parents=True, exist_ok=True)
    folders: list[Path] = []
    for project in PROJECTS:
        base = out / "projects" / project
        base.mkdir(parents=True, exist_ok=True)
        folders.append(base)
        for mid in ("planos", "pesquisas", "reunioes"):
            mid_dir = base / mid
            mid_dir.mkdir(exist_ok=True)
            folders.append(mid_dir)
            for leaf in ("2025", "2026", "rascunhos"):
                leaf_dir = mid_dir / leaf
                leaf_dir.mkdir(exist_ok=True)
                folders.append(leaf_dir)
    for area in AREAS:
        base = out / "areas" / area
        base.mkdir(parents=True, exist_ok=True)
        folders.append(base)
        for mid in ("notas", "decisoes"):
            mid_dir = base / mid
            mid_dir.mkdir(exist_ok=True)
            folders.append(mid_dir)
            for leaf in ("geral", "pendentes"):
                leaf_dir = mid_dir / leaf
                leaf_dir.mkdir(exist_ok=True)
                folders.append(leaf_dir)
    for project in PROJECTS:
        base = out / "archive" / f"{project}-antigo"
        base.mkdir(parents=True, exist_ok=True)
        folders.append(base)
        for leaf in ("2024", "2025"):
            leaf_dir = base / leaf
            leaf_dir.mkdir(exist_ok=True)
            folders.append(leaf_dir)
    return folders


def _sentence(rng: random.Random, templates: list[str], words: list[str]) -> str:
    """Render one sentence from a template with two word slots."""
    return rng.choice(templates).format(a=rng.choice(words), b=rng.choice(words))


def _paragraph(rng: random.Random, templates: list[str], words: list[str]) -> str:
    """Render one 4-6 sentence paragraph."""
    return " ".join(_sentence(rng, templates, words) for _ in range(rng.randint(4, 6)))


def _doc_language(rng: random.Random) -> str:
    """Pick the document language: pt, en or mixed."""
    return rng.choices(["pt", "en", "mixed"], weights=[0.45, 0.35, 0.20])[0]


def _language_parts(rng: random.Random, language: str) -> tuple[list[str], list[str], list[str]]:
    """Return (section titles, sentence templates, word pool) for a language."""
    if language == "pt":
        return PT_SECTIONS, PT_SENTENCES, PT_WORDS
    if language == "en":
        return EN_SECTIONS, EN_SENTENCES, EN_WORDS
    return PT_SECTIONS + EN_SECTIONS, PT_SENTENCES + EN_SENTENCES, PT_WORDS + EN_WORDS


def _sections(rng: random.Random, language: str, count: int) -> list[tuple[str, list[str]]]:
    """Build ``count`` (title, paragraphs) sections for the language."""
    titles, templates, words = _language_parts(rng, language)
    sections: list[tuple[str, list[str]]] = []
    for title in rng.sample(titles, min(count, len(titles))):
        body = [_paragraph(rng, templates, words) for _ in range(rng.randint(2, 3))]
        sections.append((title, body))
    while len(sections) < count:
        title = f"{rng.choice(titles)} ({len(sections) + 1})"
        body = [_paragraph(rng, templates, words) for _ in range(rng.randint(2, 3))]
        sections.append((title, body))
    return sections


def _doc_date(rng: random.Random) -> str:
    """Return a plausible ISO date in 2024-2026."""
    return f"{rng.randint(2024, 2026)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def _file_stem(rng: random.Random, used: set[str]) -> str:
    """Return a unique file stem built from two or three words and a date."""
    while True:
        pool = EN_WORDS + PT_WORDS
        words = [rng.choice(pool).replace(" ", "-") for _ in range(2)]
        stem = "-".join(words)
        if rng.random() < 0.4:
            stem = f"{stem}-{rng.randint(2024, 2026)}"
        if stem not in used:
            used.add(stem)
            return stem


def _tags_for(rng: random.Random, folder: Path, root: Path) -> list[str]:
    """Derive folder/project tags for a file placed in ``folder``."""
    tags: list[str] = []
    rel = folder.relative_to(root).parts
    if len(rel) >= 2 and rel[0] == "projects":
        tags.append(f"project:{rel[1]}")
    if len(rel) >= 2 and rel[0] == "areas":
        tags.append(f"area:{rel[1]}")
    tags.append(rng.choice(TOPIC_TAGS))
    return tags


def _write_markdown(
    path: Path,
    rng: random.Random,
    *,
    title: str,
    tags: list[str],
    supersedes: str | None,
    overlap_sections: list[tuple[str, list[str]]] | None,
) -> list[tuple[str, list[str]]]:
    """Write one Markdown file and return its sections (for `_v2` overlap)."""
    language = _doc_language(rng)
    if overlap_sections is None:
        sections = _sections(rng, language, rng.randint(3, 15))
    else:
        keep = rng.uniform(0.70, 0.90)
        _, templates, words = _language_parts(rng, language)
        sections = [
            (title_i, body_i)
            if rng.random() < keep
            else (title_i, [_paragraph(rng, templates, words) for _ in range(2)])
            for title_i, body_i in overlap_sections
        ]
    frontmatter = ["---", f"title: {title}", f"date: {_doc_date(rng)}"]
    if tags:
        frontmatter.append("tags:")
        frontmatter.extend(f"  - {tag}" for tag in tags)
    if supersedes:
        frontmatter.append(f"supersedes: {supersedes}")
    frontmatter.append("---")
    parts = ["\n".join(frontmatter), "", f"# {title}", ""]
    for section_title, paragraphs in sections:
        parts.append(f"## {section_title}")
        parts.append("")
        parts.extend(paragraphs)
        parts.append("")
    path.write_text("\n".join(parts), encoding="utf-8")
    return sections


def _write_text(path: Path, rng: random.Random) -> None:
    """Write one plain-text note (no frontmatter)."""
    parts: list[str] = [f"Nota: {_sentence(rng, PT_SENTENCES, PT_WORDS)}", ""]
    templates, words = PT_SENTENCES + EN_SENTENCES, PT_WORDS + EN_WORDS
    for _ in range(rng.randint(4, 10)):
        parts.append(_paragraph(rng, templates, words))
        parts.append("")
    path.write_text("\n".join(parts), encoding="utf-8")


def _pdf_flow(
    doc: fitz.Document, title: str, lines: list[tuple[str, float]], max_pages: int
) -> None:
    """Flow ``(text, fontsize)`` lines across at most ``max_pages`` pages."""
    rect = fitz.Rect(50, 60, 545, 780)
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 25, 545, 50), title, fontname="helv", fontsize=14)
    cursor = 60.0
    pages_left = max_pages - 1
    for text, fontsize in lines:
        while True:
            if cursor >= rect.y1 - fontsize * 2:
                if pages_left <= 0:
                    return
                page = doc.new_page()
                page.insert_textbox(fitz.Rect(50, 25, 545, 50), title, fontname="helv", fontsize=11)
                cursor = 60.0
                pages_left -= 1
            spent = page.insert_textbox(
                fitz.Rect(rect.x0, cursor, rect.x1, rect.y1),
                text,
                fontname="helv",
                fontsize=fontsize,
            )
            if spent >= 0:
                cursor += (rect.y1 - cursor - spent) + 14
                break
            if pages_left <= 0:
                return
            page = doc.new_page()
            page.insert_textbox(fitz.Rect(50, 25, 545, 50), title, fontname="helv", fontsize=11)
            cursor = 60.0
            pages_left -= 1


def _write_pdf(path: Path, rng: random.Random, title: str) -> None:
    """Write one PDF report with 2-10 pages using PyMuPDF."""
    language = _doc_language(rng)
    sections = _sections(rng, language, rng.randint(2, 5))
    lines: list[tuple[str, float]] = []
    for section_title, paragraphs in sections:
        lines.append((section_title, 12.0))
        for paragraph in paragraphs:
            lines.append((paragraph, 10.0))
    doc = fitz.open()
    doc.set_metadata(
        {
            "producer": "mynotes-perf",
            "creator": "mynotes-perf",
            "creationDate": _DOC_TIMES,
            "modDate": _DOC_TIMES,
        }
    )
    _pdf_flow(doc, title, lines, rng.randint(2, 10))
    doc.save(str(path), deflate=True)
    doc.close()


def _write_docx(path: Path, rng: random.Random, title: str) -> None:
    """Write one DOCX document with headings and paragraphs."""
    document = docx.Document()
    document.add_heading(title, level=0)
    for section_title, paragraphs in _sections(rng, _doc_language(rng), rng.randint(3, 6)):
        document.add_heading(section_title, level=1)
        for paragraph in paragraphs:
            document.add_paragraph(paragraph)
    document.core_properties.created = _OFFICE_EPOCH
    document.core_properties.modified = _OFFICE_EPOCH
    document.save(str(path))


def _write_pptx(path: Path, rng: random.Random, title: str) -> None:
    """Write one PPTX deck with 4-10 slides."""
    deck = pptx.Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[0])
    slide.shapes.title.text = title
    slide.placeholders[1].text = _sentence(rng, EN_SENTENCES, EN_WORDS)
    for section_title, paragraphs in _sections(rng, _doc_language(rng), rng.randint(3, 9)):
        slide = deck.slides.add_slide(deck.slide_layouts[1])
        slide.shapes.title.text = section_title
        body = slide.placeholders[1].text_frame
        body.text = paragraphs[0]
        for paragraph in paragraphs[1:3]:
            body.add_paragraph().text = paragraph
    deck.core_properties.created = _OFFICE_EPOCH
    deck.core_properties.modified = _OFFICE_EPOCH
    deck.save(str(path))


def generate(out: Path, files: int, seed: int) -> dict[str, object]:
    """Generate the synthetic corpus and return the summary counters.

    Args:
        out: Corpus root directory (deleted first when it exists; refuses
            the repository, the home directory and any of their ancestors).
        files: Number of files to generate.
        seed: RNG seed (deterministic output).

    Returns:
        Counters: per-format file counts, folders, v2 successors, duplicates
        and supersedes links.
    """
    resolved = out.resolve()
    protected = {Path.home().resolve(), Path.cwd().resolve()}
    if resolved in protected or any(root in resolved.parents for root in protected):
        raise SystemExit(f"refusing to generate into {resolved}")
    rng = random.Random(seed)
    if out.exists():
        shutil.rmtree(out)
    folders = build_folders(out)
    used_stems: set[str] = set()
    counts = plan_counts(files)
    plan: list[str] = []
    for name, count in counts.items():
        plan.extend([name] * count)
    rng.shuffle(plan)

    written_by_format: dict[str, list[Path]] = {name: [] for name in counts}
    md_bases: list[tuple[Path, list[tuple[str, list[str]]], list[str]]] = []
    summary: dict[str, object] = {
        "folders": len(folders),
        "files": 0,
        "v2_successors": 0,
        "duplicates": 0,
        "supersedes": 0,
        "counts": {},
    }

    weights = [3 if len(folder.relative_to(out).parts) >= 3 else 1 for folder in folders]
    for fmt in plan:
        folder = rng.choices(folders, weights=weights)[0]
        if written_by_format[fmt] and rng.random() < 0.02:
            source = rng.choice(written_by_format[fmt])
            target = folder / f"{source.stem}-copia{source.suffix}"
            shutil.copyfile(source, target)
            written_by_format[fmt].append(target)
            summary["duplicates"] += 1
            summary["files"] += 1
            continue
        stem = _file_stem(rng, used_stems)
        tags = _tags_for(rng, folder, out)
        title = stem.replace("-", " ").capitalize()
        if fmt == "md":
            roll = rng.random()
            if roll < 0.05 and md_bases:
                base_path, base_sections, base_tags = md_bases[
                    rng.randrange(max(1, len(md_bases) // 2))
                ]
                v2_name = f"{Path(base_path).stem}_v2"
                suffix = 2
                while v2_name in used_stems:
                    suffix += 1
                    v2_name = f"{Path(base_path).stem}_v{suffix}"
                used_stems.add(v2_name)
                path = folder / f"{v2_name}.md"
                title = f"{Path(base_path).stem.replace('-', ' ').capitalize()} v2"
                _write_markdown(
                    path,
                    rng,
                    title=title,
                    tags=base_tags,
                    supersedes=None,
                    overlap_sections=base_sections,
                )
                summary["v2_successors"] += 1
            elif roll < 0.06 and md_bases:
                supersedes_rel = md_bases[rng.randrange(len(md_bases))][0].relative_to(out)
                path = folder / f"{stem}.md"
                _write_markdown(
                    path,
                    rng,
                    title=title,
                    tags=tags,
                    supersedes=str(supersedes_rel),
                    overlap_sections=None,
                )
                summary["supersedes"] += 1
            else:
                path = folder / f"{stem}.md"
                sections = _write_markdown(
                    path,
                    rng,
                    title=title,
                    tags=tags,
                    supersedes=None,
                    overlap_sections=None,
                )
                md_bases.append((path, sections, tags))
        elif fmt == "txt":
            path = folder / f"{stem}.txt"
            _write_text(path, rng)
        elif fmt == "pdf":
            path = folder / f"{stem}.pdf"
            _write_pdf(path, rng, title)
        elif fmt == "docx":
            path = folder / f"{stem}.docx"
            _write_docx(path, rng, title)
        else:
            path = folder / f"{stem}.pptx"
            _write_pptx(path, rng, title)
        written_by_format[fmt].append(path)
        summary["files"] += 1
    summary["counts"] = {fmt: len(paths) for fmt, paths in written_by_format.items()}
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse flags, generate, print the JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="Corpus root directory.")
    parser.add_argument("--files", type=int, default=20000, help="Number of files.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    args = parser.parse_args(argv)
    summary = generate(args.out, args.files, args.seed)
    json.dump(summary, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
