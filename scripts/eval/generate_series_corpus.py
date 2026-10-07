"""Synthetic *meeting-series* corpus and golden set v4.

Meeting notes, dailies and status reports come in series: one note per week with
the same title, the same sections and mostly the same items, where only a few
values change from week to week, so a question about one week (or about when a
value changed) is easily answered from the wrong week. This corpus reproduces the
pattern deterministically (no LLM calls, invented names):

- ``meetings/<series>/<YYYY-MM-DD>.md``: one note per week per series, title
  ``<Series title> - <date>``, with a *Status* section listing the tracked items and
  their current value (the same lines repeat week after week), a *Decisões* section
  with the week's unique decision, and a *Pendentes* list of boilerplate;
- tracked items change value at planned weeks ("Deploy do Motor de Busca: dev" becomes
  "DEMO" in week 9), so a series carries a history;
- distractor series share people and item names.

Golden set v4 question types (``category``):

- ``series-date``: a fact of one named week ("Na reunião de 2026-03-09, ...");
- ``series-latest``: the current value of a tracked item (the newest note that states it);
- ``series-asof``: the value in force at a date (``as-of:`` token);
- ``series-change``: the week a tracked item changed to a given value;
- ``control``: the unique decision of a week, which no other note shares.

    uv run python scripts/eval/generate_series_corpus.py --out /tmp/mynotes-series-corpus \\
        --golden-out eval/golden_set_v4.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

_SERIES = [
    ("Reuniao semanal Atlas", "atlas", "projeto Atlas"),
    ("Reuniao semanal Boreal", "boreal", "projeto Boreal"),
    ("Daily equipa de plataforma", "plataforma", "equipa de plataforma"),
    ("Status dos dashboards", "dashboards", "dashboards"),
    ("Reuniao de coordenacao Norte", "norte", "coordenacao Norte"),
    ("Ponto de situacao Catalogo", "catalogo", "Catalogo"),
]
_PEOPLE = [
    "Marta Quelhas",
    "Rui Tavares",
    "Inês Barroso",
    "Nuno Faria",
    "Tomas Correia",
    "Leonor Vaz",
]
_ITEMS = [
    ("Deploy do Motor de Busca", ["dev", "DEMO", "producao"]),
    ("Deploy do Conversor", ["em desenvolvimento", "dev", "DEMO"]),
    ("Dashboards de contadores", ["por iniciar", "em curso", "entregue"]),
    ("Documentacao da API", ["pendente", "em revisao", "publicada"]),
    ("Ambiente do tenant de demo", ["a preparar", "pronto", "entregue ao cliente"]),
    ("Migracao do mailserver", ["planeada", "em curso", "concluida"]),
]
_OWNED_TASKS = [
    "device commissioning",
    "traducao do conteudo dinamico",
    "deploy de tenants",
    "estatisticas da base de dados",
    "landing page",
    "notificacoes nas apps",
]
_DECISIONS = [
    "adotar o {thing} como padrao",
    "adiar a entrega de {thing} uma semana",
    "retirar o {thing} do ambito desta fase",
    "pedir acesso ao {thing} ao fornecedor",
    "congelar alteracoes ao {thing} ate ao fecho",
    "duplicar os testes do {thing} antes do deploy",
]
_THINGS = [
    "cluster de Kubernetes",
    "repositorio de ambientes",
    "servico de notificacoes",
    "modulo de exportacao",
    "painel de administracao",
    "indice de decisoes",
    "camada de sincronizacao",
    "motor de pesquisa",
    "quadro de tarefas",
    "gestor de citacoes",
    "vista de nuvem de etiquetas",
    "catalogo de dispositivos",
]
_PENDING = [
    "Atualizar a marcacao das ferias",
    "Rever os pacotes com vulnerabilidades",
    "Preparar a demo para o cliente",
    "Validar o plano de reversao com os donos das aplicacoes",
    "Confirmar a janela de manutencao com operacoes",
]
_START = datetime(2026, 1, 5, tzinfo=UTC)  # a Monday


@dataclass
class Week:
    date: str
    rel_path: str
    title: str
    values: dict[str, str]
    owners: dict[str, str]
    decision: str
    decision_thing: str


@dataclass
class Series:
    key: str
    title: str
    subject: str
    weeks: list[Week] = field(default_factory=list)
    changes: list[dict[str, str]] = field(default_factory=list)  # item, value, date, previous


@dataclass
class Manifest:
    seed: int
    series: list[Series] = field(default_factory=list)


def _write(path: Path, lines: list[str], date: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stamp = datetime.fromisoformat(date).replace(tzinfo=UTC, hour=10).timestamp()
    os.utime(path, (stamp, stamp))


def plan(rng: random.Random, *, series: int, weeks: int) -> Manifest:
    manifest = Manifest(seed=0)
    for key_index in range(series):
        title, key, subject = _SERIES[key_index % len(_SERIES)]
        if key_index >= len(_SERIES):
            key = f"{key}-{key_index}"
            title = f"{title} {key_index}"
        items = rng.sample(_ITEMS, 3)
        tasks = rng.sample(_OWNED_TASKS, 3)
        # each tracked item changes value at two distinct weeks (never the first week)
        change_weeks = {name: sorted(rng.sample(range(2, weeks), 2)) for name, _values in items}
        owner_change_week = {task: rng.randrange(3, weeks) for task in tasks}
        owners_before = {task: rng.choice(_PEOPLE) for task in tasks}
        owners_after = {
            task: rng.choice([p for p in _PEOPLE if p != owners_before[task]]) for task in tasks
        }
        things = rng.sample(_THINGS, len(_THINGS))
        current = Series(key=key, title=title, subject=subject)
        previous_values: dict[str, str] = {}
        for week_index in range(weeks):
            date = (_START + timedelta(days=7 * week_index)).strftime("%Y-%m-%d")
            values: dict[str, str] = {}
            for name, stages in items:
                stage = sum(1 for w in change_weeks[name] if week_index >= w)
                values[name] = stages[stage]
                if previous_values.get(name) not in (None, stages[stage]):
                    current.changes.append(
                        {
                            "item": name,
                            "value": stages[stage],
                            "date": date,
                            "previous": previous_values[name],
                        }
                    )
            previous_values = dict(values)
            owners = {
                task: (owners_after if week_index >= owner_change_week[task] else owners_before)[
                    task
                ]
                for task in tasks
            }
            thing = things[week_index % len(things)]
            decision = _DECISIONS[(week_index + key_index) % len(_DECISIONS)].format(thing=thing)
            current.weeks.append(
                Week(
                    date=date,
                    rel_path=f"meetings/{key}/{date}.md",
                    title=f"{title} - {date}",
                    values=values,
                    owners=owners,
                    decision=decision,
                    decision_thing=thing,
                )
            )
        manifest.series.append(current)
    return manifest


def write_corpus(out: Path, manifest: Manifest, rng: random.Random) -> None:
    for series in manifest.series:
        for week in series.weeks:
            lines = [
                "---",
                f"title: {week.title}",
                f"date: {week.date}",
                "tags:",
                f"  - project:{series.key}",
                "  - meetings",
                "---",
                "",
                f"# {week.title}",
                "",
                "## Status",
                "",
                *(f"- {name}: {value}" for name, value in week.values.items()),
                *(f"- Responsavel por {task}: {owner}" for task, owner in week.owners.items()),
                "",
                "## Decisoes",
                "",
                f"- Decidiu-se {week.decision}.",
                "",
                "## Pendentes",
                "",
                *(f"- [ ] {item}" for item in rng.sample(_PENDING, 3)),
                "",
            ]
            _write(out / week.rel_path, lines, week.date)


def build_golden(manifest: Manifest, rng: random.Random, note: str) -> dict[str, Any]:
    questions: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    def add(category: str, question: str, docs: list[str], answer: str, tags: list[str]) -> None:
        counts[category] = counts.get(category, 0) + 1
        questions.append(
            {
                "id": f"s{len(questions) + 1:03d}",
                "question": question,
                "language": "pt",
                "category": category,
                "tags": tags,
                "filters": {"tags": [], "folders": []},
                "expected_docs": docs,
                "reference_answer": answer,
            }
        )

    for series in manifest.series:
        tag = f"series:{series.key}"
        weeks = series.weeks
        # series-date: an owner fact of two named weeks (one before, one after an owner change)
        for week in (weeks[1], weeks[-2]):
            task = rng.choice(list(week.owners))
            add(
                "series-date",
                f"Na reuniao de {week.date} ({series.title}), quem ficou responsavel por {task}?",
                [week.rel_path],
                week.owners[task],
                [tag, "kind:owner"],
            )
        # series-latest: the current value of each tracked item; every week since the last
        # change states it, so any of those notes answers correctly
        for name in weeks[-1].values:
            final = weeks[-1].values[name]
            docs = [w.rel_path for w in weeks if w.values[name] == final]
            add(
                "series-latest",
                f"Qual e o estado atual de '{name}' na {series.title}?",
                docs,
                final,
                [tag, "kind:latest"],
            )
        # series-asof: the value in force at a date between two changes (every week up to
        # the as-of date that states that value is a correct source)
        for change in series.changes[:2]:
            change_week = next(i for i, w in enumerate(weeks) if w.date == change["date"])
            asof_week = weeks[max(0, change_week - 1)]
            docs = [
                w.rel_path
                for w in weeks[:change_week]
                if w.values[change["item"]] == change["previous"]
            ]
            add(
                "series-asof",
                f"as-of: {asof_week.date} Qual era o estado de '{change['item']}' "
                f"na {series.title}?",
                docs,
                change["previous"],
                [tag, "kind:asof"],
            )
        # series-change: the week an item changed to a value
        for change in series.changes[:2]:
            add(
                "series-change",
                f"Em que reuniao da {series.title} '{change['item']}' passou a {change['value']}?",
                [f"meetings/{series.key}/{change['date']}.md"],
                f"Na reuniao de {change['date']}",
                [tag, "kind:change"],
            )
        # control: a unique decision
        week = rng.choice(weeks[2:-2])
        add(
            "control",
            f"O que se decidiu sobre o {week.decision_thing} na {series.title}?",
            [week.rel_path],
            week.decision,
            [tag, "kind:decision"],
        )
    return {
        "meta": {
            "created": datetime.now(UTC).strftime("%Y-%m-%d"),
            "corpus_note": note,
            "review": (
                "GENERATED (not human-reviewed): questions are correct by construction from the "
                "synthetic corpus manifest; review the wording, not the answers."
            ),
            "counts": counts,
        },
        "questions": questions,
    }


def generate(out: Path, *, seed: int, series: int, weeks: int, golden_out: Path | None) -> dict:
    if out.exists():
        for child in sorted(out.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    manifest = plan(rng, series=series, weeks=weeks)
    manifest.seed = seed
    write_corpus(out, manifest, rng)
    (out / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if golden_out is not None:
        note = f"synthetic meeting-series corpus: {series} series x {weeks} weeks, seed {seed}"
        golden = build_golden(manifest, random.Random(seed + 1), note)
        header = (
            "# Golden set v4 - synthetic meeting-series corpus\n"
            "# (scripts/eval/generate_series_corpus.py).\n"
            "# GENERATED, not human-reviewed. Answers are correct by construction; review the\n"
            "# wording. Regenerate with the same --seed to reproduce. Never edit between two\n"
            "# runs you want to compare.\n"
        )
        golden_out.write_text(
            header + yaml.safe_dump(golden, sort_keys=False, allow_unicode=True, width=100),
            encoding="utf-8",
        )
    files = [p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"]
    return {"out": str(out), "series": series, "weeks": weeks, "files": len(files)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("/tmp/mynotes-series-corpus"))
    parser.add_argument("--golden-out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--series", type=int, default=6)
    parser.add_argument("--weeks", type=int, default=20)
    args = parser.parse_args()
    print(
        json.dumps(
            generate(
                args.out,
                seed=args.seed,
                series=args.series,
                weeks=args.weeks,
                golden_out=args.golden_out,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
