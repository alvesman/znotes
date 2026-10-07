"""Synthetic *multi-hop* corpus and golden set v3 for the bridge-question eval.

A multi-hop question reaches its answer through an intermediate note: "who maintains
the plugin that the note 'Tag cloud view' uses?" names the *bridge* note (the tag
cloud note says which plugin it uses) while the answer (the maintainer) lives in the
plugin's own note, which shares no words with the question. Retrieval by the
question's words alone cannot find the answer note; something has to follow the
bridge (query decomposition, the knowledge graph, ...).

The corpus is deterministic and templated (no LLM calls) and every name is invented,
so the answering model cannot know the facts from memory:

- **tool notes** (``tools/<tool>.md``): one per invented tool, stating its facts
  (maintainer, licence, language, config file, port) in a generic description;
- **bridge notes** (``projects/<project>/<usage>.md``): a project note that says
  which tool it uses for one purpose, without stating any of the tool's facts;
- **component notes** (``projects/<project>/<component>.md``): for 3-hop chains,
  the project note points at a component note which names the tool;
- **distractor notes** (``areas/...``): unrelated notes sharing people, licences and
  languages with the tool notes.

Golden set v3: 2-hop questions in two phrasings (quoting the bridge note's title, or
describing the usage and project), 3-hop questions, and 1-hop controls that name the
tool. Every question carries ``bridge_docs`` so failures can be classed as "bridge not
followed" (answer note absent) or "answer note ranked low".

    uv run python scripts/eval/generate_multihop_corpus.py --out /tmp/mynotes-multihop-corpus \\
        --golden-out eval/golden_set_v3.yaml
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

FACTS = ("maintainer", "license", "language", "config_file", "port")

_TOOL_NAMES = [
    "Tagloom",
    "Quillboard",
    "Notegrid",
    "Inkwell",
    "Folios",
    "Marginalia",
    "Scribekit",
    "Vellum",
    "Lattice",
    "Pagewright",
    "Draftline",
    "Glyphbox",
    "Memoranda",
    "Stanza",
    "Codexa",
    "Threadnote",
    "Linkforge",
    "Outlinely",
    "Papyra",
    "Tabular",
    "Cardstock",
    "Kanbanist",
    "Timeglass",
    "Backlinker",
    "Snipster",
    "Mindmapr",
    "Annotely",
    "Clipmark",
    "Refstack",
    "Journalix",
    "Slidecraft",
    "Formulary",
]
_PEOPLE = [
    "Ana Peixoto",
    "Rui Tavares",
    "Marta Carvalho",
    "Tiago Nunes",
    "Ines Barros",
    "Pedro Antunes",
    "Sofia Leitao",
    "Nuno Fonseca",
    "Carla Mendes",
    "Joao Ribeiro",
    "Beatriz Lopes",
    "Hugo Martins",
    "Rita Figueiredo",
    "Miguel Cordeiro",
    "Teresa Pinho",
    "Vasco Almeida",
    "Helena Guerreiro",
    "Bruno Sequeira",
    "Leonor Matos",
    "Diogo Pacheco",
    "Clara Vicente",
    "Andre Rocha",
    "Patricia Serra",
    "Goncalo Brito",
    "Filipa Santos",
    "Ricardo Neves",
    "Mariana Teles",
    "Paulo Esteves",
    "Catarina Lemos",
    "Jorge Valente",
    "Sara Moniz",
    "Luis Baptista",
]
_LICENSES = ["MIT", "Apache-2.0", "GPL-3.0", "BSD-3-Clause", "MPL-2.0", "ISC"]
_LANGUAGES = ["Rust", "Go", "TypeScript", "Python", "Kotlin", "Elixir", "Zig", "Dart"]
_PROJECTS = [
    "Orion",
    "Vega",
    "Lince",
    "Tejo",
    "Alvor",
    "Sado",
    "Monchique",
    "Cintra",
    "Douro",
    "Minho",
    "Zezere",
    "Tamega",
    "Lima",
    "Cavado",
    "Mira",
    "Guadiana",
    "Arade",
    "Mondego",
    "Nabao",
    "Vouga",
    "Ave",
    "Leca",
    "Neiva",
    "Ancora",
    "Coa",
    "Sabor",
    "Tua",
    "Paiva",
    "Dao",
    "Alva",
    "Ceira",
    "Zambujal",
    "Odelouca",
    "Seda",
    "Caia",
    "Sorraia",
    "Almansor",
    "Raia",
    "Degebe",
    "Xarrama",
]
# Usages: (pt phrase, en phrase, pt note title, en note title). The tool note never uses
# these words, so the answer note cannot be found from the usage alone.
_USAGES = [
    (
        "a vista de nuvem de etiquetas",
        "the tag cloud view",
        "Vista de nuvem de etiquetas",
        "Tag cloud view",
    ),
    (
        "a revisao semanal das notas",
        "the weekly note review",
        "Rotina de revisao semanal",
        "Weekly review routine",
    ),
    ("o diario de reunioes", "the meeting journal", "Diario de reunioes", "Meeting journal"),
    ("a exportacao de relatorios", "report export", "Exportacao de relatorios", "Report export"),
    ("o quadro de tarefas", "the task board", "Quadro de tarefas", "Task board"),
    ("a captura de ideias rapidas", "quick idea capture", "Captura de ideias", "Idea capture"),
    ("o indice de decisoes", "the decision index", "Indice de decisoes", "Decision index"),
    ("a linha temporal do projeto", "the project timeline", "Linha temporal", "Project timeline"),
    (
        "as citacoes bibliograficas",
        "bibliographic citations",
        "Gestao de citacoes",
        "Citation workflow",
    ),
    ("os mapas de conceitos", "concept maps", "Mapas de conceitos", "Concept maps"),
    (
        "a publicacao de apresentacoes",
        "slide publishing",
        "Publicacao de apresentacoes",
        "Slide publishing",
    ),
    ("o controlo de formularios", "form handling", "Controlo de formularios", "Form handling"),
]
_COMPONENTS = [
    ("o modulo de exportacao", "the export module", "Modulo de exportacao", "Export module"),
    (
        "o servico de notificacoes",
        "the notification service",
        "Servico de notificacoes",
        "Notification service",
    ),
    ("a camada de sincronizacao", "the sync layer", "Camada de sincronizacao", "Sync layer"),
    ("o painel de administracao", "the admin panel", "Painel de administracao", "Admin panel"),
    (
        "o motor de pesquisa interno",
        "the internal search engine",
        "Motor de pesquisa",
        "Search engine",
    ),
]
_WINDOW = (datetime(2025, 3, 1, tzinfo=UTC), datetime(2026, 8, 31, tzinfo=UTC))


@dataclass
class Tool:
    name: str
    slug: str
    rel_path: str
    facts: dict[str, str]
    date: str


@dataclass
class Bridge:
    """A project note that uses one tool for one purpose (and, for 3-hop, via a component)."""

    id: str
    project: str
    language: str
    usage: tuple[str, str, str, str]
    tool: str
    rel_path: str
    title: str
    date: str
    component: tuple[str, str, str, str] | None = None
    component_path: str | None = None
    component_title: str | None = None
    hops: int = 2


@dataclass
class Manifest:
    kind: str = "multihop"
    seed: int = 0
    tools: list[Tool] = field(default_factory=list)
    bridges: list[Bridge] = field(default_factory=list)
    distractors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- text


def _slug(text: str) -> str:
    return text.lower().replace(" ", "-")


def _date(rng: random.Random) -> str:
    start, end = _WINDOW
    return (start + timedelta(days=rng.randrange((end - start).days + 1))).strftime("%Y-%m-%d")


def _set_mtime(path: Path, date: str) -> None:
    stamp = datetime.fromisoformat(date).replace(tzinfo=UTC, hour=10).timestamp()
    os.utime(path, (stamp, stamp))


def _write(path: Path, lines: list[str], date: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    _set_mtime(path, date)


_TOOL_NOTE = {
    "pt": [
        "## Descricao",
        "{name} e uma ferramenta de produtividade para o cofre de notas, distribuida como extensao "
        "e configurada por ficheiro. Esta nota reune os dados de referencia da ferramenta.",
        "## Dados de referencia",
        "A ferramenta {name} e mantida por {maintainer}.",
        "A licenca da {name} e {license}.",
        "A {name} esta escrita em {language}.",
        "O ficheiro de configuracao da {name} chama-se {config_file}.",
        "O porto por omissao da {name} e {port}.",
        "## Notas de instalacao",
        "Instalar a partir do repositorio oficial e reiniciar o cofre. As atualizacoes sao "
        "verificadas no arranque; em caso de erro, apagar a cache local e repetir.",
    ],
    "en": [
        "## Description",
        "{name} is a productivity tool for the notes vault, distributed as an extension and "
        "configured through a file. This note gathers the tool's reference data.",
        "## Reference data",
        "The {name} tool is maintained by {maintainer}.",
        "The licence of {name} is {license}.",
        "{name} is written in {language}.",
        "The configuration file of {name} is called {config_file}.",
        "The default port of {name} is {port}.",
        "## Installation notes",
        "Install from the official repository and restart the vault. Updates are checked at "
        "startup; on errors, clear the local cache and retry.",
    ],
}

_BRIDGE_NOTE = {
    "pt": [
        "## Contexto",
        "Esta nota descreve como o Projeto {project} trata {usage}. Faz parte do conjunto de "
        "notas operacionais do projeto.",
        "## Ferramenta",
        "Para {usage} usamos {tool_ref}. A escolha foi feita pela equipa depois de comparar "
        "tres alternativas durante duas semanas.",
        "## Procedimento",
        "O procedimento corre todas as semanas e demora cerca de vinte minutos. Os resultados "
        "ficam na pasta partilhada do projeto e sao revistos na reuniao seguinte.",
    ],
    "en": [
        "## Context",
        "This note describes how Project {project} handles {usage}. It belongs to the project's "
        "operational notes.",
        "## Tool",
        "For {usage} we use {tool_ref}. The team chose it after comparing three alternatives "
        "over two weeks.",
        "## Procedure",
        "The procedure runs every week and takes about twenty minutes. The results go to the "
        "project's shared folder and are reviewed at the next meeting.",
    ],
}

_PROJECT_3HOP = {
    "pt": [
        "## Contexto",
        "Esta nota descreve como o Projeto {project} trata {usage}.",
        "## Arquitetura",
        'A parte tecnica fica a cargo de {component}, descrito na nota "{component_title}". '
        "A equipa reve esse componente a cada trimestre.",
        "## Procedimento",
        "O procedimento corre todas as semanas e os resultados ficam na pasta partilhada "
        "do projeto.",
    ],
    "en": [
        "## Context",
        "This note describes how Project {project} handles {usage}.",
        "## Architecture",
        'The technical part is handled by {component}, described in the note "{component_title}". '
        "The team reviews that component every quarter.",
        "## Procedure",
        "The procedure runs every week and the results go to the project's shared folder.",
    ],
}

_COMPONENT_NOTE = {
    "pt": [
        "## Papel",
        "{component_cap} do Projeto {project} liga as notas operacionais as ferramentas externas.",
        "## Implementacao",
        "{component_cap} assenta na ferramenta {tool}. A configuracao e gerida pela equipa de "
        "plataforma e revista a cada trimestre.",
    ],
    "en": [
        "## Role",
        "{component_cap} of Project {project} connects the operational notes to external tools.",
        "## Implementation",
        "{component_cap} is built on the {tool} tool. Its configuration is managed by the "
        "platform team and reviewed every quarter.",
    ],
}

_TOOL_REF = {"pt": "o plugin {tool}", "en": "the {tool} plugin"}


def _frontmatter(title: str, date: str, tags: list[str]) -> list[str]:
    return [
        "---",
        f"title: {title}",
        f"date: {date}",
        "tags:",
        *[f"  - {t}" for t in tags],
        "---",
        "",
        f"# {title}",
        "",
    ]


def _fill(lines: list[str], **values: str) -> list[str]:
    out: list[str] = []
    for line in lines:
        text = line.format(**values)
        out.append(text)
        out.append("")
    return out


# ---------------------------------------------------------------- planning and writing


def plan(rng: random.Random, *, tools: int, two_hop: int, three_hop: int) -> Manifest:
    manifest = Manifest(seed=0)
    names = list(_TOOL_NAMES)
    rng.shuffle(names)
    people = list(_PEOPLE)
    rng.shuffle(people)
    for i in range(tools):
        name = names[i]
        slug = _slug(name)
        facts = {
            "maintainer": people[i % len(people)],
            "license": rng.choice(_LICENSES),
            "language": rng.choice(_LANGUAGES),
            "config_file": f"{slug}.{rng.choice(['toml', 'yaml', 'json'])}",
            "port": str(rng.randrange(4100, 4999)),
        }
        manifest.tools.append(
            Tool(name=name, slug=slug, rel_path=f"tools/{slug}.md", facts=facts, date=_date(rng))
        )
    projects = list(_PROJECTS)
    rng.shuffle(projects)
    usages = list(_USAGES)
    index = 0
    for k in range(two_hop + three_hop):
        project = projects[k % len(projects)]
        language = "pt" if k % 2 == 0 else "en"
        usage = usages[k % len(usages)]
        tool = manifest.tools[k % len(manifest.tools)]
        title = usage[2] if language == "pt" else usage[3]
        title = f"{title} - {project}"
        folder = f"projects/{_slug(project)}"
        bridge = Bridge(
            id=f"m{k + 1:02d}",
            project=project,
            language=language,
            usage=usage,
            tool=tool.name,
            rel_path=f"{folder}/{_slug(usage[2] if language == 'pt' else usage[3])}.md",
            title=title,
            date=_date(rng),
        )
        if k >= two_hop:
            component = _COMPONENTS[index % len(_COMPONENTS)]
            index += 1
            ctitle = (component[2] if language == "pt" else component[3]) + f" - {project}"
            bridge.hops = 3
            bridge.component = component
            bridge.component_title = ctitle
            bridge.component_path = (
                f"{folder}/{_slug(component[2] if language == 'pt' else component[3])}.md"
            )
        manifest.bridges.append(bridge)
    return manifest


def write_corpus(out: Path, manifest: Manifest, rng: random.Random, distractors: int) -> None:
    for i, tool in enumerate(manifest.tools):  # tool notes alternate PT/EN by position
        language = "pt" if i % 2 == 0 else "en"
        lines = _frontmatter(tool.name, tool.date, ["tool", "reference"]) + _fill(
            _TOOL_NOTE[language], name=tool.name, **tool.facts
        )
        _write(out / tool.rel_path, lines, tool.date)
    for bridge in manifest.bridges:
        lang = bridge.language
        usage = bridge.usage[0] if lang == "pt" else bridge.usage[1]
        tags = [f"project:{_slug(bridge.project)}", "operations"]
        if bridge.hops == 2:
            lines = _frontmatter(bridge.title, bridge.date, tags) + _fill(
                _BRIDGE_NOTE[lang],
                project=bridge.project,
                usage=usage,
                tool_ref=_TOOL_REF[lang].format(tool=bridge.tool),
            )
            _write(out / bridge.rel_path, lines, bridge.date)
        else:
            assert bridge.component and bridge.component_path and bridge.component_title
            comp = bridge.component[0] if lang == "pt" else bridge.component[1]
            lines = _frontmatter(bridge.title, bridge.date, tags) + _fill(
                _PROJECT_3HOP[lang],
                project=bridge.project,
                usage=usage,
                component=comp,
                component_title=bridge.component_title,
            )
            _write(out / bridge.rel_path, lines, bridge.date)
            comp_lines = _frontmatter(bridge.component_title, bridge.date, tags) + _fill(
                _COMPONENT_NOTE[lang],
                component_cap=comp[0].upper() + comp[1:],
                project=bridge.project,
                tool=bridge.tool,
            )
            _write(out / bridge.component_path, comp_lines, bridge.date)
    # Distractors: notes that mention the same people, licences and languages.
    topics_pt = [
        "Checklist de onboarding",
        "Procedimento de compras",
        "Notas sobre backups",
        "Guia de estilo",
        "Resumo do workshop",
        "Politica de ferias",
    ]
    topics_en = [
        "Monthly cost review",
        "Incident template",
        "Laptop refresh plan",
        "Reading list",
        "Vendor criteria",
        "Room etiquette",
    ]
    fillers = {
        "pt": [
            "A nota foi revista por {person} em conjunto com a equipa.",
            "O codigo auxiliar esta em {language} e publicado com licenca {license}.",
            "Em caso de duvida, falar com {person}.",
            "O servico interno escuta no porto {port} apenas na rede do laboratorio.",
        ],
        "en": [
            "The note was reviewed by {person} together with the team.",
            "The helper code is in {language} and published under the {license} licence.",
            "If in doubt, talk to {person}.",
            "The internal service listens on port {port} on the lab network only.",
        ],
    }
    for i in range(distractors):
        lang = "pt" if i % 2 == 0 else "en"
        topics = topics_pt if lang == "pt" else topics_en
        title = f"{topics[i % len(topics)]} {i + 1}"
        rel = f"areas/{_slug(topics[i % len(topics)])}/{_slug(title)}.md"
        date = _date(rng)
        body = [
            f.format(
                person=rng.choice(_PEOPLE),
                language=rng.choice(_LANGUAGES),
                license=rng.choice(_LICENSES),
                port=rng.randrange(4100, 4999),
            )
            for f in rng.sample(fillers[lang], k=3)
        ]
        _write(out / rel, _frontmatter(title, date, ["area"]) + _fill(body), date)
        manifest.distractors.append(rel)


# ---------------------------------------------------------------- golden set

_FACT_Q = {
    "pt": {
        "maintainer": "Quem mantem {subject}?",
        "license": "Qual e a licenca de {subject}?",
        "language": "Em que linguagem esta escrita {subject}?",
        "config_file": "Como se chama o ficheiro de configuracao de {subject}?",
        "port": "Qual e o porto por omissao de {subject}?",
    },
    "en": {
        "maintainer": "Who maintains {subject}?",
        "license": "What is the licence of {subject}?",
        "language": "Which language is {subject} written in?",
        "config_file": "What is the configuration file of {subject} called?",
        "port": "What is the default port of {subject}?",
    },
}
_SUBJECT = {
    "pt": {
        "direct": "o plugin {tool}",
        "title": 'o plugin usado para {usage} na nota "{title}"',
        "descriptive": "a ferramenta que o Projeto {project} usa para {usage}",
        "three": "a ferramenta em que assenta {component} do Projeto {project}",
    },
    "en": {
        "direct": "the {tool} plugin",
        "title": 'the plugin used for {usage} in the note "{title}"',
        "descriptive": "the tool that Project {project} uses for {usage}",
        "three": "the tool that {component} of Project {project} is built on",
    },
}


def build_golden(
    manifest: Manifest, rng: random.Random, controls: int, note: str
) -> dict[str, Any]:
    tools = {t.name: t for t in manifest.tools}
    questions: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    def add(
        category: str,
        hops: int,
        language: str,
        question: str,
        tool: Tool,
        fact: str,
        bridges: list[str],
        family: str,
        style: str,
    ) -> None:
        counts[category] = counts.get(category, 0) + 1
        questions.append(
            {
                "id": f"h{len(questions) + 1:03d}",
                "question": question,
                "language": language,
                "category": category,
                "tags": [f"hops:{hops}", f"fact:{fact}", f"style:{style}", f"family:{family}"],
                "filters": {"tags": [], "folders": []},
                "expected_docs": [tool.rel_path],
                "bridge_docs": bridges,
                "reference_answer": tool.facts[fact],
            }
        )

    for i, bridge in enumerate(manifest.bridges):
        lang = bridge.language
        tool = tools[bridge.tool]
        fact = FACTS[i % len(FACTS)]
        usage = bridge.usage[0] if lang == "pt" else bridge.usage[1]
        if bridge.hops == 2:
            style = "title" if i % 2 == 0 else "descriptive"
            subject = _SUBJECT[lang][style].format(
                usage=usage, title=bridge.title, project=bridge.project
            )
            add(
                "multi-hop",
                2,
                lang,
                _FACT_Q[lang][fact].format(subject=subject),
                tool,
                fact,
                [bridge.rel_path],
                bridge.id,
                style,
            )
        else:
            comp = bridge.component[0] if lang == "pt" else bridge.component[1]
            subject = _SUBJECT[lang]["three"].format(component=comp, project=bridge.project)
            add(
                "multi-hop",
                3,
                lang,
                _FACT_Q[lang][fact].format(subject=subject),
                tool,
                fact,
                [bridge.rel_path, bridge.component_path or ""],
                bridge.id,
                "three",
            )
    for j in range(controls):
        tool = manifest.tools[(j * 3) % len(manifest.tools)]
        lang = "pt" if j % 2 == 0 else "en"
        fact = FACTS[j % len(FACTS)]
        subject = _SUBJECT[lang]["direct"].format(tool=tool.name)
        add(
            "control",
            1,
            lang,
            _FACT_Q[lang][fact].format(subject=subject),
            tool,
            fact,
            [],
            f"c{j + 1:02d}",
            "direct",
        )
    return {
        "meta": {
            "created": datetime.now(UTC).strftime("%Y-%m-%d"),
            "corpus_note": note,
            "review": "GENERATED (not human-reviewed): answers are correct by construction "
            "from the synthetic corpus manifest; review the wording, not the answers.",
            "counts": counts,
        },
        "questions": questions,
    }


# ---------------------------------------------------------------- main


def generate(
    out: Path,
    *,
    seed: int,
    tools: int,
    two_hop: int,
    three_hop: int,
    controls: int,
    distractors: int,
    golden_out: Path | None,
) -> dict[str, Any]:
    if out.exists():
        for child in sorted(out.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    manifest = plan(rng, tools=tools, two_hop=two_hop, three_hop=three_hop)
    manifest.seed = seed
    write_corpus(out, manifest, rng, distractors)
    (out / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if golden_out is not None:
        note = (
            f"synthetic multi-hop corpus: {tools} tools, {two_hop} two-hop and {three_hop} "
            f"three-hop bridges, {controls} controls, {distractors} distractors, seed {seed}"
        )
        golden = build_golden(manifest, random.Random(seed + 1), controls, note)
        header = (
            "# Golden set v3 - synthetic multi-hop corpus\n"
            "# (scripts/eval/generate_multihop_corpus.py).\n"
            "# GENERATED, not human-reviewed. Answers are correct by construction; review the\n"
            "# wording. Regenerate with the same --seed to reproduce. Never edit between two\n"
            "# runs you want to compare.\n"
        )
        golden_out.write_text(
            header + yaml.safe_dump(golden, sort_keys=False, allow_unicode=True, width=100),
            encoding="utf-8",
        )
    files = [p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"]
    return {
        "out": str(out),
        "tools": tools,
        "bridges": len(manifest.bridges),
        "files": len(files),
        "golden_out": str(golden_out) if golden_out else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("/tmp/mynotes-multihop-corpus"))
    parser.add_argument("--golden-out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--tools", type=int, default=30)
    parser.add_argument("--two-hop", type=int, default=40)
    parser.add_argument("--three-hop", type=int, default=10)
    parser.add_argument("--controls", type=int, default=10)
    parser.add_argument("--distractors", type=int, default=40)
    args = parser.parse_args()
    print(
        json.dumps(
            generate(
                args.out,
                seed=args.seed,
                tools=args.tools,
                two_hop=args.two_hop,
                three_hop=args.three_hop,
                controls=args.controls,
                distractors=args.distractors,
                golden_out=args.golden_out,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
