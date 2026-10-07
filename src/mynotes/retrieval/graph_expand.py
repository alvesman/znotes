"""Graph expansion in retrieval: expander, graph facts block (PLAN.md §9 step 4/6, task 43).

:class:`GraphExpander` is the ``graph`` fusion source: it links the question to
KG entities (exact n-grams plus the entities index), seeds the expansion with
the entities mentioned by the leading hybrid hits, fetches the 1-2 hop
subgraph (:mod:`mynotes.graph.expand`), scores it (one-hop or Personalized
PageRank) and loads the evidence chunks as hits - filtered in Python through
:func:`~mynotes.retrieval.hybrid.hit_matches_filters` so the ACL, tags and
folders of the query always hold, even for chunks that never went through a
filtered search. It adds the structural hits: documents linked from the top
hits and same-``project:`` siblings, searched with one restricted hybrid query.

:class:`GraphFacts` renders the ``graph_facts`` context block: the
top-scoring edges among the entities connected to the selected documents plus
the current attribute values of the top seed entity.

Both register themselves on import; :mod:`mynotes.retrieval.pipeline` imports
this module so a plain ``retrieve()`` call gets graph expansion (behind
``retrieval.graph_expansion``).
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from mynotes.config import get_settings
from mynotes.core.acl import filter_subgraph_by_acl, relation_has_allowed_evidence
from mynotes.db.models import (
    Chunk,
    DocumentTag,
    DocumentVersion,
    KgEntity,
    KgMention,
    KgRelation,
    KgRelationEvidence,
    Tag,
)
from mynotes.db.session import session_scope
from mynotes.graph.expand import Subgraph, evidence_chunks, fetch_subgraph, score_entities
from mynotes.graph.structural import linked_documents
from mynotes.retrieval.assemble import register_extra_block
from mynotes.retrieval.entity_link import link_entities
from mynotes.retrieval.hybrid import _to_hit, hit_matches_filters, hybrid_search
from mynotes.retrieval.pipeline import register_expander
from mynotes.retrieval.types import SearchHit
from mynotes.text.normalize import normalize_key

if TYPE_CHECKING:
    from mynotes.config import Settings
    from mynotes.retrieval.analyze import QueryAnalysis
    from mynotes.retrieval.pipeline import RetrievalResult

logger = logging.getLogger(__name__)

_SECOND_HOP_ENTITIES = 4
"""Rare entities of the first-hop bridge document followed on deep chains."""
_SECOND_HOP_MIN_WEIGHT = 0.25
"""Below this (hub) weight an entity of the bridge document is not followed."""
_SEED_MENTION_SCORE = 0.9
"""Weight of an entity mentioned by the top document, before rarity damping."""
_SEED_DOC_DECAY = 0.5
"""Each further document in the top hits halves the mention weight step."""


_CHAIN_PATTERNS = (
    r'"[^"]{3,}"',  # a quoted note title
    r"\b(usad[oa]s?|utilizad[oa]s?|referid[oa]s?|mencionad[oa]s?|descrit[oa]s?)\b",
    r"\bque\b.{0,60}\b(usa|usam|utiliza|utilizam|refere|menciona|descreve|assenta)\b",
    r"\bem que assenta\b",
    r"\bcuj[oa]s?\b",
    r"\b(used|mentioned|referenced|described|built on|relies on|based on)\b",
    r"\bthat\b.{0,60}\b(uses?|mentions?|references?|describes?|relies on)\b",
    r"\bwhose\b",
)
_CHAIN_RE = re.compile("|".join(_CHAIN_PATTERNS), re.IGNORECASE)


def graph_expansion_active(settings: Settings, analysis: QueryAnalysis) -> bool:
    """Resolve ``retrieval.graph_expansion`` for one query.

    ``True``/``False`` apply to every query; ``"auto"`` (the default) runs the
    expansion only for chain-shaped questions, where it finds the second hop
    and where direct questions would only pay for noise.
    """
    setting = settings.retrieval.graph_expansion
    if setting == "auto":
        return is_chain_question(analysis.raw)
    return bool(setting)


_DEEP_CHAIN_PATTERNS = (
    r"\bem que assenta\b",
    r"\b(built on|relies on|based on|runs on)\b",
    r"\bdescrit[oa]s? na nota\b",
    r"\bdescribed in the note\b",
    r"\b(do|da) (modulo|servico|componente|camada|painel|motor)\b",
    r"\bthe (module|service|component|layer|panel|engine) of\b",
)
_DEEP_CHAIN_RE = re.compile("|".join(_DEEP_CHAIN_PATTERNS), re.IGNORECASE)


def is_deep_chain_question(text: str) -> bool:
    """A chain question that goes through a component note before the subject.

    "the tool that the export module of Project X is built on": the top hit
    is the project note, the component note is the first hop and the tool's
    note the second. Only these questions get the second-hop bridge; on plain
    two-hop questions it would pull in notes about the subject's own facts
    (its maintainer, its licence) as noise.
    """
    folded = normalize_key(text)
    return bool(_DEEP_CHAIN_RE.search(text) or _DEEP_CHAIN_RE.search(folded))


def is_chain_question(text: str) -> bool:
    """Whether a question refers to its subject through another note or entity.

    "Who maintains the plugin that the note 'Tag cloud' uses?" names the bridge
    (the note) rather than the subject (the plugin). Only such questions get
    the second-hop rerank hints: on direct questions the hints promote notes
    about whatever rare entity the top document happens to mention.

    Args:
        text: The question (raw or analyzed text; accents don't matter).

    Returns:
        True when a chain pattern matches.
    """
    folded = normalize_key(text)
    return bool(_CHAIN_RE.search(text) or _CHAIN_RE.search(folded))


_SUBJECT_TYPES: tuple[tuple[re.Pattern[str], frozenset[str]], ...] = (
    (
        re.compile(
            r"\b(plugin|plugins|tool|tools|ferramenta|ferramentas|library|biblioteca|"
            r"extension|extensao|framework|servico|service|sistema|system|software)\b"
        ),
        frozenset({"Technology"}),
    ),
    (
        re.compile(r"\b(quem|who|whom|responsavel|responsible|maintains?|mantem)\b"),
        frozenset({"Person", "Organization"}),
    ),
    (re.compile(r"\b(projeto|projetos|project|projects)\b"), frozenset({"Project"})),
)
_TYPE_BOOST = 1.5
"""Mention-seed multiplier for entities of the type the question asks about."""
_HUB_TYPE_FACTOR = 0.5
"""Mention-seed multiplier for ``Concept`` entities (procedures, generic nouns)."""


_FACT_RE = re.compile(
    r"\b(quem|who|whom|mant[eé]m|maintain(?:s|er|ed)?|licen[cç]a|licen[cs]e|porto|port|"
    r"vers[aã]o|version|linguagem|language|configura\w*|config|ficheiro|file|reposit\w*|"
    r"repository|url|autor|author|owner|dono|responsavel|responsible)\b"
)
_SUBJECT_ENTITY_TYPES = frozenset({"Technology", "Project", "Organization"})
"""Entity types a question can name as the subject of a fact ("who maintains X")."""


def asks_fact_about_subject(text: str) -> bool:
    """Whether the question asks for a fact (maintainer, licence, port, ...) of something.

    Used to let an entity the question *names* act as a bridge: its own note is
    where such facts live, while the notes that merely use it rank first.
    """
    return bool(_FACT_RE.search(normalize_key(text)) or _FACT_RE.search(text))


def min_mention_confidence() -> float:
    """The lowest ``kg_mention.confidence`` expansion uses (``retrieval.graph_lexical_mentions``).

    0.0 when lexical mentions take part (weighted by their confidence), 1.0
    when only extractor mentions count (ablation).
    """
    return 0.0 if get_settings().retrieval.graph_lexical_mentions else 1.0


def _title_keys(hits: Sequence[SearchHit]) -> set[str]:
    """The folded titles and file stems of ``hits``' documents."""
    return {
        key
        for hit in hits
        for key in (normalize_key(hit.title or ""), normalize_key(PurePosixPath(hit.path).stem))
        if key
    }


def mention_confidence_filter() -> list[Any]:
    """The ``kg_mention`` filter matching :func:`min_mention_confidence`."""
    floor = min_mention_confidence()
    return [KgMention.confidence >= floor] if floor > 0 else []


def preferred_entity_types(text: str) -> frozenset[str]:
    """The entity types a question is about, from its subject nouns.

    "the plugin that …" asks for a ``Technology``; "who maintains …" for a
    ``Person``/``Organization``; both patterns may apply (the plugin is the
    bridge, the person the answer). Empty when nothing matches.
    """
    folded = normalize_key(text)
    types: set[str] = set()
    for pattern, matched in _SUBJECT_TYPES:
        if pattern.search(folded):
            types |= matched
    return frozenset(types)


def bridge_entity_types(text: str) -> frozenset[str]:
    """The entity type of the *bridge* a chain question walks through.

    Only the "thing" nouns ("the plugin/tool/library that ...") name the bridge;
    "who"/"quem" names the answer, which lives in the bridge's own note, so an
    organisation hint must not win promotion over the tool (v3 h046/h047).
    """
    folded = normalize_key(text)
    pattern, types = _SUBJECT_TYPES[0]
    return types if pattern.search(folded) else frozenset()


def type_factor(entity_type: str, preferred: frozenset[str]) -> float:
    """Weight multiplier of a mention seed by entity type (see :data:`_TYPE_BOOST`)."""
    if entity_type in preferred:
        return _TYPE_BOOST
    if entity_type == "Concept":
        return _HUB_TYPE_FACTOR
    return 1.0


def mention_seed_weight(doc_rank: int, mention_count: int) -> float:
    """Seed weight of an entity mentioned by the ``doc_rank``-th top document.

    ``0.9 / (1 + 0.5 * doc_rank)`` for the document rank (0 = top document),
    times ``1 / (1 + ln(1 + mention_count))`` for rarity: an entity mentioned
    in two or three notes keeps most of its weight, one mentioned in sixty
    keeps a fifth.

    Args:
        doc_rank: 0 for the top document, 1 for the next distinct one, ...
        mention_count: The entity's ``mention_count_current``.

    Returns:
        The weight in ``(0, 0.9]``.
    """
    rank_weight = _SEED_MENTION_SCORE / (1.0 + _SEED_DOC_DECAY * max(0, doc_rank))
    return rank_weight / (1.0 + math.log1p(max(0, mention_count)))


"""Score of entities mentioned by the leading hybrid hits."""

_SEED_TOP_HITS = 10
"""Hybrid hits whose mention entities join the seeds."""

_STRUCTURAL_TOP_HITS = 5
"""Hybrid hits whose linked documents join the structural search."""

_STRUCTURAL_SIZE = 10
"""Hits fetched by the restricted structural search."""

_STRUCTURAL_DOC_CAP = 100
"""Maximum documents offered to the restricted structural search."""

_FACT_LINES = 15
"""Maximum relation lines in the graph facts block."""

_FACT_ATTRIBUTE_LINES = 3
"""Maximum current attribute lines of the top seed entity."""


@dataclass
class ExpansionState:
    """What one expansion produced, shared with the facts provider.

    Attributes:
        seeds: Seed entity id to seed score.
        scores: Entity id to expansion score.
        sub: The fetched subgraph.
    """

    seeds: dict[int, float]
    scores: dict[int, float]
    sub: Subgraph


def format_relation_line(
    *,
    src: str,
    dst: str,
    relation_type: str,
    attribute: str | None,
    value: str | None,
    status: str,
    valid_from: datetime | None,
) -> str:
    """One graph-facts line (task 43 format).

    ``{src} —{type}→ {dst}`` plus ``[attr = value]``, the ``valid_from`` date
    and ``[historical]`` for non-active relations.
    """
    line = f"{src} —{relation_type}→ {dst}"
    if attribute:
        line += f" [{attribute} = {value}]"
    if valid_from is not None:
        line += f" ({valid_from:%Y-%m-%d})"
    if status != "active":
        line += " [historical]"
    return line


class GraphExpander:
    """The ``graph`` fusion source (PLAN.md §9 step 4).

    Attributes:
        name: The fusion source name used by the client-side RRF.
        last: The most recent expansion's state, read by the facts provider;
            None when the last run was disabled or returned nothing.
    """

    name = "graph"

    def __init__(self) -> None:
        """Args:
        last: Starts empty; set by every :meth:`expand` call.
        """
        self.last: ExpansionState | None = None

    def enabled(self, settings: Settings, analysis: QueryAnalysis) -> bool:
        """Whether graph expansion runs for this query (see :func:`graph_expansion_active`)."""
        return graph_expansion_active(settings, analysis)

    def expand(self, analysis: QueryAnalysis, hybrid_hits: Sequence[SearchHit]) -> list[SearchHit]:
        """Return the graph and structural candidate hits, best first.

        Graph hits are ranked by evidence score; structural hits follow in
        their restricted-search rank order. The analysis filters (tags,
        folders, ACL) are re-applied in Python to the graph hits.

        Args:
            analysis: The query analysis (text, filters, intent, as-of).
            hybrid_hits: The hybrid search hits in rank order.

        Returns:
            The candidate hits; empty when there is nothing to add.
        """
        settings = get_settings()
        seeds, question_seeds, bridge_seeds = self._collect_seeds(analysis, hybrid_hits)
        graph_hits: list[SearchHit] = []
        state: ExpansionState | None = None
        if seeds:
            with session_scope() as session:
                sub = fetch_subgraph(
                    session,
                    list(seeds),
                    settings.retrieval.graph_hops,
                    as_of=analysis.as_of,
                    half_life_days=(
                        settings.retrieval.graph_freshness_half_life_days
                        if settings.retrieval.graph_freshness
                        else None
                    ),
                )
                scores = score_entities(sub, seeds, method=settings.retrieval.graph_method)
                hints: dict[UUID, int] = {}
                top_documents = [hybrid_hits[0].document_id] if hybrid_hits else []
                pairs = evidence_chunks(
                    session,
                    scores,
                    sub,
                    limit=30,
                    seeds={entity_id: seeds[entity_id] for entity_id in question_seeds},
                    current=analysis.intent != "history",
                    hints=hints,
                    min_mention_confidence=min_mention_confidence(),
                    exclude_document_ids=top_documents,
                )
                state = ExpansionState(seeds=dict(seeds), scores=scores, sub=sub)
                hint_names = {}
                hint_types: dict[UUID, str] = {}
                if is_chain_question(analysis.raw):
                    # Only the entities the top document mentions are bridges worth hinting.
                    hint_names = {
                        chunk_id: sub.nodes[entity_id]
                        for chunk_id, entity_id in hints.items()
                        if entity_id in sub.nodes
                        and entity_id in bridge_seeds
                        and entity_id not in question_seeds
                    }
                    hinted_ids = {hints[chunk_id] for chunk_id in hint_names}
                    types = (
                        dict(
                            session.execute(
                                select(KgEntity.id, KgEntity.type).where(
                                    KgEntity.id.in_(list(hinted_ids))
                                )
                            ).all()
                        )
                        if hinted_ids
                        else {}
                    )
                    hint_types = {
                        chunk_id: str(types.get(hints[chunk_id], "")) for chunk_id in hint_names
                    }
            graph_hits = self._load_hits(pairs, analysis, hint_names, hint_types)
            if hint_names and is_deep_chain_question(analysis.raw):
                top_document = hybrid_hits[0].document_id if hybrid_hits else None
                graph_hits += self._second_hop(graph_hits, seeds, analysis, top_document)
        structural_hits = self._structural_hits(analysis, hybrid_hits)
        self.last = state if (graph_hits or structural_hits) else None
        return graph_hits + structural_hits

    def _second_hop(
        self,
        first_hop: list[SearchHit],
        seeds: Mapping[int, float],
        analysis: QueryAnalysis,
        top_document: int | None,
    ) -> list[SearchHit]:
        """Evidence of the rare entities mentioned by the best first-hop bridge document.

        For deep chains the first hinted hit is the component note; its own rare
        mentions (the tool) are the real subject, so their notes are collected
        as further hinted hits. At most :data:`_SECOND_HOP_ENTITIES` entities,
        each weighted like a top-document mention.
        """
        hinted = [hit for hit in first_hop if hit.query_hint and hit.document_id != top_document]
        if not hinted:
            return []
        bridge_doc = hinted[0].document_id
        with session_scope() as session:
            rows = session.execute(
                select(
                    KgMention.entity_id,
                    KgEntity.canonical_name,
                    KgEntity.mention_count_current,
                    KgEntity.type,
                    func.max(KgMention.confidence),
                )
                .join(Chunk, Chunk.id == KgMention.chunk_id)
                .join(KgEntity, KgEntity.id == KgMention.entity_id)
                .where(
                    Chunk.document_id == bridge_doc,
                    KgEntity.status == "active",
                    *mention_confidence_filter(),
                )
                .group_by(
                    KgMention.entity_id,
                    KgEntity.canonical_name,
                    KgEntity.mention_count_current,
                    KgEntity.type,
                )
            ).all()
            preferred = preferred_entity_types(analysis.raw)
            weights = {
                int(entity_id): mention_seed_weight(0, int(count or 0))
                * type_factor(str(entity_type), preferred)
                * float(confidence or 1.0)
                for entity_id, _name, count, entity_type, confidence in rows
                if int(entity_id) not in seeds
            }
            weights = dict(
                sorted(weights.items(), key=lambda item: -item[1])[:_SECOND_HOP_ENTITIES]
            )
            weights = {k: v for k, v in weights.items() if v >= _SECOND_HOP_MIN_WEIGHT}
            if not weights:
                return []
            names = {int(entity_id): str(name) for entity_id, name, _count, _type, _conf in rows}
            hints: dict[UUID, int] = {}
            pairs = evidence_chunks(
                session,
                weights,
                Subgraph(nodes={k: names[k] for k in weights}),
                limit=10,
                seeds=None,
                current=analysis.intent != "history",
                hints=hints,
                min_mention_confidence=min_mention_confidence(),
                exclude_document_ids=[d for d in (top_document, bridge_doc) if d is not None],
            )
            hint_names = {chunk_id: names[entity_id] for chunk_id, entity_id in hints.items()}
            entity_types = {int(entity_id): str(t) for entity_id, _n, _c, t, _conf in rows}
            hint_types = {
                chunk_id: entity_types.get(entity_id, "") for chunk_id, entity_id in hints.items()
            }
        hits = self._load_hits(pairs, analysis, hint_names, hint_types)
        known = {hit.chunk_id for hit in first_hop}
        return [hit for hit in hits if hit.chunk_id not in known]

    def _collect_seeds(
        self, analysis: QueryAnalysis, hybrid_hits: Sequence[SearchHit]
    ) -> tuple[dict[int, float], set[int], set[int]]:
        """Linked entities (their scores) plus mention entities of the top hits.

        Mention seeds are weighted by the rank of the first document they appear
        in (the top document's entities count most) and damped by how common the
        entity is across the corpus (:func:`mention_seed_weight`), so the tool a
        bridge note names outranks "team" or "procedure".

        Returns:
            ``(seeds, question_seed_ids, bridge_seed_ids)``: every seed with its
            weight, the ids linked from the question text itself, and the ids
            mentioned by the top-ranked document (the bridge candidates).
        """
        # The entity that *is* the top note ("Diario de reunioes - Alva": the extractor names
        # the note itself) is not a bridge: its evidence is the note, which crowds out what the
        # note mentions. When the question quotes a title, the same-named notes of other
        # folders ("Diario de reunioes - Xarrama") that the question text links to are noise
        # too. A topic entity that merely shares its name with some note ("DataviewJS") stays.
        hits = list(hybrid_hits)[:_SEED_TOP_HITS]
        quoted = {normalize_key(phrase) for phrase in getattr(analysis, "quoted", ())}
        # Only when the question quotes the top note's title (the title lock fired): an
        # unquoted top note's own entity is often the subject that carries the facts
        # ("Colorcode tasks ... uses DataviewJS", golden set v1 q078), and a quoted entity
        # name that is not a title ("As DataviewJS") stays a seed.
        top_keys = _title_keys(hits[:1])
        title_keys = (top_keys | quoted) if quoted & top_keys else set()
        sibling_keys = _title_keys(hits[1:]) if title_keys else set()
        seeds: dict[int, float] = {}
        linked_types: dict[int, str] = {}
        for linked in link_entities(analysis.text):
            if normalize_key(linked.name) in title_keys | sibling_keys:
                continue
            seeds[linked.entity_id] = max(seeds.get(linked.entity_id, 0.0), linked.score)
            linked_types[linked.entity_id] = linked.type
        question_seeds = set(seeds)
        bridge_seeds: set[int] = set()
        # An entity the question names, when the question asks for one of its facts, is
        # itself the bridge: its own note holds the fact while the notes that use it match
        # the question's words. Treat it as a bridge seed (hint-eligible), not a question seed.
        if asks_fact_about_subject(analysis.raw):
            subjects = {
                entity_id
                for entity_id, entity_type in linked_types.items()
                if entity_type in _SUBJECT_ENTITY_TYPES
            }
            question_seeds -= subjects
            bridge_seeds |= subjects
        preferred = preferred_entity_types(analysis.raw)
        doc_rank: dict[int, int] = {}
        chunk_rank: dict[UUID, int] = {}
        for hit in list(hybrid_hits)[:_SEED_TOP_HITS]:
            if hit.document_id not in doc_rank:
                doc_rank[hit.document_id] = len(doc_rank)
            chunk_rank[UUID(hit.chunk_id)] = doc_rank[hit.document_id]
        if chunk_rank:
            with session_scope() as session:
                rows = session.execute(
                    select(
                        KgMention.chunk_id,
                        KgMention.entity_id,
                        KgEntity.mention_count_current,
                        KgEntity.type,
                        KgMention.confidence,
                        KgEntity.norm_name,
                    )
                    .join(KgEntity, KgEntity.id == KgMention.entity_id)
                    .where(KgMention.chunk_id.in_(list(chunk_rank)), *mention_confidence_filter())
                ).all()
            for chunk_id, entity_id, mention_count, entity_type, confidence, norm_name in rows:
                if str(norm_name) in title_keys:
                    continue
                rank = chunk_rank[UUID(str(chunk_id))]
                weight = (
                    mention_seed_weight(rank, int(mention_count or 0))
                    * type_factor(str(entity_type), preferred)
                    * float(confidence or 1.0)
                )
                seeds[int(entity_id)] = max(seeds.get(int(entity_id), 0.0), weight)
                if rank == 0:
                    bridge_seeds.add(int(entity_id))
        return seeds, question_seeds, bridge_seeds

    def _load_hits(
        self,
        pairs: Sequence[tuple[UUID, float]],
        analysis: QueryAnalysis,
        hint_names: Mapping[UUID, str] | None = None,
        hint_types: Mapping[UUID, str] | None = None,
    ) -> list[SearchHit]:
        """Load the evidence chunks from the index and apply the query filters."""
        if not pairs:
            return []
        from mynotes.index.opensearch import get_client

        settings = get_settings()
        client = get_client()
        response = client.mget(
            index=settings.opensearch.chunks_alias,
            body={
                "docs": [
                    {"_id": str(chunk_id), "_source": {"excludes": ["embedding"]}}
                    for chunk_id, _score in pairs
                ]
            },
        )
        score_by_chunk = {str(chunk_id): score for chunk_id, score in pairs}
        hits: list[SearchHit] = []
        for doc in response.get("docs") or []:
            if not doc.get("found"):
                continue
            hit = _to_hit(doc, rank=1)
            if hit.chunk_id not in score_by_chunk:
                continue
            if not hit_matches_filters(hit, analysis.filters):
                continue
            hit.score = score_by_chunk[hit.chunk_id]
            if hint_names:
                hit.query_hint = hint_names.get(UUID(hit.chunk_id))
                if hint_types:
                    hit.query_hint_type = hint_types.get(UUID(hit.chunk_id)) or None
            hits.append(hit)
        hits.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        for rank, hit in enumerate(hits, start=1):
            hit.rank = rank
        return hits

    def _structural_hits(
        self, analysis: QueryAnalysis, hybrid_hits: Sequence[SearchHit]
    ) -> list[SearchHit]:
        """Search the documents linked from, and sharing a project with, the top hits."""
        top_documents: list[int] = []
        for hit in list(hybrid_hits)[:_STRUCTURAL_TOP_HITS]:
            if hit.document_id not in top_documents:
                top_documents.append(hit.document_id)
        if not top_documents:
            return []
        with session_scope() as session:
            linked = linked_documents(session, top_documents, hops=1)
            siblings = _same_project_document_ids(session, top_documents)
        candidates = set(linked) | set(siblings)
        candidates -= set(top_documents)
        if not candidates:
            return []
        document_ids = sorted(candidates)[:_STRUCTURAL_DOC_CAP]
        filters = replace(analysis.filters, document_ids=document_ids)
        return hybrid_search(analysis.text, filters, size=_STRUCTURAL_SIZE)


class GraphFacts:
    """The ``graph_facts`` extra block (PLAN.md §9 step 6).

    Attributes:
        name: The provider name, rendered as the ``### Graph facts`` heading.
    """

    name = "graph_facts"

    def __init__(self, expander: GraphExpander) -> None:
        """Args:
        expander: The registered expander whose last expansion state feeds
            this block (they are registered together by :func:`register`).
        """
        self._expander = expander

    def block(self, result: RetrievalResult, selected: Sequence[SearchHit]) -> str | None:
        """Render the relation and attribute lines, or None.

        Only when graph expansion is enabled and the last expansion returned
        hits. The relation lines are the top-scoring subgraph edges among the
        entities connected to the selected documents; up to three current
        attribute values of the top seed entity follow. With ACL prefixes on
        the query, the subgraph is first restricted to edges with allowed
        evidence and attribute values without allowed evidence are dropped,
        so facts never leak restricted documents (PLAN.md §11, task 52).

        Args:
            result: The pipeline result (its analysis carries the settings and
                the ACL filters).
            selected: The hits chosen for the context.

        Returns:
            The block text, or None when there is nothing to say.
        """
        if not graph_expansion_active(get_settings(), result.analysis):
            return None
        state = self._expander.last
        if state is None or not state.scores:
            return None
        document_ids: list[int] = []
        for hit in selected:
            if hit.document_id not in document_ids:
                document_ids.append(hit.document_id)
        if not document_ids:
            return None
        acl_prefixes = result.analysis.filters.acl_prefixes
        with session_scope() as session:
            if acl_prefixes is not None:
                state = replace(state, sub=filter_subgraph_by_acl(session, state.sub, acl_prefixes))
            connected = _connected_entities(session, document_ids)
            lines = self._edge_lines(session, state, connected)
            lines.extend(self._attribute_lines(session, state, acl_prefixes))
        return "\n".join(lines) if lines else None

    def _edge_lines(
        self, session: Session, state: ExpansionState, connected: set[int]
    ) -> list[str]:
        """The strongest subgraph edges whose endpoints the documents mention."""
        candidates = [
            (src, dst, relation_id, max(state.scores.get(src, 0.0), state.scores.get(dst, 0.0)))
            for src, dst, relation_id, _weight in state.sub.edges
            if src in connected and dst in connected
        ]
        candidates = [edge for edge in candidates if edge[3] > 0.0]
        candidates.sort(key=lambda edge: (-edge[3], edge[2]))
        if not candidates:
            return []
        relations = {
            int(relation.id): relation
            for relation in session.execute(
                select(KgRelation).where(
                    KgRelation.id.in_([edge[2] for edge in candidates[:_FACT_LINES]])
                )
            ).scalars()
        }
        lines: list[str] = []
        for src, dst, relation_id, _score in candidates:
            relation = relations.get(relation_id)
            if relation is None:
                continue
            src_name = state.sub.nodes.get(src, f"entity {src}")
            dst_name = state.sub.nodes.get(dst, f"entity {dst}")
            lines.append(
                format_relation_line(
                    src=src_name,
                    dst=dst_name,
                    relation_type=relation.type,
                    attribute=relation.attribute,
                    value=relation.value,
                    status=relation.status,
                    valid_from=relation.valid_from,
                )
            )
            if len(lines) >= _FACT_LINES:
                break
        return lines

    def _attribute_lines(
        self, session: Session, state: ExpansionState, acl_prefixes: Sequence[str] | None = None
    ) -> list[str]:
        """The current attribute values of the highest-scoring seed entity.

        With ``acl_prefixes``, values whose backing relation has no allowed
        evidence chunk are dropped (task 52).
        """
        from mynotes.graph.temporal import current_attribute_values

        seed_id = max(state.seeds.items(), key=lambda item: (item[1], -item[0]))[0]
        values = current_attribute_values(session, seed_id)
        if acl_prefixes is not None:
            values = {
                attribute: value
                for attribute, value in values.items()
                if relation_has_allowed_evidence(session, value.relation_id, acl_prefixes)
            }
        name = state.sub.nodes.get(seed_id, f"entity {seed_id}")
        lines = [
            f"{name} [{value.attribute} = {value.value}]"
            for value in list(values.values())[:_FACT_ATTRIBUTE_LINES]
        ]
        return lines


def _connected_entities(session: Session, document_ids: Sequence[int]) -> set[int]:
    """Entities mentioned in, or evidenced by relations of, the documents."""
    connected: set[int] = set()
    for (entity_id,) in session.execute(
        select(KgMention.entity_id)
        .join(Chunk, Chunk.id == KgMention.chunk_id)
        .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
        .where(
            DocumentVersion.document_id.in_(list(document_ids)),
            # Extractor mentions only: the block lists what the selected notes state, and
            # lexical mentions would let their edges crowd out the stated ones.
            KgMention.confidence >= 1.0,
        )
        .distinct()
    ).all():
        connected.add(int(entity_id))
    for src, dst in session.execute(
        select(KgRelation.src_entity_id, KgRelation.dst_entity_id)
        .join(KgRelationEvidence, KgRelationEvidence.relation_id == KgRelation.id)
        .join(Chunk, Chunk.id == KgRelationEvidence.chunk_id)
        .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
        .where(DocumentVersion.document_id.in_(list(document_ids)))
        .distinct()
    ).all():
        connected.add(int(src))
        connected.add(int(dst))
    return connected


def _same_project_document_ids(session: Session, document_ids: Sequence[int]) -> set[int]:
    """Other documents sharing an accepted ``project:*`` tag with any input."""
    rows = session.execute(
        select(DocumentTag.document_id)
        .join(Tag, Tag.id == DocumentTag.tag_id)
        .where(
            Tag.name.like("project:%"),
            DocumentTag.status == "accepted",
            DocumentTag.document_id.notin_(list(document_ids)),
            DocumentTag.tag_id.in_(
                select(DocumentTag.tag_id).where(
                    DocumentTag.document_id.in_(list(document_ids)),
                    DocumentTag.status == "accepted",
                )
            ),
        )
        .distinct()
    ).scalars()
    return {int(document_id) for document_id in rows}


def register() -> None:
    """(Re-)register the expander and the facts provider with the registries.

    Called at import and by tests that unregistered them for isolation;
    registering replaces any entry with the same name and keeps the provider
    bound to the registered expander's state.
    """
    expander = GraphExpander()
    register_expander(expander)
    register_extra_block(GraphFacts(expander))


register()
