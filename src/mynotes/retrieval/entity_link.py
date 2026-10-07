"""Query-time entity linking (PLAN.md §9 step 1, task 43).

:func:`link_entities` finds the KG entities a question talks about, combining
two signals: exact n-gram matches of the folded question against
``kg_entity.norm_name``/``kg_entity_alias.alias_norm`` (score 1.0), and a
hybrid search on the entities index through the same RRF pipeline as the
chunks (score ``1/rank``), kept only when a name is actually similar to some
question n-gram (trigram similarity >= 0.5) so vaguely related entities never
seed the expansion. The OpenSearch leg degrades to exact-only when the index
or the embedding endpoint is unavailable, so linking keeps working offline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import ARRAY, Text, any_, bindparam, or_, select
from sqlalchemy.orm import Session

from mynotes.config import get_settings
from mynotes.db.models import KgEntity, KgEntityAlias
from mynotes.db.session import session_scope
from mynotes.index.mappings import PIPELINE_NAME
from mynotes.text.normalize import normalize_key

logger = logging.getLogger(__name__)

MAX_NGRAM_WORDS = 4
"""Longest exact-match n-gram (words)."""

SEARCH_SIZE = 10
"""Candidates fetched from the entities index before the similarity gate."""

_MIN_TRIGRAM = 0.5
"""Minimum trigram similarity between a name and some query n-gram."""


@dataclass(frozen=True)
class LinkedEntity:
    """One entity linked from the query text.

    Attributes:
        entity_id: The KG entity id.
        name: The canonical name.
        type: The entity type.
        score: 1.0 for exact matches, ``1/rank`` for search matches.
        via: ``"exact"`` (Postgres n-gram match) or ``"search"`` (index).
    """

    entity_id: int
    name: str
    type: str
    score: float
    via: Literal["exact", "search"]


_TRIM_EDGES = re.compile(r"^\W+|\W+$", re.UNICODE)
"""Leading/trailing punctuation trimmed from query tokens (``alfa?`` → ``alfa``)."""


def word_ngrams(text_key: str, *, max_words: int = MAX_NGRAM_WORDS) -> list[str]:
    """The 1..``max_words``-word n-grams of an already-folded key, in order.

    Tokens lose their leading/trailing punctuation (sentence marks, quotes) so
    ``Projeto Alfa?`` still links to the ``projeto alfa`` entity; punctuation
    inside a token (``c++``, ``node.js``) is kept.

    Args:
        text_key: The folded text (``normalize_key`` output).
        max_words: Longest n-gram.

    Returns:
        The unique n-grams, shortest first, first occurrence order kept.
    """
    tokens = [token for token in (_TRIM_EDGES.sub("", raw) for raw in text_key.split()) if token]
    ngrams: list[str] = []
    seen: set[str] = set()
    for size in range(1, min(max_words, len(tokens)) + 1):
        for start in range(len(tokens) - size + 1):
            gram = " ".join(tokens[start : start + size])
            if gram not in seen:
                seen.add(gram)
                ngrams.append(gram)
    return ngrams


def trigram_similarity(a: str, b: str) -> float:
    """pg_trgm-style trigram similarity of two strings (set Jaccard).

    Both strings are padded with two leading and one trailing space (like
    ``pg_trgm``) and split into character trigrams; the score is the size of
    the shared trigram set over the union.

    Args:
        a: First string.
        b: Second string.

    Returns:
        The similarity in ``[0.0, 1.0]``; 1.0 when both are empty, 0.0 when
        only one is.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    def grams(text: str) -> set[str]:
        padded = f"  {text} "
        return {padded[index : index + 3] for index in range(len(padded) - 2)}

    left, right = grams(a), grams(b)
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _exact_matches(session: Session, ngrams: list[str]) -> list[LinkedEntity]:
    """Active entities whose normalized name or an alias equals some n-gram.

    One query with ``= ANY(:ngrams)`` over both columns; aliases join so a
    match through either wins.
    """
    grams_param = bindparam("ngrams", value=ngrams, type_=ARRAY(Text))
    rows = session.execute(
        select(KgEntity.id, KgEntity.canonical_name, KgEntity.type)
        .join(KgEntityAlias, KgEntityAlias.entity_id == KgEntity.id, isouter=True)
        .where(
            KgEntity.status == "active",
            or_(
                KgEntity.norm_name == any_(grams_param),
                KgEntityAlias.alias_norm == any_(grams_param),
            ),
        )
        .distinct()
        .order_by(KgEntity.id)
    ).all()
    return [
        LinkedEntity(
            entity_id=int(row.id), name=row.canonical_name, type=row.type, score=1.0, via="exact"
        )
        for row in rows
    ]


def build_entity_hybrid_body(query: str, query_vector: list[float], size: int) -> dict[str, Any]:
    """Build the native hybrid body for the entities index (task 43).

    The k-NN leg comes first (the RRF pipeline weights are ``[knn, bm25]``);
    the BM25 leg is a ``most_fields`` multi_match over the name fields and the
    description. Both legs keep active entities only.

    Args:
        query: The raw query text.
        query_vector: The query embedding.
        size: Candidates per leg and overall.

    Returns:
        The ``client.search`` body.
    """
    status_filter = [{"term": {"status": "active"}}]
    knn_leg = {
        "knn": {
            "embedding": {
                "vector": list(query_vector),
                "k": size,
                "filter": {"bool": {"filter": status_filter}},
            }
        }
    }
    bm25_leg = {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": query,
                        "type": "most_fields",
                        "fields": [
                            "canonical_name^2",
                            "canonical_name.pt",
                            "aliases^2",
                            "aliases.pt",
                            "description",
                        ],
                    }
                }
            ],
            "filter": status_filter,
        }
    }
    return {
        "size": size,
        "_source": {"excludes": ["embedding"]},
        "query": {"hybrid": {"queries": [knn_leg, bm25_leg]}},
    }


def _name_keys(source: dict[str, Any]) -> list[str]:
    """The folded names of an indexed entity: canonical name plus aliases."""
    keys = [normalize_key(str(source.get("canonical_name") or ""))]
    keys.extend(normalize_key(str(alias)) for alias in source.get("aliases") or [])
    return [key for key in keys if key]


def _search_matches(query_text: str, ngrams: list[str]) -> list[LinkedEntity]:
    """Entities found by the hybrid entities search, gated by trigram similarity.

    A hit survives only when one of its names reaches
    :data:`_MIN_TRIGRAM` similarity against some query n-gram; its score is
    ``1/rank`` in the returned order.
    """
    from mynotes.embed.client import get_embedder
    from mynotes.index.opensearch import get_client

    settings = get_settings()
    vector = get_embedder().embed_query(query_text, task="retrieval")
    body = build_entity_hybrid_body(query_text, vector, SEARCH_SIZE)
    response = get_client().search(
        index=settings.opensearch.entities_alias,
        body=body,
        params={"search_pipeline": PIPELINE_NAME},
    )
    linked: list[LinkedEntity] = []
    for rank, raw_hit in enumerate(response.get("hits", {}).get("hits", []), start=1):
        source = raw_hit.get("_source") or {}
        keys = _name_keys(source)
        similar = any(
            trigram_similarity(key, ngram) >= _MIN_TRIGRAM for key in keys for ngram in ngrams
        )
        if not similar:
            continue
        linked.append(
            LinkedEntity(
                entity_id=int(source["entity_id"]),
                name=str(source.get("canonical_name") or ""),
                type=str(source.get("type") or ""),
                score=1.0 / rank,
                via="search",
            )
        )
    return linked


def link_entities(query_text: str, *, limit: int = 5) -> list[LinkedEntity]:
    """Link the query text to KG entities (exact n-grams, then index search).

    Args:
        query_text: The question text (filters already stripped).
        limit: Maximum number of linked entities.

    Returns:
        At most ``limit`` entities, best first (score descending, then id).
        Exact matches win over search matches of the same entity.
    """
    ngrams = word_ngrams(normalize_key(query_text))
    if not ngrams:
        return []
    exact: list[LinkedEntity] = []
    with session_scope() as session:
        exact = _exact_matches(session, ngrams)
    by_id: dict[int, LinkedEntity] = {linked.entity_id: linked for linked in exact}
    try:
        search = _search_matches(query_text, ngrams)
    except Exception:
        logger.warning("entity search failed; linking from exact matches only", exc_info=True)
        search = []
    for linked in search:
        by_id.setdefault(linked.entity_id, linked)
    ordered = sorted(by_id.values(), key=lambda linked: (-linked.score, linked.entity_id))
    return ordered[:limit]
