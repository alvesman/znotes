"""OpenSearch client factory and admin helpers for index lifecycle management.

Covers what tasks 11 (``mynotes init``/``status``), 22 (outbox sync) and 23
(rebuild) need: a client built from ``settings.opensearch``, versioned physical
indices behind aliases, atomic alias swaps, the RRF search pipeline, and a
live-mapping inspection that reports exactly what to run to fix drift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from opensearchpy import NotFoundError, OpenSearch

from mynotes.config import Settings, get_settings
from mynotes.index.mappings import (
    PIPELINE_NAME,
    chunks_index_body,
    entities_index_body,
    rrf_pipeline_body,
)

logger = logging.getLogger(__name__)

_CLIENT_TIMEOUT_S = 30


def get_client() -> OpenSearch:
    """Build an OpenSearch client from ``settings.opensearch``.

    Returns:
        A client using the configured URL, basic auth when a user is set,
        ``verify_certs`` from settings, warnings silenced and a 30 s timeout.
    """
    cfg = get_settings().opensearch
    http_auth: tuple[str, str] | None = None
    if cfg.user is not None:
        password = cfg.password.get_secret_value() if cfg.password is not None else ""
        http_auth = (cfg.user, password)
    return OpenSearch(
        hosts=[cfg.url],
        http_auth=http_auth,
        use_ssl=cfg.url.startswith("https"),
        verify_certs=cfg.verify_certs,
        ssl_show_warn=False,
        timeout=_CLIENT_TIMEOUT_S,
    )


def physical_name(alias: str, version: int) -> str:
    """Return the physical index name for ``alias`` at ``version``.

    Args:
        alias: The alias name (e.g. ``mynotes_chunks``).
        version: The physical index version.

    Returns:
        ``f"{alias}_v{version}"`` (e.g. ``mynotes_chunks_v1``).
    """
    return f"{alias}_v{version}"


def current_physical_index(client: OpenSearch, alias: str) -> str | None:
    """Return the physical index ``alias`` points to, or None when missing.

    Args:
        client: The OpenSearch client.
        alias: The alias to resolve.

    Returns:
        The physical index name; when several share the alias (should not
        happen), the lexicographically first is returned.
    """
    try:
        result = client.indices.get_alias(name=alias)
    except NotFoundError:
        return None
    return sorted(result)[0]


def create_index(client: OpenSearch, name: str, body: dict[str, Any]) -> None:
    """Create the index with settings and mappings; no-op when it already exists.

    Args:
        client: The OpenSearch client.
        name: The physical index name.
        body: ``{"settings": ..., "mappings": ...}`` from :mod:`mappings`.
    """
    if client.indices.exists(index=name):
        logger.info("OpenSearch index '%s' already exists", name)
        return
    client.indices.create(index=name, body=body)
    logger.info("Created OpenSearch index '%s'", name)


def swap_alias(client: OpenSearch, alias: str, new_index: str) -> None:
    """Point ``alias`` at ``new_index`` atomically.

    One ``_aliases`` update removes the alias from the current physical index
    (when any) and adds it to ``new_index``, so readers never see a missing or
    half-swapped alias.

    Args:
        client: The OpenSearch client.
        alias: The alias to move.
        new_index: The physical index the alias should point to.
    """
    actions: list[dict[str, Any]] = []
    old = current_physical_index(client, alias)
    if old is not None and old != new_index:
        actions.append({"remove": {"index": old, "alias": alias}})
    actions.append({"add": {"index": new_index, "alias": alias}})
    client.indices.update_aliases(body={"actions": actions})
    logger.info("Alias '%s' now points to '%s'", alias, new_index)


def ensure_indices(client: OpenSearch, settings: Settings) -> dict[str, str]:
    """Create the chunk and entity indices behind their aliases when missing.

    For each configured alias: when it does not exist yet, ``<alias>_v1`` is
    created with the mapping from :mod:`mappings` (dimension and folding from
    settings) and the alias is pointed at it. Existing indices are left alone,
    so the function is idempotent and never re-embeds or re-creates.

    Args:
        client: The OpenSearch client.
        settings: Application settings (aliases, embedding dim, folding).

    Returns:
        ``{alias: physical_index}`` after the call.
    """
    dim = settings.models.embedding.dim
    folding = settings.text_matching.folding
    result: dict[str, str] = {}
    for alias, body in (
        (settings.opensearch.chunks_alias, chunks_index_body(dim, folding)),
        (settings.opensearch.entities_alias, entities_index_body(dim, folding)),
    ):
        existing = current_physical_index(client, alias)
        if existing is not None:
            logger.info("Alias '%s' already points to '%s'", alias, existing)
            result[alias] = existing
            continue
        physical = physical_name(alias, 1)
        create_index(client, physical, body)
        swap_alias(client, alias, physical)
        result[alias] = physical
    return result


def ensure_pipeline(client: OpenSearch, settings: Settings) -> None:
    """Upsert the RRF search pipeline from ``retrieval.fusion`` settings.

    Args:
        client: The OpenSearch client.
        settings: Application settings (``rank_constant``, ``hybrid_weights``).
    """
    fusion = settings.retrieval.fusion
    body = rrf_pipeline_body(fusion.rank_constant, fusion.hybrid_weights)
    client.transport.perform_request("PUT", f"/_search/pipeline/{PIPELINE_NAME}", body=body)
    logger.info(
        "Search pipeline '%s' ready (rank_constant=%s, weights=%s)",
        PIPELINE_NAME,
        fusion.rank_constant,
        fusion.hybrid_weights,
    )


@dataclass
class LiveMappingReport:
    """Live state of one alias, with the problems found and how to fix them."""

    alias: str
    physical: str | None
    engine: str | None
    dim: int | None
    has_folded_subfields: bool
    problems: list[str] = field(default_factory=list)


def _mapping_properties(client: OpenSearch, physical: str) -> dict[str, Any]:
    """Return the ``properties`` dict of a physical index (empty on surprises)."""
    raw = client.indices.get_mapping(index=physical)
    if not isinstance(raw, dict) or physical not in raw:
        logger.warning("Unexpected mapping payload for '%s'", physical)
        return {}
    return raw[physical].get("mappings", {}).get("properties", {})


def _pipeline_exists(client: OpenSearch) -> bool:
    """Whether the RRF search pipeline exists on the cluster."""
    try:
        client.transport.perform_request("GET", f"/_search/pipeline/{PIPELINE_NAME}")
    except NotFoundError:
        return False
    return True


def _inspect_alias(
    client: OpenSearch, alias: str, folded_field: str, expected_dim: int
) -> LiveMappingReport:
    """Check one alias against the expected mapping, collecting problems."""
    physical = current_physical_index(client, alias)
    report = LiveMappingReport(
        alias=alias, physical=physical, engine=None, dim=None, has_folded_subfields=False
    )
    if physical is None:
        report.problems.append(f"alias '{alias}' is missing; run `mynotes index rebuild`")
        return report
    props = _mapping_properties(client, physical)
    embedding = props.get("embedding", {})
    report.engine = embedding.get("method", {}).get("engine")
    report.dim = embedding.get("dimension")
    if report.engine != "lucene":
        report.problems.append(
            f"'{physical}' k-NN engine is {report.engine!r}, expected 'lucene'; "
            "run `mynotes index rebuild`"
        )
    if report.dim != expected_dim:
        report.problems.append(
            f"'{physical}' embedding dimension is {report.dim}, expected {expected_dim}; "
            "run `mynotes index rebuild`"
        )
    subfields = props.get(folded_field, {}).get("fields", {})
    report.has_folded_subfields = "pt" in subfields and "en" in subfields
    if not report.has_folded_subfields:
        report.problems.append(
            f"'{physical}' field '{folded_field}' is missing the pt/en folded subfields; "
            "run `mynotes index rebuild`"
        )
    return report


def inspect_live(client: OpenSearch, settings: Settings) -> list[LiveMappingReport]:
    """Inspect the live chunk and entity indices against the configuration.

    Reports a problem when an alias is missing, the k-NN engine is not
    ``lucene``, the stored dimension differs from ``models.embedding.dim``,
    the folded pt/en subfields are absent, or the search pipeline is missing
    (checked once, reported on the chunks entry).

    Args:
        client: The OpenSearch client.
        settings: Application settings (expected dim, aliases).

    Returns:
        One :class:`LiveMappingReport` per alias (chunks first, entities second).
    """
    expected_dim = settings.models.embedding.dim
    pipeline_missing = not _pipeline_exists(client)
    reports: list[LiveMappingReport] = []
    for alias, folded_field in (
        (settings.opensearch.chunks_alias, "text"),
        (settings.opensearch.entities_alias, "description"),
    ):
        report = _inspect_alias(client, alias, folded_field, expected_dim)
        if pipeline_missing:
            report.problems.append(
                f"search pipeline '{PIPELINE_NAME}' is missing; run `mynotes init`"
            )
        reports.append(report)
    return reports
