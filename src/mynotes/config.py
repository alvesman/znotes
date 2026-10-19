"""Typed application settings loaded from ``mynotes.yaml`` and ``MYNOTES_*`` variables.

The single entry point is :func:`get_settings`. Sources, highest priority
first: explicit keyword arguments to :class:`Settings`, flat ``MYNOTES_*``
variables (process environment, then ``.env`` in the current directory), the
YAML file (``MYNOTES_CONFIG``, else ``./mynotes.yaml``, else the per-OS user
config directory), and model defaults. Secrets are held in ``SecretStr`` and
never appear in ``repr()``/``str()``; use :meth:`Settings.redacted_dict` for
display.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import platformdirs
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)


def mask_secret(value: str | None) -> str:
    """Mask a secret for logs and status output.

    Args:
        value: The secret value, or None when not configured.

    Returns:
        ``"abc…yz (N chars)"``, or ``"(none)"`` when the value is empty.
    """
    if not value:
        return "(none)"
    return f"{value[:3]}…{value[-2:]} ({len(value)} chars)"


def _redacted(value: Any) -> Any:
    """Recursively replace ``SecretStr`` values by their masked representation."""
    if isinstance(value, SecretStr):
        return mask_secret(value.get_secret_value())
    if isinstance(value, dict):
        return {key: _redacted(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redacted(item) for item in value]
    return value


def _read_dotenv() -> dict[str, str | None]:
    """Parse ``.env`` in the current directory, or return an empty dict."""
    return dict(dotenv_values(".env"))


def _set_nested(data: dict[str, Any], keys: list[str], value: Any) -> None:
    """Set ``value`` at the nested ``keys`` path inside ``data``, creating dicts."""
    node = data
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def _resolve_yaml_path() -> Path | None:
    """Locate the YAML file to load, following the documented search order.

    Returns:
        The path from ``MYNOTES_CONFIG`` (process env, then ``.env``), else
        ``./mynotes.yaml`` if it exists, else ``<user config dir>/mynotes.yaml``
        if it exists, else None.
    """
    configured = os.environ.get("MYNOTES_CONFIG") or _read_dotenv().get("MYNOTES_CONFIG")
    if configured:
        return Path(configured).expanduser()
    local = Path("mynotes.yaml")
    if local.is_file():
        return local
    user_config = Path(platformdirs.user_config_dir("mynotes")) / "mynotes.yaml"
    if user_config.is_file():
        return user_config
    return None


class _StrictModel(BaseModel):
    """Base for configuration sections; unknown keys are validation errors."""

    model_config = ConfigDict(extra="forbid")


class RootConfig(_StrictModel):
    """A source root with optional per-root overrides for watching and hashing."""

    name: str
    path: Path
    watch: bool | None = None
    watch_backend: Literal["auto", "native", "polling"] | None = None
    case_sensitive: Literal["auto"] | bool = "auto"
    hash_always: bool | None = None

    @field_validator("path")
    @classmethod
    def _expand_user(cls, value: Path) -> Path:
        """Expand a leading ``~`` so the path is usable as-is."""
        return value.expanduser()


class ScheduleConfig(_StrictModel):
    """Scheduled scan triggers (interval, cron or manual)."""

    mode: Literal["interval", "cron", "manual"] = "interval"
    interval_minutes: int = 15
    cron: str | None = None
    full_scan_cron: str | None = "0 3 * * *"
    run_on_start: bool = True


class WatchConfig(_StrictModel):
    """File-watcher defaults, overridable per root."""

    enabled: bool = False
    backend: Literal["auto", "native", "polling"] = "auto"
    debounce_ms: int = 1600
    poll_delay_ms: int = 2000
    max_batch_paths: int = 500


class ScanConfig(_StrictModel):
    """Scan behaviour: schedule, settle/grace times, ignore rules, watcher."""

    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    settle_seconds: int = 10
    delete_grace_seconds: int = 60
    mtime_tolerance_ms: int = 2000
    hash_always: bool = False
    watch: WatchConfig = Field(default_factory=WatchConfig)
    ignore: list[str] = Field(
        default_factory=lambda: [
            ".git/",
            "node_modules/",
            "~$*",
            ".~lock*",
            "*.tmp",
            "*.crdownload",
            "*.partial",
            ".DS_Store",
            "._*",
            "Thumbs.db",
            "desktop.ini",
        ]
    )
    skip_cloud_placeholders: bool = True


class ExtractConfig(_StrictModel):
    """Extraction options for PDF and Office documents."""

    pdf_ocr: bool = False
    office_backend: Literal["markitdown", "docling"] = "markitdown"


class ChunkingConfig(_StrictModel):
    """Token targets for the chunker."""

    target_tokens: int = 350
    max_tokens: int = 512
    overlap_tokens: int = 80


class TextMatchingConfig(_StrictModel):
    """Analyzer options shared by the OpenSearch index."""

    folding: Literal["asciifolding", "icu"] = "asciifolding"
    stem_languages: list[str] = Field(default_factory=lambda: ["pt", "en"])


class FolderRule(_StrictModel):
    """Tags assigned to files under a folder matching ``pattern``."""

    pattern: str
    tags: list[str]


class TagsConfig(_StrictModel):
    """Tag derivation: folder rules, folder segments and LLM auto-tagging."""

    folder_rules: list[FolderRule] = Field(default_factory=list)
    folder_segments_as_tags: bool = True
    ignore_segments: list[str] = Field(default_factory=lambda: ["misc", "tmp"])
    llm_autotag: bool = True
    llm_auto_accept: float = 0.85


class VersionsConfig(_StrictModel):
    """Document version detection, retention and thresholds."""

    max_per_document: int | None = 5
    max_age_days: int | None = None
    min_interval_hours: float = 24
    index_old_versions: bool = Field(
        default=False,
        description=(
            "Index retained non-latest snapshots (is_latest=false) so history "
            "intent and --as-of can search them. Changing this requires "
            "'mynotes index rebuild'; the regular embed/sync flow only keeps "
            "newly written chunks consistent."
        ),
    )
    auto_threshold: float = 0.80
    propose_threshold: float = 0.55
    archive_folders: list[str] = Field(
        default_factory=lambda: ["archive", "old", "versions", "arquivo"]
    )


class RetentionConfig(_StrictModel):
    """Housekeeping thresholds for tombstones, caches, jobs and the outbox."""

    deleted_days: int = 30
    jobs_done_days: int = 14
    outbox_processed_days: int = 7
    cache_orphan_days: int = 30


class EndpointLimits(_StrictModel):
    """Concurrency, rate and timeout limits for one model endpoint."""

    concurrency: int = 8
    max_requests_per_minute: int | None = None
    timeout_s: float = 60


class EmbeddingConfig(_StrictModel):
    """Embedding endpoint, model, query format and limits."""

    # No defaults: every installation names its own endpoint, model and dimension
    # (examples for OpenAI, Ollama and vLLM/TEI in mynotes.example.yaml).
    base_url: str | None = None
    api_key: SecretStr | None = None
    model: str | None = None
    dim: int | None = Field(default=None, gt=0)
    # How a search query is sent to the embedding model. "{text}" (the query as typed) suits
    # most models; instruction-tuned ones want their own format, e.g. Nemotron/E5-Mistral
    # "Instruct: {task}\nQuery: {text}", nomic-embed-text "search_query: {text}".
    query_template: str = "{text}"
    passage_prefix: str = ""
    tasks: dict[str, str] = Field(
        default_factory=lambda: {
            "retrieval": "Given a question, retrieve passages that answer the question"
        }
    )
    normalize: Literal["auto", "always", "never"] = "auto"
    batch: int = 8
    limits: EndpointLimits = Field(default_factory=EndpointLimits)


class RerankerConfig(_StrictModel):
    """Reranker endpoint, model and limits."""

    provider: str = "cohere"
    # Optional: without a url, results keep their fused order (no reranking).
    url: str | None = None
    api_key: SecretStr | None = None
    model: str | None = None
    top_n: int = 0
    max_documents_per_request: int = 40
    limits: EndpointLimits = Field(default_factory=lambda: EndpointLimits(timeout_s=30))


class LLMLimits(_StrictModel):
    """Concurrency windows and timeout for the LLM endpoint."""

    interactive_concurrency: int = 2
    background_concurrency: int = 8
    max_total_concurrency: int | None = None
    max_requests_per_minute: int | None = None
    background_window: str | None = None
    timeout_s: float = 120


class LLMConfig(_StrictModel):
    """Chat-completion endpoint, model parameters and limits."""

    base_url: str | None = None
    api_key: SecretStr | None = None
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 32000
    context_window: int = 64000
    # True for models with a thinking mode switched by chat_template_kwargs (Qwen3 on vLLM):
    # the client then asks for thinking off. Leave False for OpenAI and most servers.
    thinking_model: bool = False
    limits: LLMLimits = Field(default_factory=LLMLimits)


class RetryConfig(_StrictModel):
    """Retry and backoff behaviour shared by all model endpoints."""

    max_attempts: int = 4
    backoff_base_s: float = 2.0
    backoff_max_s: float = 60.0
    honor_retry_after: bool = True


class ModelsConfig(_StrictModel):
    """Model endpoints: embedding, reranker, LLM and shared retry policy."""

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    extract_llm: LLMConfig | None = None
    retry: RetryConfig = Field(default_factory=RetryConfig)


class FusionConfig(_StrictModel):
    """Hybrid fusion parameters shared with the OpenSearch RRF pipeline."""

    technique: Literal["rrf"] = "rrf"
    rank_constant: int = 20
    hybrid_weights: list[float] = Field(default_factory=lambda: [0.7, 0.3])
    source_weights: dict[str, float] = Field(
        default_factory=lambda: {"hybrid": 1.0, "graph": 0.6, "version": 0.8}
    )

    @model_validator(mode="after")
    def _hybrid_weights_sum_to_one(self) -> FusionConfig:
        """Require ``hybrid_weights`` to sum to 1.0 (tolerance 1e-6)."""
        total = sum(self.hybrid_weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"hybrid_weights must sum to 1.0 (+/-1e-6); got {total:.6f}")
        return self


class GraphConfig(_StrictModel):
    """Knowledge-graph construction options (the extractor itself is LLM-driven)."""

    # Lexical mention linking (docs/eval/multihop.md): after a section group is resolved, names
    # and aliases of entities the workspace already knows that occur verbatim (whole word,
    # accent/case-folded) in its chunks get mentions the extractor did not produce. Guards:
    # named-thing types only, a minimum name length, a per-chunk cap, and three rarity bounds: an
    # entity with more extractor mentions than lexical_max_mentions is a hub; a name found in more
    # than lexical_max_chunk_share of the chunks (at least 20), or in more than lexical_max_ratio
    # times the chunks the extractor found it in, is a common word rather than a missed mention.
    lexical_mentions: bool = True
    lexical_min_length: int = Field(default=4, ge=2)
    lexical_max_mentions: int = Field(default=50, ge=0)
    lexical_max_chunk_share: float = Field(default=0.02, gt=0, le=1.0)
    lexical_max_ratio: float = Field(default=8.0, gt=0)
    lexical_per_chunk: int = Field(default=8, ge=1)
    lexical_types: list[str] = Field(
        default_factory=lambda: ["Technology", "Project", "Person", "Organization"]
    )


class RetrievalConfig(_StrictModel):
    """Retrieval pipeline tuning: candidate counts, reranking and context size."""

    fusion: FusionConfig = Field(default_factory=FusionConfig)
    # 0.0 disables the rerank-score "not found" gate: on golden set v1 the reranker scored many
    # correct PT/multi-hop/history hits very low, and the answer prompt already makes the LLM
    # decline when the notes lack the answer (docs/eval/v1-reviewed-results.md).
    min_rerank_score: float = 0.0
    candidates: int = 50
    # Neighbours the approximate k-NN search considers before the top `candidates` are kept
    # (None = candidates). HNSW with k = 50 missed a note the exact search ranks 14th; 200 fixed
    # it with no regression or added latency on golden set v1 (dev and hold-out, 2026-10-19).
    knn_k: int | None = Field(default=200, gt=0)
    rerank_top: int = 40
    max_docs: int = 4
    # Score-gap rule (docs/improvement-plan.md step 3): after the first max_docs documents, up
    # to extra_docs_max more are admitted while their best rerank score is at least
    # extra_docs_min_ratio x the top score (1.0 = off; the depth sweep showed a flat max_docs 6
    # dilutes other questions).
    extra_docs_min_ratio: float = Field(default=1.0, gt=0, le=1.0)
    extra_docs_max: int = Field(default=2, ge=0)
    max_chunks: int = 10
    # "auto" (default): graph expansion only for chain-shaped questions ("the plugin that note
    # X uses"), where it helps (docs/eval/multihop.md); True = always; False = never.
    graph_expansion: bool | Literal["auto"] = "auto"
    graph_hops: int = 1
    graph_method: Literal["one_hop", "ppr"] = "one_hop"
    # Freshness (docs/eval/freshness.md): decay graph-expansion edge weights by the age of their
    # latest evidence (off by default), and mildly favour recent documents in the final ranking
    # (on by default: no regression on golden set v1; history questions are never boosted).
    graph_freshness: bool = False
    graph_freshness_half_life_days: float = Field(default=365.0, gt=0)
    # Lexical mentions (graph.lexical_mentions) at query time: seeds and evidence weighted by
    # the mention's confidence (0.6 for lexical, 1.0 for extractor mentions); False ignores them
    # in expansion (ablation).
    graph_lexical_mentions: bool = True
    # A quoted phrase that equals a note's title (or file name) moves that note's chunks to the
    # front of the candidates, loading them when retrieval missed the note entirely.
    title_lock: bool = True
    # A full date written in a current-intent question ("na reunião de 2026-01-12") puts the
    # notes named or dated that day first (weekly notes otherwise lose to the newest week).
    date_lock: bool = True
    # "When did X change to Y" questions: no recency boost, the other notes of the top hit's
    # folder join the pool, and the near-tied best notes are ordered earliest first with the
    # note just before the change second (docs/eval/series.md).
    change_questions: bool = True
    # Chat follow-ups: the last N turns of the conversation are used to rewrite a follow-up
    # into a standalone question for retrieval, and are given to the answer (0 = off).
    followup_turns: int = Field(default=3, ge=0, le=10)
    recency_boost: bool = True
    recency_half_life_days: float = Field(default=365.0, gt=0)
    recency_min_multiplier: float = Field(default=0.8, gt=0, le=1.0)
    context_tokens: int = 3500


class PostgresConfig(_StrictModel):
    """PostgreSQL connection (secret DSN from ``.env``)."""

    dsn: SecretStr | None = None


class OpenSearchConfig(_StrictModel):
    """OpenSearch connection and index aliases."""

    url: str = "https://localhost:9200"
    user: str | None = None
    password: SecretStr | None = None
    verify_certs: bool = False
    chunks_alias: str = "mynotes_chunks"
    entities_alias: str = "mynotes_entities"


class OIDCConfig(_StrictModel):
    """OIDC (Keycloak) issuer settings for team mode (PLAN.md §11, task 52)."""

    issuer: str | None = None
    audience: str | None = None
    jwks_cache_s: int = 3600
    groups_claim: str = "groups"
    user_claim: str = "preferred_username"


class AuthConfig(_StrictModel):
    """Authentication mode for the HTTP API and web UI.

    In ``oidc`` mode ``issuer`` and ``audience`` are required; ``admin_principals``
    lists the usernames (or ``group:<name>`` entries) allowed to run mutations
    (tag merges, version review, entity merges) and see the status endpoint.
    With ``trust_proxy_header`` the ``X-Forwarded-Access-Token`` header is
    accepted in addition to the bearer token and the ``mynotes_token`` cookie
    (run behind an auth proxy such as oauth2-proxy).
    """

    mode: Literal["none", "oidc"] = "none"
    oidc: OIDCConfig = Field(default_factory=OIDCConfig)
    trust_proxy_header: bool = False
    admin_principals: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _oidc_mode_needs_issuer_and_audience(self) -> AuthConfig:
        """Require the issuer and audience when ``mode`` is ``oidc``."""
        if self.mode == "oidc" and (not self.oidc.issuer or not self.oidc.audience):
            raise ValueError("auth.oidc.issuer and auth.oidc.audience are required in oidc mode")
        return self


def missing_model_settings(settings: Settings) -> list[str]:
    """The model settings an installation must still provide, as dotted keys.

    The embedding endpoint, model and dimension and the chat endpoint and model
    are required; the reranker is optional, but a reranker url needs a model.
    """
    models = settings.models
    missing = [
        key
        for key, value in (
            ("models.embedding.base_url", models.embedding.base_url),
            ("models.embedding.model", models.embedding.model),
            ("models.embedding.dim", models.embedding.dim),
            ("models.llm.base_url", models.llm.base_url),
            ("models.llm.model", models.llm.model),
        )
        if not value
    ]
    if models.reranker.url and not models.reranker.model:
        missing.append("models.reranker.model")
    return missing


class ModelsNotConfigured(RuntimeError):
    """A model endpoint is used before ``models.*`` is configured."""

    def __init__(self, keys: list[str]) -> None:
        super().__init__(
            "model endpoints not configured: "
            + ", ".join(keys)
            + " (set them in mynotes.yaml or MYNOTES_* variables; see mynotes.example.yaml)"
        )
        self.keys = keys


class FlatEnvSource(PydanticBaseSettingsSource):
    """Maps flat ``MYNOTES_*`` variables onto nested settings paths.

    The names from PLAN.md §12.1 do not follow pydantic's nested naming, so
    each variable is listed with its explicit settings path. Values are read
    from the process environment first, then from ``.env`` in the current
    directory (the process wins). Empty values are treated as unset, so an
    unfilled ``.env`` template never clobbers YAML values or defaults.
    """

    ENV_VAR_TO_PATH: ClassVar[dict[str, str]] = {
        "MYNOTES_PG_DSN": "postgres.dsn",
        "MYNOTES_OPENSEARCH_URL": "opensearch.url",
        "MYNOTES_OPENSEARCH_USER": "opensearch.user",
        "MYNOTES_OPENSEARCH_PASSWORD": "opensearch.password",
        "MYNOTES_OPENSEARCH_CHUNKS_ALIAS": "opensearch.chunks_alias",
        "MYNOTES_EMBEDDING_BASE_URL": "models.embedding.base_url",
        "MYNOTES_EMBEDDING_API_KEY": "models.embedding.api_key",
        "MYNOTES_EMBEDDING_MODEL": "models.embedding.model",
        "MYNOTES_EMBEDDING_DIM": "models.embedding.dim",
        "MYNOTES_EMBEDDING_QUERY_TEMPLATE": "models.embedding.query_template",
        "MYNOTES_EMBEDDING_PASSAGE_PREFIX": "models.embedding.passage_prefix",
        "MYNOTES_RERANKER_PROVIDER": "models.reranker.provider",
        "MYNOTES_RERANKER_URL": "models.reranker.url",
        "MYNOTES_RERANKER_API_KEY": "models.reranker.api_key",
        "MYNOTES_RERANKER_MODEL": "models.reranker.model",
        "MYNOTES_RERANKER_TOP_N": "models.reranker.top_n",
        "MYNOTES_LLM_BASE_URL": "models.llm.base_url",
        "MYNOTES_LLM_API_KEY": "models.llm.api_key",
        "MYNOTES_LLM_MODEL": "models.llm.model",
        "MYNOTES_LLM_TEMPERATURE": "models.llm.temperature",
        "MYNOTES_LLM_MAX_TOKENS": "models.llm.max_tokens",
        "MYNOTES_LLM_CONTEXT_WINDOW": "models.llm.context_window",
        "MYNOTES_LLM_TIMEOUT_S": "models.llm.limits.timeout_s",
        "MYNOTES_LLM_THINKING_MODEL": "models.llm.thinking_model",
    }

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        """Unused: values are produced wholesale by :meth:`__call__`."""
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        """Build the nested settings dict from ``MYNOTES_*`` values."""
        values: dict[str, str | None] = _read_dotenv()
        values.update(os.environ)
        data: dict[str, Any] = {}
        for env_var, path in self.ENV_VAR_TO_PATH.items():
            value = values.get(env_var)
            if value:
                _set_nested(data, path.split("."), value)
        return data


class Settings(BaseSettings):
    """The full mynotes configuration; see ``mynotes.example.yaml``."""

    workspace: str = "default"
    roots: list[RootConfig] = Field(default_factory=list)
    scan: ScanConfig = Field(default_factory=ScanConfig)
    formats: list[str] = Field(
        default_factory=lambda: ["md", "txt", "pdf", "docx", "pptx", "xlsx", "html", "epub"]
    )
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    text_matching: TextMatchingConfig = Field(default_factory=TextMatchingConfig)
    tags: TagsConfig = Field(default_factory=TagsConfig)
    versions: VersionsConfig = Field(default_factory=VersionsConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    opensearch: OpenSearchConfig = Field(default_factory=OpenSearchConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Prioritize init kwargs, flat ``MYNOTES_*`` variables, then YAML."""
        yaml_kwargs: dict[str, Any] = {}
        yaml_path = _resolve_yaml_path()
        if yaml_path is not None:
            yaml_kwargs["yaml_file"] = yaml_path
        return (
            init_settings,
            FlatEnvSource(settings_cls),
            YamlConfigSettingsSource(settings_cls, **yaml_kwargs),
        )

    @model_validator(mode="after")
    def _root_names_are_unique(self) -> Settings:
        """Require ``roots[].name`` values to be unique."""
        names = [root.name for root in self.roots]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"root names must be unique; duplicated: {duplicates}")
        return self

    def redacted_dict(self) -> dict[str, Any]:
        """Return the settings dump with every secret masked by :func:`mask_secret`."""
        return cast("dict[str, Any]", _redacted(self.model_dump()))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached process-wide settings, reading all sources once."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the :func:`get_settings` cache (used by tests)."""
    get_settings.cache_clear()
