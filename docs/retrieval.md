# How retrieval ranks the notes

This page explains how mynotes finds the passages behind a search or an answer: where the candidate
chunks come from, how they are combined and ranked, and how the final context for the answer is
chosen. It is for developers and for anyone tuning the `retrieval:` settings in `mynotes.yaml`.
Every setting named here can also be set there (see `mynotes.example.yaml`).

The entry point is `retrieve()` in `src/mynotes/retrieval/pipeline.py`. Search (web, CLI, API, MCP)
and Chat both go through it.

```
question ─► analysis ─► hybrid search (k-NN + BM25, fused by OpenSearch) ──┐
                        version expander (always on)                       ├─► weighted RRF ─► rerank ─► adjustments ─► context
                        graph expander (chain-shaped questions)            ┘
```

In short: candidates from several sources are merged with **Reciprocal Rank Fusion** (RRF), which
uses positions only, never raw scores. The fused list is then reordered by the **cross-encoder
reranker**, whose score decides the final order. The reranker is optional; without one, the fused
order is the final order.

## 1. Query analysis

`retrieval/analyze.py`, rules only, no model call.

- Inline filter tokens are removed from the text and become filters: `#tag`, `in:folder`,
  `format:pdf`, `since:`/`until:`, `history:` and `as-of:`. They win over the filters chosen in the
  interface or on the command line.
- **Intent** is *current* by default, or *history* when the question uses a history word
  ("previous", "earlier", "changed", "used to", "antes", "anterior", "mudou", "histórico", and a
  few more) or the `history:` token. History intent includes superseded documents.
- Quoted phrases, full dates written in the question ("2026-01-12") and "when did X change to Y"
  questions are recorded for the adjustments below.
- In Chat, a follow-up question ("and who maintains it?") is first rewritten into a standalone
  question by the chat model, using the last `retrieval.followup_turns` turns (default 3; 0 turns
  this off). Only retrieval uses the rewritten question.

## 2. Candidate sources

### Hybrid search (the main source)

`retrieval/hybrid.py`. One OpenSearch request runs two searches over the chunks:

- **k-NN**: the question's embedding against the chunk embeddings (cosine similarity).
- **BM25**: keyword matching over the chunk text, the document title (boosted ×2) and the heading
  path, each with Portuguese and English stemming and accent folding.

OpenSearch merges the two lists in its `mynotes-hybrid-rrf` search pipeline, with RRF weights
`retrieval.fusion.hybrid_weights` (default `[0.7, 0.3]`: vector first, then keywords) and
`retrieval.fusion.rank_constant` (default 20). Changing either requires running `mynotes init` again,
which updates the pipeline.

The filters (tags, folders, formats, dates, access control, latest versions, document status) go
into **both** searches, so only eligible chunks are ranked.

- It fetches `retrieval.candidates` chunks (default 50). The vector search is approximate (HNSW):
  it considers `retrieval.knn_k` neighbours (default 200) and keeps the best 50. With `k` equal to
  50 it sometimes misses a close neighbour that the exact search ranks in its top 20.
- It then keeps at most 2 chunks per document page, slide or section, so one long note can't fill
  the list.

Three optional adjustments reorder this list before fusion (`retrieval/title_lock.py`):

| Setting | When | Effect |
|---|---|---|
| `title_lock` (on) | The question quotes a note's title or file name | That note's chunks move to the front, loaded from the database if the search missed them |
| `date_lock` (on) | A current-intent question writes a full date | The notes named or dated that day move to the front |
| `change_questions` (on) | "When did X change to Y" | The other dated notes in the top hit's folder join the list |

### Version expander

`retrieval/version_expand.py`, source name `version`, always on. It looks at the first 10 hybrid
hits:

- **Current intent**: when one of them belongs to an archived or duplicate document, it searches the
  newest document of that version chain and adds its best chunks.
- **History intent**: it searches the whole chain of each hit, superseded documents included (old
  snapshots of edited notes only when `versions.index_old_versions` is on).

### Graph expander

`retrieval/graph_expand.py`, source name `graph`. It runs only when `retrieval.graph_expansion`
allows it:
- `auto` (the default): only for **chain-shaped questions**, the ones that point at their subject
  through another note or entity ("who maintains the plugin that the note 'Tag cloud' uses?").
  On direct questions it would mostly add noise.
- `true` or `false`: always or never.

It works in four steps:
1. Seeds come from the entities named in the question and the entities mentioned by the first 10
   hybrid hits (the top document's entities count most; very common entities count less).
   Mentions found by plain name matching count at 0.6 of an extracted mention, or not at all when
   `graph_lexical_mentions` is off.
2. It fetches the subgraph around the seeds (`graph_hops`, default 1) and scores it, with
   `graph_method` `one_hop` (default) or `ppr` (Personalized PageRank). With `graph_freshness`,
   older evidence weighs less.
3. The chunks that are evidence for the best edges become candidates, best first. Each carries a
   *hint*: the name of the entity it was reached through. The query's filters and access rules are
   applied to them again.
4. *Structural* candidates follow: chunks of documents linked from the top 5 hits and of documents
   sharing their `project:` tag, found with a restricted hybrid search.

A source that fails is logged and skipped; the search goes on without it.

## 3. Fusion

`retrieval/fuse.py`. The hybrid list and the expander lists are merged with weighted RRF:

```
score(chunk) = Σ over the sources that returned it:  weight(source) / (rank_constant + rank in that source)
```

- Weights: `retrieval.fusion.source_weights`, default `hybrid: 1.0`, `version: 0.8`, `graph: 0.6`.
  The `rank_constant` is the same as in the OpenSearch pipeline (default 20), so all sources share
  one scale.
- A chunk found by several sources adds up their contributions, so agreement lifts it. With the
  defaults, rank 1 in hybrid search is worth 1/21 ≈ 0.048; rank 1 in the graph source is worth
  0.6/21 ≈ 0.029.
- Ties go to the better hybrid rank. A chunk found by both the text search and the graph keeps the
  graph's hint.

Hybrid search is therefore fused twice: inside OpenSearch (vector with keywords), then with the
expanders. `hybrid_weights` only decides the order inside the hybrid list; `source_weights` decide
how far the expanders can move that list.

RRF scores depend only on rank, so they can't tell whether anything relevant was found. That
decision belongs to the reranker (section 5).

## 4. Reranking

`retrieval/rerank.py`, with any Cohere-style `/v1/rerank` endpoint (`models.reranker`).

**Window.** The first `retrieval.rerank_top` fused hits (default 40) are scored, plus up to 10
hinted graph hits from further down: second-hop evidence often fuses in late. Hits outside the
window are dropped.

**What is scored.** Each chunk is sent with its document title and heading path in front of the
text, unless the text already contains them. Notes made from a template often differ only in
their title.

**Hinted hits** are scored twice more, against:
- the question followed by the hint;
- the question rewritten around the resolved entity: "what is the licence of the tool that the
  notification service is built on?" becomes "what is the licence of *Memoranda*?".

The best of the three scores is kept. Without this, a tool's own note scores low against a question
that never names the tool.

**Final score** used for sorting (the raw reranker score is kept for display):

```
final = rerank score
        × 0.9                     if the document is archived
        × 1.00 to 1.05            newer documents within the same project: tag (newest gets 1.05)
        × recency multiplier      when retrieval.recency_boost is on (the default)
```

The **recency multiplier** is `min + (1 − min) × 0.5^(age / half-life)`, with
`recency_half_life_days` 365 and `recency_min_multiplier` 0.8. A note from today gets ×1.0, a
year-old note ×0.9, and a very old note never less than ×0.8. Age is measured from the `as-of:`
date when the question has one. History questions and "when did it change" questions get no recency
boost.

**After sorting**, three adjustments apply:
1. Notes locked by a quoted title or a named date go back to the front.
2. For change questions, the near-tied best notes are ordered earliest first, with the note just
   before the change second.
3. For chain questions, the best hinted hit (rerank score at least 0.05) moves to second place,
   right after the bridge note, when it would otherwise miss the first `max_docs` documents. For
   three-hop questions two hinted notes get a place.

**Without a reranker** (none configured, the endpoint fails, or Search's *Rerank* option is off),
the fused order is the final order. A failing reranker never fails the search.

## 5. "Nothing relevant"

There is no score threshold on the search itself. An answer counts as having no relevant notes only
when no hit's rerank score reaches `retrieval.min_rerank_score`. The default 0.0 turns this gate
off: rerankers score many correct hits very low (Portuguese text, multi-step and history
questions), and the answer prompt already tells the model to say so when the notes don't answer the
question.

The **Search page** applies a display cut instead: results whose rerank score is under 5% of the
best one are collapsed under "Show N weaker results". A one-word search ("portainer") would otherwise
show many unrelated notes, because the vector search always returns the nearest passages, related
or not.

## 6. From ranked hits to the answer's context

`retrieval/assemble.py`:

1. **Documents.** The first `retrieval.max_docs` distinct documents in ranked order (default 4).
   Optionally up to `extra_docs_max` more (default 2) while their best score is at least
   `extra_docs_min_ratio` × the top score. The default ratio of 1.0 turns this extension off.
2. **Chunks.** Up to `retrieval.max_chunks` hits (default 10) from those documents, in ranked order.
3. **Expansion.** The 2 strongest hits are widened to their whole section or page, up to 600 tokens
   each, so the model sees the surrounding text.
4. **Grouping.** Snippets are grouped per document (and page or slide) under a header with the
   title, path and date, and numbered for the `[n]` citations.
5. **Extra blocks.** Version notes (what replaced what, edit history) and graph facts are appended,
   together at most 20% of the budget.
6. **Budget.** Everything is fitted into `retrieval.context_tokens` (default 3500 tokens).

## Settings at a glance

| Setting (`retrieval.…`) | Default | Section |
|---|---|---|
| `fusion.hybrid_weights` | `[0.7, 0.3]` | 2 (needs `mynotes init`) |
| `fusion.rank_constant` | `20` | 2, 3 (needs `mynotes init`) |
| `fusion.source_weights` | `hybrid 1.0, version 0.8, graph 0.6` | 3 |
| `candidates` | `50` | 2 |
| `knn_k` | `200` | 2 |
| `title_lock`, `date_lock`, `change_questions` | on | 2, 4 |
| `graph_expansion` | `auto` | 2 |
| `graph_hops`, `graph_method` | `1`, `one_hop` | 2 |
| `graph_freshness`, `graph_freshness_half_life_days` | off, `365` | 2 |
| `graph_lexical_mentions` | on | 2 |
| `rerank_top` | `40` | 4 |
| `recency_boost`, `recency_half_life_days`, `recency_min_multiplier` | on, `365`, `0.8` | 4 |
| `min_rerank_score` | `0.0` | 5 |
| `max_docs`, `extra_docs_max`, `extra_docs_min_ratio` | `4`, `2`, `1.0` | 6 |
| `max_chunks` | `10` | 6 |
| `context_tokens` | `3500` | 6 |
| `followup_turns` | `3` | 1 |

To compare settings, use the evaluation harness in `eval/` (`eval/README.md`): change one setting
with `--set` and compare the retrieval metrics against a baseline run.
