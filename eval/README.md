# Eval harness

Measures retrieval and answer quality of the mynotes pipeline against a golden
question set. Run before/after every accuracy-affecting
change; commit or reference the baseline results JSON so deltas are auditable.

## Golden set

`eval/golden_set.yaml`: `meta` (creation date, corpus note) plus one entry per
question with `id`, `question`, `language` (pt/en), `category` (`factual`,
`language`, `table`, `tag-filtered`, `current-version`, `history`,
`multi-hop`, `no-answer`, `global`), free `tags` for slicing, optional
explicit `filters: {tags, folders}`, `expected_docs` (rel paths; any match
counts), optional `expected_pages`, optional `stale_docs`/`stale_values`
(superseded paths/values for the version metrics) and `reference_answer`.

Documents are identified by **rel path** (stable across re-ingestion), not by
database ids. Questions without `expected_docs` (e.g. `no-answer`) have no
measurable retrieval target: their retrieval metrics are `None` and they are
measured by `no_answer_correct` in answers mode.

The 3 example questions point at real corpus files; the real golden set is
task 32.

## Running

```bash
uv run python -m eval.run_eval --mode retrieval --tag baseline   # no answer LLM, fast
uv run python -m eval.run_eval --mode both --tag c2-assembler    # + answers & judge
uv run python -m eval.run_eval --mode retrieval --variants --tag robustness
uv run python -m eval.compare eval/results/<a>.json eval/results/<b>.json
```

Makefile shortcuts: `make eval-retrieval TAG=x`, `make eval-full TAG=x`,
`make eval-compare A=<file> B=<file>`. Add `--ids q001,q002` or `--limit N` to
shrink a run, `--no-judge` to skip the LLM judge.

`--variants` re-runs retrieval for every question in three variants: accents
stripped with the original case kept, all upper case, all lower case. It
records per-variant retrieval metrics and a `parity` block (the delta of
doc_hit@5 and MRR vs the original). Target: no drop larger than 0.02.

## Modes and metrics

- **retrieval**: `retrieve()` per question; `doc_hit@{1,3,5,10}` (any expected
  rel path), `page_hit@k` (expected page/slide), MRR, latency, sliced by
  `language`, `category` and each tag. The metrics are also recorded on the
  **pre-rerank fused list** (`aggregate_fused`), which is what the RRF tuning
  in task 32 optimizes.
- **answers**: `answer()` per question; LLM judge (correctness /
  groundedness / completeness 0–2 + verdict, background priority),
  `current_version_precision` (share of cited sources that are current and not
  in `stale_docs`), `stale_answer` (a `stale_values` string appears in the
  answer, compared case/accent-insensitively), `no_answer_correct`,
  source-attribution correctness, latency. The judge is secondary evidence;
  retrieval metrics are primary.

The runner uses the real local stack and endpoints (Postgres/OpenSearch via
the configured settings, the real embedding/rerank/chat endpoints). No auth
bypass is needed because local mode has no ACL.

## Results and comparability

- Each run writes `eval/results/<UTC timestamp>_<tag>.json` with the git
  commit + dirty flag, the golden-set SHA-256, a config snapshot
  (`retrieval`, `models` without keys, `versions`), the chunk count of the
  chunks alias, per-question rows and aggregates, and prints the overall
  aggregates.
- Never edit `golden_set.yaml` between two runs you intend to compare
  (`compare.py` warns on a dataset hash mismatch).
- `eval/results/*.json` is gitignored; commit chosen baselines explicitly
  with `git add -f`.
