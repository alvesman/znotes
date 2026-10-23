# Changelog

## 0.4.0rc4 (2026-10-23, preview of server mode)

- **Knowledge graph:** a section group whose extraction answer does not parse (usually a dense
  passage whose JSON ran past the model's output cap) is now split at paragraph boundaries and
  extracted in parts, up to four parts, and the merged result is cached. Before, the group was
  skipped and its entities were missing from the graph.
- **`mynotes graph backfill --retry-errors`** forgets the groups whose extraction failed, so
  their documents are queued again (for example after this fix).

## 0.4.0rc3 (2026-10-23, preview of server mode)

- **Extraction:** a note whose leading `---` block is not valid YAML (an Obsidian `%% comment %%`,
  a template, or simply a horizontal rule) is now extracted as text with a warning instead of
  failing; a PDF for which the table finder breaks is extracted again without table detection.
- **New `mynotes maintenance retry-extract`:** queues extraction again for every document whose
  last extraction failed (`--dry-run` counts them). Until now a failed file was only retried when
  its content changed.

## 0.4.0rc2 (2026-10-23, preview of server mode)

- **Server mode:** `mynotes token create --root NAME` creates the remote root for the token's user
  when it does not exist yet; `deployment/server` publishes the web port (`MYNOTES_WEB_BIND`) and
  makes Caddy optional (`--profile tls`) for hosts with their own TLS proxy; agents must run the
  server's version.

## 0.4.0rc1 (2026-10-23, preview of server mode)

- **Server mode:** one mynotes server for several people, each feeding their notes from their own
  machine with the new `mynotes agent` (it watches the folder and uploads files; no database,
  search index or model endpoint on that machine). The server keeps the uploaded files, indexes
  them, and shows each person only their own notes. New commands `mynotes root add --remote`,
  `mynotes token create|list|revoke`, `mynotes agent run|sync|status`; a sign-in page for the
  web UI (`auth.mode: token`); the ingest API under `/v1/roots/{root}` (`docs/api.md`); the
  `deployment/server/` compose files with a TLS proxy. Single-machine installs are unchanged.
- **Status:** `POST /v1/scan` and `/v1/status` require an admin in every auth mode; the status page
  lists agents and their last heartbeat, and the size of the uploaded-files store.

## 0.3.0 (2026-10-20)

- **Web interface redesigned:** a cleaner, responsive layout that works on phones (the menu folds
  behind a button, panels collapse, controls are touch-sized, tables scroll) and uses the width
  of a desktop screen (search filters beside the results, the graph beside its entity card,
  status cards in a grid). Chat's question box stays at the bottom of the page. Keyboard and
  screen-reader fixes: skip link, visible focus, labels on every input, tab roles on Reviews.
- **Theme switch:** choose **System**, **Light** or **Dark** at the top right; the choice is
  remembered in the browser, and the graph recolours with it.
- **Status page:** the panel no longer refreshes twice (its wrapper carried the same id as the
  panel itself).
- **Docs:** the web manual now says correctly that Enter sends a question and Shift+Enter adds a
  line.

- **Graph explorer:** a name shared by an entity and a duplicate that was merged into it opened a
  list of two identical-looking entries that led nowhere. Lookups now resolve merged records to
  the live entity (web UI, CLI, API and MCP); `graph unmerge` still finds the merged record. When
  several entities really share a name, the list's links use the entity id (`#519`), which the
  search box and `graph entity` also accept.
- **Graph explorer:** well-connected entities showed "The neighbourhood could not be loaded". The
  neighbours API returned the edges of the whole neighbourhood but only the 20 best neighbours, and
  the drawing failed on an edge to an unlisted entity. It now returns only the edges between the
  listed nodes, and the page skips any edge it can't draw.

## 0.2.0 (2026-10-19)

- **Note pages:** code blocks written between triple backticks have a Copy button, as in Chat
  answers, and are shown in a shaded box.
- **Better recall from the vector search:** it now considers 200 neighbours before keeping the
  best 50 candidates (new setting `retrieval.knn_k`). With 50, the approximate search sometimes
  missed a close match.
- **Removed** the `retrieval.query_translation` setting, which did nothing. Delete it from your
  `mynotes.yaml` if you copied it from the 0.1.0 example, or mynotes won't start.
- **Documentation:** `docs/retrieval.md` explains how candidates are found, fused, reranked and
  turned into the answer's context, with every setting that tunes them.
- **Evaluation harness:** stable dev and hold-out splits of each golden set (`--split`), metrics for
  whether the right note is among the candidates and reaches the reranker (`pool_hit`,
  `window_hit`, with a summary of where questions are lost), and
  `build_golden_set --alternates` to find notes that repeat a reference answer.

## 0.1.0 (2026-10-07)

First public release.

- Watches one or more folders of notes (Markdown, text, PDF, Office documents) and indexes them into
  PostgreSQL and OpenSearch: hybrid keyword and vector search with optional reranking.
- Answers questions with citations to the notes, keeps previous versions of each note, and builds a
  knowledge graph of the people, projects and terms they mention.
- Web interface (Chat, Search, History, Reviews, Graph, Status), command line, HTTP API and MCP server.
- Works with any OpenAI-compatible embedding and chat endpoint (OpenAI, Ollama, vLLM, TEI). There are
  no built-in model defaults: configure your own in `.env` or `mynotes.yaml`.
- Docker image `ghcr.io/alvesman/znotes` and a Docker Compose deployment.
- Licensed under the GNU AGPL v3 or later.
