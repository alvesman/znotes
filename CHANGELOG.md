# Changelog

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
