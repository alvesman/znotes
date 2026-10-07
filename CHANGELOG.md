# Changelog

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
