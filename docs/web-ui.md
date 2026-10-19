# Web UI

The web UI (tasks 47/48) is server-rendered Jinja2 + HTMX served by the same
FastAPI app as the JSON API (`mynotes serve`):

- Chat: `/ui/chat` — streams `POST /v1/ask/stream` (SSE) with `fetch` +
  `response.body.getReader()`. Answers are rendered as Markdown by
  `static/markdown.js` (DOM nodes only, never `innerHTML`: headings, lists,
  tables, quotes, code blocks with a Copy button, bold/italic/inline code, `[n]`
  markers linked to the citations). Enter asks, Shift+Enter adds a line. Each
  page load creates a `conversation_id` (sent with every question) so the turns
  of one chat are grouped in History; `?conversation=<key>` resumes a stored
  chat (its turns are rendered server-side, new questions keep its id).
  Follow-ups: when the conversation already has turns,
  `service.ask_stream` rewrites the question into a standalone one with the last
  `retrieval.followup_turns` turns (`retrieval/followup.py`, one short LLM call)
  and sends it first as a `rewritten` SSE event (shown as "Searched as: …"),
  retrieves with it, and passes those turns to the answer prompt as earlier
  user/assistant messages (answers cut to 1,200 characters, `[n]` markers
  removed). Finished answers are stored in `chat_entry` (migrations 0006-0008)
  unless the request sends `save_history: false`.
- History: `/ui/history` — the caller's conversations (`core/chat_history.py`,
  scoped by `Principal.user`), most recently used first, newest 200; each opens
  to its turns (answers rendered by `markdown.js` via `history.js`, per-turn
  citation anchors `h<id>-cite-<n>`) and a "Continue this chat" link. The cleanup
  form posts to `POST /ui/partials/history/delete` (`scope=older&days=N` deletes
  conversations whose last turn is older than N days, `scope=all` deletes all).
- Search: `/ui/search` — facet form (folder, formats, since, history, rerank,
  then a tag picker at the bottom) posting via HTMX to `POST /ui/partials/search`,
  which renders the `partials/search_results.html` fragment. Excerpts are rendered
  by `markdown.js` (`search.js` on `htmx:afterSwap`). With rerank on, hits below
  5% of the best rerank score are collapsed behind "Show N weaker results"
  (`routes.split_weak_hits`; answers are not affected). The tag picker
  lists tags in use (most used first) as checkbox chips; `search.js` filters
  them by name (accent/case-folded), shows the selection count with a Clear
  button, and re-submits the form when a filter changes.
- Option checkboxes marked `data-persist` (Chat's history option, Search's
  history and rerank) are remembered per browser in `localStorage` by
  `static/prefs.js`; filters (tags, formats) are not.
- Citations ("View note" / "View text") open the document page in a new tab;
  results no longer show "Raw" (the document page has it).
- Document view: `/ui/documents/{id}` — extracted Markdown rendered
  server-side (`markdown_it`, raw HTML disabled, tables enabled), headings
  anchored as `sec-<slug>` (task 48's folding slug, mirrored in `chat.js`),
  fenced code blocks wrapped in `.code-block[data-copy-code]` (`render._wrap_fences`;
  indented blocks are not) and given a Copy button by `markdown.js`,
  tags grouped by source with Accept/Reject for proposed LLM tags (HTMX posts
  to `/v1/tags/{doc}/{tag}/accept|reject`, the `mynotes:tags-changed` event
  re-renders the chips), the supersedes chain and the snapshot history.
  `?chunk=<ordinal>` wraps the matching paragraphs in `<mark>` (best-effort
  60-character prefix match).
- Review queues: `/ui/reviews` — three HTMX tabs (versions / tags / entities).
  Version rows post to `/v1/versions/{id}/confirm|reject|swap`, tag rows to
  `/v1/tags/...` plus a `Merge into…` form posting JSON to `/v1/tags/merge`
  (vendored `json-enc` extension), and entity candidates merge or mark
  distinct via `POST /ui/reviews/entities/{keep}/{drop}/merge|{a}/{b}/distinct`,
  which re-render the tab. The tab partials re-render on the
  `mynotes:versions-changed` / `mynotes:tags-changed` events that those API
  responses carry as `HX-Trigger` headers.
- Graph explorer: `/ui/graph?entity=REF` — entity search box (HTMX lookup),
  entity card (summary, aliases, current attribute values), a Cytoscape
  neighbourhood view fed by `/v1/graph/entities/{ref}/neighbors?hops=N`
  (nodes colored by type, edges labeled with the relation type, non-active
  edges dashed, edge click shows the evidence documents) and a dated timeline.
- Status: `/ui/status` — dashboard over the facade `status()` (roots with
  watcher state and last scans, queue by status and type, KG backlog with ETA,
  alias → physical index with doc counts and drift report, `inspect_live`
  problems highlighted, endpoint snapshots, unsupported/placeholder files).
  The panel auto-refreshes every 30 s and re-renders after `POST /v1/scan`
  ("Scan now").
- Help: `/ui/help[/{cli|web|mcp}]` (tabs in that order; `cli` is the default) — the manuals in `docs/` (`web-interface-manual.md`,
  `user-manual.md`, `mcp.md`) rendered with the document-view parser
  (`render_help_markdown`): headings get `sec-<slug>` ids, the manuals' GitHub-style
  anchors are rewritten to them, links between the manuals become Help tabs, and other
  relative links point at the repository. `DOCS_DIR` is the repo's `docs/` folder
  (`/app/docs` in the Docker image, which copies it).
- Static assets: `/static/app.css`, `/static/markdown.js`, `/static/chat.js`,
  `/static/search.js`, `/static/history.js`, `/static/prefs.js`,
  `/static/graph.js` and the vendored releases below, referenced through the
  `asset()` template global, which appends the file's mtime as `?v=` so a changed
  script or stylesheet is never served from the browser cache. There is no CDN
  and no inline script/style (the CSP forbids them).

## Vendored HTMX

Pinned release: **htmx 2.0.11** (`htmx.org`, 0BSD license), downloaded from
<https://unpkg.com/htmx.org@2.0.11/dist/htmx.min.js> and saved as
`src/mynotes/web/static/htmx-2.0.11.min.js`. To upgrade, download the new
release file, update the `<script src>` in `base.html` and the version here.

htmx's own injected indicator styles are disabled via the `htmx-config` meta
tag because the UI CSP (`style-src 'self'`) blocks inline `<style>` elements;
the equivalent rules live in `app.css`.

## Vendored Cytoscape and json-enc

Pinned release: **cytoscape 3.34.3** (MIT license), taken from the npm
tarball `https://registry.npmjs.org/cytoscape/-/cytoscape-3.34.3.tgz`
(`dist/cytoscape.min.js`) and saved as
`src/mynotes/web/static/cytoscape-3.34.3.min.js`. Only the graph explorer
loads it. To upgrade, replace the file and the version in this section.

Pinned release: **htmx-ext-json-enc 2.0.3** (BSD-2-Clause license), taken
from the npm tarball
`https://registry.npmjs.org/htmx-ext-json-enc/-/htmx-ext-json-enc-2.0.3.tgz`
(`dist/json-enc.js`) and saved as
`src/mynotes/web/static/json-enc-2.0.3.js`. The review page loads it so the
`Merge into…` form can post a JSON body to `/v1/tags/merge`.

