# mynotes user manual

This manual explains how to install, use and look after mynotes. It assumes no prior knowledge of the system: every command is shown in full, and new terms are explained when they first appear.

## Contents

1. [What mynotes is](#1-what-mynotes-is)
2. [Before you start](#2-before-you-start)
3. [Installation and first-time setup](#3-installation-and-first-time-setup)
4. [Starting and stopping mynotes](#4-starting-and-stopping-mynotes)
5. [Asking questions](#5-asking-questions)
6. [Searching](#6-searching)
7. [Filters: narrowing a question or a search](#7-filters-narrowing-a-question-or-a-search)
8. [The web interface](#8-the-web-interface)
9. [Using mynotes from Claude (MCP)](#9-using-mynotes-from-claude-mcp)
10. [Keeping the index up to date](#10-keeping-the-index-up-to-date)
11. [Tags](#11-tags)
12. [Versions and successors](#12-versions-and-successors)
13. [The knowledge graph](#13-the-knowledge-graph)
14. [Configuration](#14-configuration)
15. [Maintenance, backups and updates](#15-maintenance-backups-and-updates)
16. [Troubleshooting](#16-troubleshooting)
17. [Shared (team) use](#17-shared-team-use)
18. [Command reference](#18-command-reference)
19. [Glossary](#19-glossary)

---

## 1. What mynotes is

mynotes answers questions about your own notes. You point it at one or more folders of documents (Markdown notes, PDFs, Word and PowerPoint files, and so on). It reads them, remembers what is in them, and then:

- **answers questions** in plain language, quoting the documents it used as numbered citations (`[1]`, `[2]`, …) so you can check the source;
- **finds passages** that match a search, even when your words differ from the document's ("deadline" also finds "due date"), in Portuguese and English, ignoring upper/lower case and accents;
- **keeps up with changes**: when you add, edit, move or delete a file, mynotes notices and updates what it knows;
- **knows about versions**: when a document replaces an older one, answers use the newer one, while the older one stays available for questions about history;
- **organises with tags**, taken from your folder names, from tags you wrote in the documents, and from suggestions made by an AI model;
- **builds a knowledge graph**: a map of the people, projects, technologies and other things your notes mention, and how they relate.

Everything stays on your computer except the text sent to three AI model services (see [Before you start](#2-before-you-start)).

### How the pieces fit together

```
  Your notes folders
         │  (read only: mynotes never changes your files)
         ▼
  ┌──────────────┐     ┌──────────────────────────────┐
  │   worker     │────▶│ PostgreSQL  (what mynotes    │
  │ (background) │     │             knows: documents,│
  └──────────────┘     │             versions, tags)  │
         │             │ OpenSearch  (the search      │
         │             │             index)           │
         ▼             └──────────────────────────────┘
  AI model services                ▲
  (embeddings, reranker, LLM)      │
                                   │
  You:  command line  ·  web interface  ·  Claude (MCP)
```

- **The worker** is a program that runs in the background. It scans your folders, reads new and changed files, and updates the databases. If the worker is not running, nothing gets updated, but you can still ask questions about what is already indexed.
- **PostgreSQL** and **OpenSearch** are two databases. They run inside **Docker**, a tool that runs software in isolated containers, so you don't have to install them yourself.
- **The AI model services** do the language work: an *embedding* model turns text into numbers so similar meanings can be found, a *reranker* sorts search results by relevance, and an *LLM* (large language model) writes the answers, suggests tags and builds the knowledge graph.
- **You** talk to mynotes through the command line, a web page in your browser, or Claude.

---

## 2. Before you start

### What you need

| Need | Why | How to check |
|---|---|---|
| A computer running Linux, macOS or Windows | | |
| **Docker** (Docker Engine on Linux, Docker Desktop on macOS and Windows) | Runs the two databases | `docker --version` and `docker compose version` print version numbers |
| **uv**, the Python package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/)) | Installs and runs mynotes. It downloads Python 3.12 by itself if needed | `uv --version` prints a version number |
| **git** | Downloads the mynotes code | `git --version` |
| Access to AI models: an embedding model and a chat model, and optionally a reranker | mynotes has no built-in models. Any OpenAI-compatible service works: OpenAI, Ollama on your own computer, vLLM. See [Choosing models](#choosing-models) | |
| About 4 GB of free memory for the databases, plus 1–4 GB while the worker indexes | OpenSearch alone uses 2 GB by default | |
| A folder of notes | | |

mynotes reads these file types:

- Markdown (`.md`, `.markdown`) and plain text (`.txt`)
- PDF (`.pdf`)
- Word (`.docx`), PowerPoint (`.pptx`) and Excel (`.xlsx`)
- HTML (`.html`, `.htm`)
- EPUB e-books (`.epub`)

Other files in your folders are ignored.

### Choosing models

mynotes uses three kinds of AI model, all reached over the network through standard APIs. There are no defaults: you choose them and write their addresses in `.env`.

| Model | Required | What it does | API |
|---|---|---|---|
| **Embedding model** | yes | Turns passages and questions into vectors, so search finds passages by meaning | OpenAI-compatible `/v1/embeddings` |
| **Chat model (LLM)** | yes | Writes the answers, suggests tags, builds the knowledge graph, rewrites follow-up questions | OpenAI-compatible `/v1/chat/completions` |
| **Reranker** | no | Re-orders the best search results more precisely. Without one, results keep their search order, and Search can't hide weaker results | Cohere-style `/v1/rerank` |

Three common setups:

| Setup | Embedding | Chat | Reranker | Good to know |
|---|---|---|---|---|
| **Everything on your computer** with [Ollama](https://ollama.com) | `bge-m3` (dimension 1024), address `http://localhost:11434/v1` | a chat model such as `qwen3:8b` | none (Ollama has no rerank API) | Your notes never leave your machine. Needs a reasonably recent computer; building the knowledge graph is slow on a laptop |
| **OpenAI** | `text-embedding-3-small` (dimension 1536), address `https://api.openai.com/v1` | `gpt-4o-mini` | none | Easiest to set up. The text of your notes and questions is sent to OpenAI, and you pay per use |
| **Your own GPU server** with vLLM or Text Embeddings Inference | the model you serve, its dimension | the model you serve | e.g. `BAAI/bge-reranker-v2-m3` served by vLLM at `/v1/rerank` | Best quality and privacy if you have the hardware |

Things to get right:

- **The embedding dimension** (`MYNOTES_EMBEDDING_DIM`) must match the model; the model card says it. Changing the embedding model or dimension later means rebuilding the search index (`uv run mynotes index rebuild`), which re-embeds every passage.
- **The query format** (`MYNOTES_EMBEDDING_QUERY_TEMPLATE`): most models take the question as typed (`{text}`, the default). Some want an instruction or prefix, written in their model card, for example `Instruct: {task}\nQuery: {text}` for Nemotron and E5-Mistral, or `search_query: {text}` for nomic-embed-text.
- **Thinking models**: set `MYNOTES_LLM_THINKING_MODEL=true` only for Qwen3-style models served by vLLM, whose thinking mode mynotes then switches off. Leave it `false` for OpenAI and Ollama.
- **Privacy**: with a hosted service, the passages used for each answer, your questions and, while the knowledge graph is built, the text of your notes are sent to that service.

Check your choices at any time with `uv run mynotes check-endpoints` (section 3, step 7).

### Using the terminal

All commands in this manual are typed into a terminal (Terminal on macOS and Linux, PowerShell on Windows) and run by pressing Enter.

- **Always start in the mynotes folder.** mynotes reads its secrets file (`.env`) from the folder you are in. If you see errors about missing passwords or connections, first check that you are in the right folder:
  ```bash
  cd ~/mynotes
  ```
  (Use the folder where you downloaded mynotes. This manual uses `~/mynotes`.)
- **Every mynotes command starts with `uv run mynotes`.** `uv run` makes sure the right Python and packages are used.
- **Put quotes around anything with spaces or special characters**, such as questions, paths and citation numbers:
  ```bash
  uv run mynotes tags show "Project Notes/Alfa plan.md"
  uv run mynotes open "[1]"
  ```
- **Help is built in.** Add `--help` to any command to see what it does and which options it takes:
  ```bash
  uv run mynotes --help
  uv run mynotes tags --help
  uv run mynotes tags show --help
  ```
- **Commands with several parts.** Some commands are groups (`tags`, `versions`, `graph`, `index`, `watch`, `service`, `maintenance`, `acl`). A group on its own only shows an error such as `Missing command.`: you have to add what you want to do, for example `uv run mynotes tags list`.
- **Stopping a running command.** Commands that keep running (`worker`, `serve`) stop with **Ctrl+C**.

---

## 3. Installation and first-time setup

You only do this once. It takes about 15 minutes, plus the time to index your notes.

> **Several people, or your notes on several machines?** Install the *server* variant instead: one server in Docker and a small agent on each machine; see [deployment/server/README.md](../deployment/server/README.md) and section 17. The steps below set up mynotes for one person on one computer.

### Step 1: Download mynotes

```bash
cd ~
git clone https://github.com/alvesman/znotes.git mynotes
cd ~/mynotes
```

Replace the address with the repository you were given, if different.

### Step 2: Install mynotes's Python packages

```bash
cd ~/mynotes
uv sync
```

This creates a private Python environment inside the `mynotes` folder. It doesn't affect anything else on your computer.

### Step 3: Create the secrets file (`.env`)

The `.env` file holds the database passwords and the AI services' addresses and keys.

```bash
cd ~/mynotes
cp .env.example .env
chmod 600 .env
```

`chmod 600` makes the file readable only by you (skip it on Windows). Open `.env` in a text editor and fill in the empty values:

| Variable | What to put there |
|---|---|
| `MYNOTES_COMPOSE_PG_PASSWORD` | A password you invent for the PostgreSQL database. Use letters and digits only |
| `MYNOTES_PG_DSN` | The database address. Replace `<password>` with the same password: `postgresql+psycopg://mynotes:YOURPASSWORD@localhost:5432/mynotes` |
| `MYNOTES_OPENSEARCH_PASSWORD` | A password you invent for OpenSearch. It must have at least 8 characters, with upper case, lower case, a digit and a special character, for example `Notes-2026-ok` |
| `MYNOTES_EMBEDDING_BASE_URL`, `_MODEL`, `_DIM` (and `_API_KEY` if the service needs one) | Your embedding model: address, model name and its dimension. See [Choosing models](#choosing-models); the file lists ready-made lines for OpenAI and Ollama |
| `MYNOTES_LLM_BASE_URL`, `_MODEL` (and `_API_KEY`) | Your chat model |
| `MYNOTES_RERANKER_URL`, `_MODEL` (and `_API_KEY`) | Optional: your reranker. Leave empty to do without |

Leave the other values as they are. **Never share or commit `.env`**: it contains your passwords and keys.

### Step 4: Create the settings file (`mynotes.yaml`) and choose your notes folders

The settings file tells mynotes which folders to read and how to behave. Copy the example to your user configuration folder:

**Linux:**
```bash
mkdir -p ~/.config/mynotes
cp ~/mynotes/mynotes.example.yaml ~/.config/mynotes/mynotes.yaml
```

**macOS:**
```bash
mkdir -p ~/Library/Application\ Support/mynotes
cp ~/mynotes/mynotes.example.yaml ~/Library/Application\ Support/mynotes/mynotes.yaml
```

**Windows (PowerShell):**
```powershell
New-Item -ItemType Directory -Force "$env:LOCALAPPDATA\mynotes\mynotes"
Copy-Item "$HOME\mynotes\mynotes.example.yaml" "$env:LOCALAPPDATA\mynotes\mynotes\mynotes.yaml"
```

Open the copied `mynotes.yaml` and edit the `roots:` section near the top. Each **root** is a folder mynotes reads, including all its subfolders. Give each a short name (letters, no spaces) and its path:

```yaml
roots:
  - name: ideas
    path: ~/Documents/ideas
```

For several folders, add one entry per folder:

```yaml
roots:
  - name: ideas
    path: ~/Documents/ideas
  - name: work
    path: ~/Work/notes
```

On Windows, write paths like `C:\Users\me\Documents\ideas`. Delete the example roots you don't use (the example file includes a network-share root, `shared`, that you probably don't have).

The rest of the file is optional; every option is explained by a comment, and [section 14](#14-configuration) describes the common ones.

### Step 5: Start the databases

```bash
cd ~/mynotes
docker compose up -d
```

The first time, Docker downloads PostgreSQL and OpenSearch (about 1 GB). Check that both are running and healthy:

```bash
docker compose ps
```

Wait until both lines show `(healthy)`. OpenSearch takes up to a minute after starting.

### Step 6: Prepare mynotes

```bash
cd ~/mynotes
uv run mynotes init
```

`init` creates the database tables, registers your roots and creates the search indexes. It prints a table of checks; each should say `ok`. Warnings about `tesseract` and `ocrmypdf` are normal: those tools are only needed to read scanned PDFs, which is off by default.

### Step 7: Check the AI services

```bash
cd ~/mynotes
uv run mynotes check-endpoints
```

Every line should say `ok`. If one fails, check the matching address and key in `.env`.

### Step 8: Index your notes

Start the worker. It scans your folders straight away and processes everything it finds:

```bash
cd ~/mynotes
uv run mynotes worker
```

Leave this terminal open: the worker keeps running until you press Ctrl+C. (To have it start automatically instead, see [Start the worker and the web interface automatically](#start-the-worker-and-the-web-interface-automatically).)

Open a **second terminal** to watch the progress:

```bash
cd ~/mynotes
uv run mynotes status
```

What happens, and how long it takes:

1. **Scanning** finds your files (seconds).
2. **Reading, splitting and embedding**: each file's text is extracted, split into passages (*chunks*) and sent to the embedding service. A few thousand files take a few minutes. **Search and questions work as soon as this is done.**
3. **Tagging and the knowledge graph**: the LLM suggests tags and extracts people, projects and relations from each document. This is much slower (roughly 10 documents per minute) and runs in the background. You can use mynotes normally meanwhile. Follow it with:
   ```bash
   uv run mynotes graph extract-status
   ```

### Step 9: Ask your first question

```bash
cd ~/mynotes
uv run mynotes ask "What projects am I working on?"
```

Setup is complete. The next sections explain everyday use.

---

## 4. Starting and stopping mynotes

mynotes has three parts that can run independently:

| Part | Start | Stop | Needed for |
|---|---|---|---|
| Databases | `docker compose up -d` | `docker compose stop` | Everything |
| Worker, interactive (runs in a terminal you keep open) | `uv run mynotes worker` | Ctrl+C in that terminal, or close it | Picking up changes to your notes |
| Worker, as a service (runs in the background, starts at every login) | `uv run mynotes service install` (once) | `uv run mynotes service uninstall` | Picking up changes to your notes |
| Web interface, interactive | `uv run mynotes serve` | Ctrl+C | The browser pages and the HTTP API |
| Web interface, as a service (starts at every login) | `uv run mynotes service install --component web` (once) | `uv run mynotes service uninstall --component web` | The browser pages and the HTTP API |

Run each part one way or the other, not both: one worker and one web server are enough. The interactive way suits trying mynotes out or watching what happens; the services suit everyday use ([details](#start-the-worker-and-the-web-interface-automatically)).

The command line (`ask`, `search`, …) and Claude only need the databases.

**After restarting your computer**, the databases start again by themselves if Docker starts automatically (the default for Docker Desktop; on Linux, `sudo systemctl enable docker`): the containers are configured to restart unless you stopped them. A worker or web interface installed as a service starts again when you log in; interactive ones have to be started again by hand. To have everything up after a reboot, once:

```bash
cd ~/mynotes
sudo systemctl enable docker                       # Linux only: Docker itself at boot
uv run mynotes service install --component all     # worker and web interface at login
loginctl enable-linger $USER                       # Linux only: also without logging in
```

**Check what is running:**

```bash
cd ~/mynotes
docker compose ps                  # databases
uv run mynotes service status      # automatic worker service
uv run mynotes status              # everything mynotes knows about its own state
```

`uv run mynotes status` shows several panels: the settings file in use, the database (documents, versions, queued work), OpenSearch (the indexes and their document counts), index drift (differences between the database and the search index), files mynotes could not read, and the AI services (calls, response times and errors, per process: the worker reports its own every 30 seconds while it runs). Add `--json` for machine-readable output:

```bash
uv run mynotes status --json
```

---

## 5. Asking questions

```bash
cd ~/mynotes
uv run mynotes ask "What is the deadline for the Alfa project?"
```

mynotes finds the most relevant passages in your notes, sends them to the LLM, and prints an answer as it is written, followed by the list of sources. Each `[n]` in the answer refers to a numbered source: its title, its path, and the page, slide or section it came from.

Good to know:

- **Ask in Portuguese or English.** The answer comes in the language of your question, whatever the language of the documents.
- **Case and accents don't matter**: `"Qual é o prazo?"`, `"qual e o prazo?"` and `"QUAL É O PRAZO?"` find the same documents.
- **Quote a note's title to ask about that note.** `"O que diz a nota "Plano Alfa" sobre o prazo?"`: the note whose title (or file name) matches the quoted text is used first, even when other notes with similar titles score higher. Useful for weekly notes and other series that share a title pattern.
- **Write the date to ask about one day's note.** `"Na reunião de 2026-01-12, quem ficou com o deploy?"` uses the notes named or dated that day first, instead of the most recent week. Dates can be written `2026-01-12`, `12/01/2026`, `12 de janeiro de 2026` or `January 12, 2026`. (In a question about the past, with `as-of:`, the date is a cutoff instead.)
- **Ask when something changed.** `"Em que reunião o deploy passou a DEMO?"` or `"When did the status change to done?"` looks through the whole series of notes and answers with the earliest one that states the new value, showing the week before it for comparison.
- **If your notes don't contain the answer, mynotes says so** instead of guessing.
- **Only current documents are used**, not versions that were replaced, unless you ask about history (see [Filters](#7-filters-narrowing-a-question-or-a-search)).

### Options

| Option | Effect | Example |
|---|---|---|
| `--tag TAG` | Only documents with this tag (repeat for several) | `uv run mynotes ask "Who is the client contact?" --tag clienta` |
| `--in FOLDER` | Only documents under this folder (repeat for several) | `uv run mynotes ask "What is the budget?" --in "projects/alfa"` |
| `--format FORMAT` | Only this file type: `md`, `txt`, `pdf`, `docx`, `pptx`, `xlsx`, `html`, `epub` | `uv run mynotes ask "What does the contract say about payment?" --format pdf` |
| `--since DATE` | Only documents dated on or after `YYYY-MM` or `YYYY-MM-DD` | `uv run mynotes ask "What did we decide?" --since 2026-01` |
| `--history` | Also use older, replaced versions | `uv run mynotes ask "How did the architecture change?" --history` |
| `--as-of DATE` | Answer as things stood on that date | `uv run mynotes ask "Who led the project?" --as-of 2025-06` |
| `--no-stream` | Print the answer all at once instead of word by word | `uv run mynotes ask "Summarise the Alfa plan" --no-stream` |
| `--json` | Print the full result (answer, sources, scores) as JSON | `uv run mynotes ask "Summarise the Alfa plan" --json` |

### Opening a cited document

After `ask`, open any cited source in its usual application (a PDF viewer, Word, a text editor…) by its number:

```bash
uv run mynotes open "[1]"
```

Keep the quotes: some shells treat `[1]` as a file pattern otherwise. You can also open a document by its path, its title or its numeric id:

```bash
uv run mynotes open "projects/alfa/plan.md"
uv run mynotes open "Alfa plan"
uv run mynotes open 42
```

If several documents match a title, mynotes lists them and you run the command again with the path.

---

## 6. Searching

`search` finds and ranks matching passages without asking the LLM to write an answer. Use it when you want to see where something is written.

```bash
cd ~/mynotes
uv run mynotes search "kubernetes ingress"
```

The result is a table with the rank, the document's title and path, the section, the page or slide, and the score.

| Option | Effect | Example |
|---|---|---|
| `--limit N` or `-n N` | Show at most N results (default 10) | `uv run mynotes search "budget" -n 5` |
| `--tag`, `--in`, `--format`, `--since`, `--history` | Same filters as `ask` | `uv run mynotes search "budget" --tag clienta --format xlsx` |
| `--raw` | Skip the smart parts (filter detection, reranking): a plain keyword + meaning search | `uv run mynotes search "budget" --raw` |
| `--json` | Print the results as JSON | `uv run mynotes search "budget" --json` |

---

## 7. Filters: narrowing a question or a search

You can narrow `ask` and `search` in two equivalent ways: with options, or by writing the filter inside the question itself (*inline*). Inline filters also work in the web interface and in Claude.

| Filter | Option | Inline | Meaning |
|---|---|---|---|
| Tag | `--tag clienta` | `#clienta` | Only documents with that tag |
| Folder | `--in projects/alfa` | `in:projects/alfa` or `in:"Project Notes/alfa"` | Only documents under that folder (relative to the root). Use quotes inline when the folder name has spaces |
| File type | `--format pdf` | `format:pdf` | Only that file type |
| From a date | `--since 2026-01` | `since:2026-01` | Only documents dated on or after it |
| Up to a date | — | `until:2026-06-30` | Only documents dated on or before it |
| Include history | `--history` | `history:` | Also consider older, replaced versions |
| As of a date | `--as-of 2025-06` (ask only) | `as-of:2025-06` | Answer as things stood on that date: only documents dated on or before it are used, older versions included |

Dates are written `YYYY-MM` or `YYYY-MM-DD`; a space after the colon is fine (`as-of: 2025-06`). Filters combine: all of them must match.

Examples:

```bash
uv run mynotes ask "What is the delivery date? #clienta in:projects/alfa"
uv run mynotes search "meeting notes format:docx since:2026-03"
uv run mynotes ask "Who was responsible for the pilot? as-of:2025-09"
```

**History is detected automatically.** Questions containing words such as *previous*, *earlier*, *history*, *changed*, *used to*, *antes*, *anterior*, *versão anterior*, *histórico*, *mudou* or *alterações* include older versions without `--history`:

```bash
uv run mynotes ask "What was the previous budget for Alfa?"
```

---

## 8. The web interface

The web interface offers the same features in your browser, plus review screens and a visual graph explorer. This section is a summary: the [web interface manual](web-interface-manual.md) explains every page in detail.

**Start it:**

```bash
cd ~/mynotes
uv run mynotes serve
```

Then open **http://127.0.0.1:8765/** in your browser. Leave the terminal open; press Ctrl+C there to stop the web interface. The pages work on a phone as well, and the switch at the top right chooses a light or dark theme or follows your system. To use another port:

```bash
uv run mynotes serve --port 8800
```

### Pages

| Page | Address | What you can do |
|---|---|---|
| **Chat** | http://127.0.0.1:8765/ui/chat | Ask questions; the answer appears as it is written. Click a citation to open the document; PDF citations open at the cited page |
| **Search** | http://127.0.0.1:8765/ui/search | Search with filters for tags, folder, file type, date and history |
| **History** | http://127.0.0.1:8765/ui/history | Your past chats, newest first; the questions of one chat appear together, with their answers and sources. **Continue this chat** reopens one in Chat to ask more. Delete chats not used for a number of days, or all of them |
| **Document** | reached from search results and citations | Read a document's text, opened at the cited section; see its tags (accept or reject suggested ones), the documents it replaces or that replace it, and its edit history. "Open in app" opens the original file |
| **Reviews** | http://127.0.0.1:8765/ui/reviews | Three tabs: **versions** (confirm, reject or reverse proposed "replaces" links), **tags** (accept, reject or merge suggested tags) and **entities** (merge duplicates in the knowledge graph, or mark them as different) |
| **Graph** | http://127.0.0.1:8765/ui/graph | Look up a person, project or other entity; see its summary, its connections as an interactive diagram, and a timeline |
| **Status** | http://127.0.0.1:8765/ui/status | Scans, queued work, knowledge-graph progress, index health and AI services. **Scan now** starts a scan (the worker must be running to process it). Refreshes every 30 seconds |
| **Help** | http://127.0.0.1:8765/ui/help | Read the manuals inside the app: this one, the web interface manual and the Claude (MCP) guide |

There is also an interactive description of the HTTP API at http://127.0.0.1:8765/docs, for people who want to connect other programs.

**Security:** in the default single-user setup there is no login. The web interface only listens on your own computer (`127.0.0.1`), so nobody else can reach it. Don't change `--host` to make it reachable from other computers unless you have set up authentication ([section 17](#17-shared-team-use)).

---

## 9. Using mynotes from Claude (MCP)

MCP (Model Context Protocol) lets Claude Code, Claude Desktop and other AI assistants use mynotes as a tool: they can search your notes, ask for cited answers, read whole documents and their history, list tags and explore the knowledge graph.

**Claude Code:** run once, replacing the path with your mynotes folder:

```bash
claude mcp add mynotes -- uv run --directory /home/me/mynotes mynotes mcp
claude mcp list
```

`claude mcp list` should show `mynotes` as connected. Then ask Claude things like "Search my notes for the Alfa budget".

**Claude Desktop:** see [docs/mcp.md](mcp.md) for the configuration file entry.

Claude starts and stops mynotes's MCP server by itself; you don't run `uv run mynotes mcp` yourself. The databases must be running, and the worker too if you want the index to stay current.

---

## 10. Keeping the index up to date

The worker is what keeps mynotes in step with your folders. While it runs:

- it **scans every 15 minutes** (by default) and processes only files that are new, changed, moved or deleted;
- it runs a **full check every night at 03:00**, re-reading every file to catch anything a quick scan could miss;
- it runs **housekeeping every night at 04:30** (see [Maintenance](#15-maintenance-backups-and-updates));
- optionally, it **watches your folders** and reacts within seconds of a change (see below).

New or edited files are picked up once they have stayed unchanged for 10 seconds, so files still being saved or copied aren't read half-written.

### Start the worker and the web interface automatically

Instead of keeping terminals open, install the worker, the web interface, or both as services that start when you log in and restart if they crash:

```bash
cd ~/mynotes
uv run mynotes service install                      # the worker (default)
uv run mynotes service install --component web      # the web interface (http://127.0.0.1:8765)
uv run mynotes service install --component all      # both
```

This uses systemd on Linux (user units `mynotes-worker` and `mynotes-web`), launchd on macOS and Task Scheduler on Windows. To see what it would install without changing anything, add `--dry-run`:

```bash
uv run mynotes service install --dry-run --component all
```

Related commands (`--component worker|web|all`, worker by default):

```bash
uv run mynotes service status --component all   # installed? running?
uv run mynotes service logs                     # follow the worker's log (on Linux it prints the journalctl command to run)
uv run mynotes service logs --component web     # the web interface's log
uv run mynotes service uninstall --component all
```

Logs: Linux `journalctl --user -u mynotes-worker -f` and `journalctl --user -u mynotes-web -f`; macOS `~/Library/Logs/mynotes/worker.log` and `web.log`; Windows `%LOCALAPPDATA%\mynotes\mynotes\Logs\worker.log` and `web.log`.

On Linux the services run while you are logged in; to keep them running without a session (a headless machine), run `loginctl enable-linger $USER` once. The web service listens on localhost only; to reach it from other machines, see [Shared (team) use](#17-shared-team-use).

Don't run `uv run mynotes worker` or `uv run mynotes serve` in a terminal while the matching service is running; one of each is enough.

The databases need no service of their own: Docker restarts the containers unless you stopped them, as long as Docker itself starts at boot (Docker Desktop does; on Linux `sudo systemctl enable docker`).

### Scan right now

You don't normally need this, but you can start a scan at any time:

```bash
uv run mynotes scan                          # all roots
uv run mynotes scan --root ideas             # only the root named "ideas"
uv run mynotes scan --full                   # re-read every file, not only changed ones
uv run mynotes scan --root ideas --paths "projects/alfa/plan.md" --paths "projects/beta"
```

`--paths` scans only the given files or folders (relative to the root) and can be repeated. A scan records what changed; the worker then does the actual reading, so it must be running.

### React to changes within seconds (file watcher)

The watcher is off by default. Turn it on in `mynotes.yaml`, for all roots:

```yaml
scan:
  watch:
    enabled: true
```

or for one root:

```yaml
roots:
  - name: ideas
    path: ~/Documents/ideas
    watch: true
```

Then restart the worker (or the service: `uv run mynotes service uninstall` followed by `uv run mynotes service install`). Check that it works:

```bash
uv run mynotes watch status
```

Each watched root shows `native` (the operating system reports changes) or `polling` (mynotes checks every 2 seconds; used automatically for network drives and inside containers). On Linux, very large folder trees can exceed the system's watch limit; mynotes then switches to polling by itself.

---

## 11. Tags

Tags group documents by topic, client, project or anything else, and let you filter questions and searches.

### Where tags come from

| Source | Example | Status |
|---|---|---|
| **Folder names** | A file in `projects/alfa/` gets the tags `folder:projects` and `folder:alfa` (folders named `misc` and `tmp` are skipped). Rules in `tags.folder_rules` can map folders to other tags, such as `project:alfa` | Accepted |
| **Frontmatter**: the `tags:` or `keywords:` list at the top of a Markdown file | `tags: [clienta, proposal]` | Accepted |
| **Hashtags** in the text | `#clienta` | Accepted |
| **LLM suggestions** | The LLM reads the document and proposes tags with a confidence score | Accepted automatically at 85% confidence or more; otherwise *proposed*, waiting for your review |
| **Manual**: tags you accept, merge or add during review | | Accepted |

Tag names are normalised: lower case, no accents, spaces become `-`. So `#Cliente A` and `#cliente-a` are the same tag. A tag with a prefix gets a *kind*: `project:alfa`, `status:draft`, `topic:ai`, `folder:…`. Others have the kind `other`.

### Commands

**List all tags**, with how many documents use each and where they came from:

```bash
uv run mynotes tags list
uv run mynotes tags list --prefix project:     # only tags starting with "project:"
uv run mynotes tags list --source llm          # only tags from one source: folder, frontmatter, hashtag, llm or manual
uv run mynotes tags list --json
```

**Show one document's tags**, with their source and status. The path is relative to its root, as shown by `search`:

```bash
uv run mynotes tags show "projects/alfa/plan.md"
```

**Review the LLM's suggestions** one by one, oldest document first:

```bash
uv run mynotes tags review
```

For each suggestion you see the tag, its confidence and the LLM's reason, then type a letter and press Enter:

| Key | Action |
|---|---|
| `a` | Accept the tag |
| `r` | Reject it (it won't be suggested again for that document) |
| `m` | Merge: use an existing tag instead (you are asked which) |
| `s` or Enter | Skip for now |
| `q` | Quit the review |

**Accept or reject one suggestion directly**, without the interactive review:

```bash
uv run mynotes tags accept "projects/alfa/plan.md" machine-learning
uv run mynotes tags reject "projects/alfa/plan.md" misc-notes
```

**Merge two tags** (for example a misspelling into the right one). Every document moves to the second tag, and the first becomes an alias of it, so future occurrences map to the second automatically:

```bash
uv run mynotes tags merge clinte-a cliente-a
```

You can do all of this in the web interface too (Reviews page, tags tab, and on each document page).

---

## 12. Versions and successors

mynotes tracks two kinds of "newer version".

### Editing a file (snapshots)

When you edit and save a file, mynotes keeps a **snapshot** of the earlier content. By default it keeps the latest 5 snapshots per document, and edits less than 24 hours apart count as one snapshot. The first version is always kept. See a document's history:

```bash
uv run mynotes versions history "projects/alfa/plan.md"
```

### A new file replacing an older one (successors)

Often a new file replaces an old one: `plan-v2.md` next to `plan.md`, or the old file moved into an `archive` folder. mynotes detects these and records that the new document **supersedes** (replaces) the old one. It looks for:

- files with the same name apart from version markers (`v2`, `final`, dates…), or the same title, in the same folder, in an archive folder next to it (`archive`, `old`, `versions`, `arquivo`), or sharing a `project:` tag;
- content similar enough to be the same document.

Very likely matches are linked automatically; likely ones are **proposed** for you to review. A renamed file whose content also changed slightly is recognised as the same document.

**Once a link is active, the older document is *superseded*:** questions use the newer one, and the older one is only used for history questions (see [Filters](#7-filters-narrowing-a-question-or-a-search)).

You can also state it yourself at the top of a Markdown file (*frontmatter*). These links always win:

```markdown
---
supersedes: plan.md
---
```

`superseded_by: plan-v3.md` works the other way round. `status: obsolete` (or `archived`, `superseded`) marks a document as no longer current. Paths are relative to the document's folder or to the root, or a document title.

### Commands

**List the links** between documents:

```bash
uv run mynotes versions list
uv run mynotes versions list --status proposed     # proposed, auto, confirmed or rejected
uv run mynotes versions list --json
```

Each link has a number (its id), used by the commands below.

**Review proposed links** one by one:

```bash
uv run mynotes versions review
```

For each proposal you see the newer and the older document with a preview, then choose:

| Key | Action |
|---|---|
| `c` | Confirm: the older document becomes superseded |
| `r` | Reject: they aren't versions of each other; the older one stays current |
| `s` | Swap: the direction is wrong; reverse it and confirm |
| `k` or Enter | Skip for now |
| `q` | Quit the review |

**Act on one link directly**, by its id:

```bash
uv run mynotes versions confirm 12
uv run mynotes versions reject 12
uv run mynotes versions swap 12
```

**Add a link mynotes didn't find** (newer document first):

```bash
uv run mynotes versions add "projects/alfa/plan-2026.md" "projects/alfa/plan.md"
```

---

## 13. The knowledge graph

The knowledge graph is a map of the **entities** your notes mention and the **relations** between them ("Joana works on Alfa", "Alfa uses Kubernetes"), each backed by the documents that say so. It is built by the LLM in the background, after a document is indexed.

Entity types: `Project`, `Idea`, `Person`, `Organization`, `Technology`, `Concept`, `Decision`, `Requirement`, `Milestone`, `Risk` and `Metric`.

**Follow the progress:**

```bash
uv run mynotes graph extract-status
uv run mynotes graph extract-status --json
```

It shows how many document sections are done, cached and waiting, and an estimate of the time left once the worker has made some LLM calls.

**List entities**, most mentioned first:

```bash
uv run mynotes graph entities
uv run mynotes graph entities --type Person --limit 20
```

**Show one entity's card**: its type, other names, a summary, current facts and relations. If the name matches several entities, mynotes lists them:

```bash
uv run mynotes graph entity "Joana Ribeiro"
uv run mynotes graph entity "Alfa" --type Project
uv run mynotes graph entity "Alfa" --json
uv run mynotes graph entity "#519"            # one entity by the id shown in the list
```

A name that was merged into another entity opens the entity it was merged into.

**Show what an entity is connected to** (up to 2 steps away by default):

```bash
uv run mynotes graph neighbors "Projeto Alfa"
uv run mynotes graph neighbors "Projeto Alfa" --hops 1
```

**Show a timeline** for an entity, or for all documents of a project tag:

```bash
uv run mynotes graph timeline "Projeto Alfa"
uv run mynotes graph timeline "project:alfa"
```

**Show a document's direct connections**: its folder, tags, the documents it links to and is linked from, and its versions:

```bash
uv run mynotes graph doc "projects/alfa/plan.md"
```

**Fix duplicates.** If the same thing appears as two entities, merge them (the first name survives). A merge can be undone:

```bash
uv run mynotes graph merge "Projeto Alfa" "Alfa project"
uv run mynotes graph unmerge "Alfa project"
```

Entities that differ only by a generic word ("Kanbanist plugin" and "Kanbanist", "ferramenta Tagloom" and "Tagloom") are merged automatically by the nightly housekeeping; to do it now, or to see what would be merged:

```bash
uv run mynotes graph merge-variants --dry-run
uv run mynotes graph merge-variants
```

**Review the most connected entities.** Very connected entities ("hubs") are usually generic words or containers ("User", "Season 1") that add noise to graph answers; list them to decide what to merge or ignore:

```bash
uv run mynotes graph hubs
uv run mynotes graph hubs --type Concept --limit 10
```

Bare dates ("2023-05-20") are never entities: a note saying "X is scheduled for 2023-05-20" gives X a `date` attribute instead. Older graphs that still have date entities are cleaned by the nightly housekeeping, or right away with:

```bash
uv run mynotes graph fold-dates --dry-run
uv run mynotes graph fold-dates
```

**Link names the LLM missed.** Short notes and lists often name a tool or a person that the extractor did not record ("assenta na ferramenta Stanza"). mynotes adds those mentions by looking for the names of entities it already knows, whole words only, skipping common words and very frequent names. This happens after every extraction and in the nightly housekeeping; to run it now over all notes, or to see how many mentions it would add:

```bash
uv run mynotes graph link-mentions --dry-run
uv run mynotes graph link-mentions
```

It can be turned off with `graph.lexical_mentions: false`.

**Fill in missing extractions**, for example after an interruption. Only documents without results are queued; the worker processes them:

```bash
uv run mynotes graph backfill
uv run mynotes graph backfill --limit 100
uv run mynotes graph backfill --retry-errors   # also the passages whose extraction failed
```

A passage whose extraction failed (the worker log says `failed after the repair retry`) is
remembered as failed and not tried again on its own. `--retry-errors` forgets those failures first,
so their documents are queued again; use it after an update that improves the extractor.

By default (`retrieval.graph_expansion: auto`) answers use the graph only for questions that reach their subject through another note ("who maintains the plugin that the note 'X' uses?"), where it finds the second hop; for direct questions the graph only added noise in tests. `true` uses it for every question, `false` never. The graph also feeds the graph explorer, entity cards, timelines and Claude's graph tools.

---

## 14. Configuration

### Where settings live

| File | Contents | Location |
|---|---|---|
| `.env` | Secrets: passwords, AI service addresses and keys | The mynotes folder (`~/mynotes/.env`) |
| `mynotes.yaml` | Everything else: roots, schedule, watcher, tags, versions, search | See below |

mynotes looks for `mynotes.yaml` in this order and uses the first it finds:

1. the path in the `MYNOTES_CONFIG` variable (set in `.env` or in the environment);
2. `mynotes.yaml` in the folder you run the command from;
3. your user configuration folder: Linux `~/.config/mynotes/mynotes.yaml`, macOS `~/Library/Application Support/mynotes/mynotes.yaml`, Windows `%LOCALAPPDATA%\mynotes\mynotes\mynotes.yaml`.

`uv run mynotes status` shows which file is in use. Values in `.env` override the matching settings in `mynotes.yaml`.

**After changing `mynotes.yaml`**, restart the worker (and the web interface, if running). After adding, removing or renaming a root, also run `uv run mynotes init`.

### Common settings

All settings are listed, with comments, in `mynotes.example.yaml`. The ones you are most likely to change:

| Setting | Default | What it does |
|---|---|---|
| `roots` | — | The folders to read ([Step 4](#step-4-create-the-settings-file-mynotesyaml-and-choose-your-notes-folders)) |
| `scan.schedule.mode` | `interval` | `interval` (every N minutes), `cron` (a schedule such as `"*/30 8-20 * * 1-5"`: every 30 minutes, 08:00–20:00, Monday to Friday) or `manual` (only when you run `scan`) |
| `scan.schedule.interval_minutes` | `15` | Minutes between scans in `interval` mode |
| `scan.schedule.full_scan_cron` | `"0 3 * * *"` | When the nightly full check runs; `null` turns it off |
| `scan.watch.enabled` | `false` | React to changes within seconds ([section 10](#react-to-changes-within-seconds-file-watcher)) |
| `scan.ignore` | system and temporary files | File and folder patterns to skip, in `.gitignore` style. You can also put a `.mynotesignore` file in any folder |
| `formats` | all supported | File types to read, e.g. `[md, pdf]` |
| `extract.pdf_ocr` | `false` | Read scanned PDFs with OCR (needs `tesseract` and `ocrmypdf` installed) |
| `tags.folder_segments_as_tags` | `true` | Turn folder names into tags |
| `tags.ignore_segments` | `["misc", "tmp"]` | Folder names that don't become tags |
| `tags.llm_autotag` | `true` | Let the LLM suggest tags |
| `tags.llm_auto_accept` | `0.85` | Confidence from which suggestions are accepted without review |
| `versions.max_per_document` | `5` | Snapshots kept per document (`null` = all) |
| `versions.index_old_versions` | `false` | Make old snapshots searchable for history questions |
| `versions.archive_folders` | `[archive, old, versions, arquivo]` | Folder names that mean "older versions live here" |
| `retention.deleted_days` | `30` | Days before a deleted file is forgotten completely |
| `graph.lexical_mentions` | `true` | Add mentions of known entities whose name appears verbatim in a note (whole words; common and very frequent names are skipped, tuned by the other `graph.lexical_*` keys in `mynotes.example.yaml`) |
| `retrieval.graph_expansion` | `auto` | Use the knowledge graph when answering: `auto` = only for questions that go through another note, `true` = always, `false` = never |
| `retrieval.graph_freshness` | `false` | With graph expansion: count recently stated relations more than old ones (half-life `retrieval.graph_freshness_half_life_days`, 365 days) |
| `retrieval.followup_turns` | `3` | In the web Chat, follow-up questions use the last N questions and answers of the chat: they are rewritten into a complete question before searching, and the answer sees those turns (each earlier answer cut to 1,200 characters: roughly 300 tokens per turn, about 1,000 for the default 3). Range 0-10; `0` answers every question on its own |
| `retrieval.title_lock` | `true` | A quoted note title in a question puts that note first |
| `retrieval.date_lock` | `true` | A date written in a question puts the notes named or dated that day first |
| `retrieval.change_questions` | `true` | "When did X change to Y" questions look through the whole series and answer with the earliest note stating Y |
| `retrieval.recency_boost` | `true` | Mildly favour recent documents when ranking results: a document `retrieval.recency_half_life_days` old (365) loses half of the boost range, and none ever drops below `retrieval.recency_min_multiplier` (80%) of its score. Not applied to questions about the past |

### Removing a root

To stop mynotes reading a folder, delete its entry from `roots:` in `mynotes.yaml`, then run:

```bash
cd ~/mynotes
uv run mynotes init
```

`init` reports the root as **disabled**. From then on:

- it is no longer scanned or watched;
- its documents disappear from questions and searches (once the worker has processed the change);
- everything mynotes knew about it stays in the database: tags you reviewed, version links, snapshots. The knowledge graph also keeps the entities it learned from those documents.

To bring it back, add the entry again (with the same `name`) and run `uv run mynotes init`: its documents return to search without being read or embedded again.

Renaming a root (changing its `name`) counts as removing the old root and adding a new one, so its files are indexed again from the start. Restart the worker after any of these changes.

---

## 15. Maintenance, backups and updates

### Automatic housekeeping

Every night at 04:30 the worker removes files deleted more than 30 days ago, clears unused cache entries, deletes old finished jobs and merges knowledge-graph entities that differ only by a generic word. To run it now, or to see what it would remove:

```bash
uv run mynotes maintenance run --dry-run    # report only
uv run mynotes maintenance run
```

### Retrying failed extractions

A file whose extraction failed (the status page counts it under *failed*) is not retried on its
own: the scanner only queues new and changed files. After an update that fixes an extractor, queue
those files again:

```bash
uv run mynotes maintenance retry-extract --dry-run    # how many would be retried
uv run mynotes maintenance retry-extract
```

The worker then extracts them in the background. In server mode run it through the worker
container: `docker compose exec worker mynotes maintenance retry-extract`.

### Backups

```bash
cd ~/mynotes
uv run mynotes backup
uv run mynotes backup --out ~/Backups/mynotes
```

This saves the database as a `.dump` file named with the date and time (for example `mynotes_20261001T043000Z.dump`) and a copy of `mynotes.yaml`. By default the files go to `~/.local/share/mynotes/backups` (Linux), `~/Library/Application Support/mynotes/backups` (macOS) or `%LOCALAPPDATA%\mynotes\mynotes\backups` (Windows). **`.env` is never included**: keep a copy of it somewhere safe yourself. The search index isn't backed up because it can be rebuilt from the database.

**Restore** a backup (this replaces the current database, then rebuilds both search indexes):

```bash
uv run mynotes restore ~/Backups/mynotes/mynotes_20261001T043000Z.dump
```

It asks for confirmation; add `--yes` to skip the question. Stop the worker before restoring.

### Search index tools

The database is the source of truth; the search index is a copy built from it. If search results look wrong or incomplete:

```bash
uv run mynotes index reconcile             # compare and queue repairs for any differences
uv run mynotes index reconcile --no-fix    # compare and report only
uv run mynotes index sync                  # send queued changes to the index now
uv run mynotes index rebuild               # rebuild the search index from the database
uv run mynotes index rebuild --entities    # rebuild the knowledge-graph entity index
```

`rebuild` builds a new index alongside the old one and switches over when it is complete, so search keeps working meanwhile. It asks for confirmation (`--yes` to skip). `--reembed` also recalculates every embedding, which is only needed after changing the embedding model, and calls the embedding service for every passage.

### Updating mynotes

```bash
cd ~/mynotes
git pull
uv sync
uv run mynotes init
```

Then restart the worker (Ctrl+C and start it again, or `uv run mynotes service uninstall` followed by `uv run mynotes service install`) and the web interface.

### Stopping everything

```bash
cd ~/mynotes
uv run mynotes service uninstall    # if you installed the service
docker compose stop                 # databases; your data is kept
```

`docker compose down` also removes the containers but keeps the data. **`docker compose down -v` deletes all data** (database and index): you would have to index your notes again.

### Uninstalling

1. `uv run mynotes service uninstall`
2. `cd ~/mynotes && docker compose down -v` (deletes the databases)
3. Delete the `mynotes` folder and your `mynotes.yaml`.

Your notes are never touched.

---

## 16. Troubleshooting

| Problem | What to do |
|---|---|
| `Missing command.` | You ran a group such as `uv run mynotes tags` on its own. Add the action: `uv run mynotes tags list`. See `uv run mynotes tags --help` |
| Errors about passwords, `MYNOTES_PG_DSN` or connections | Make sure you are in the mynotes folder (`cd ~/mynotes`), that `.env` exists there, and that the databases run (`docker compose ps`) |
| `docker compose ps` shows nothing or `unhealthy` | Start them: `docker compose up -d`. If OpenSearch keeps failing, look at `docker compose logs opensearch`: often not enough memory (Docker Desktop needs at least 6 GB) or disk space (OpenSearch needs about 20 GB free) |
| `check-endpoints` fails | Check the addresses and keys in `.env`, and your network or VPN connection |
| A new or edited file doesn't show up | The worker must be running (`uv run mynotes service status`). It picks changes up at the next scan (15 minutes by default) or within seconds with the watcher. Force it with `uv run mynotes scan` |
| A file is never indexed | Check its type is supported and that it isn't matched by `scan.ignore` or a `.mynotesignore`. The "Unsupported files" panel of `uv run mynotes status` lists files mynotes skipped. Online-only cloud files (iCloud, OneDrive) are skipped until downloaded |
| Answers use an outdated document | Check whether a successor link exists: `uv run mynotes versions list`. Add one with `uv run mynotes versions add NEWER OLDER`, or confirm a proposal |
| The answer says the notes don't contain it, but they do | Try `search` with other words to see what is found; check your filters; rephrase. Very new files may still be processing (`uv run mynotes status`) |
| Search results look stale or incomplete | `uv run mynotes index reconcile`, then `uv run mynotes index sync` |
| `open "[1]"` says there are no citations | It uses the citations of your most recent `ask`. Run `ask` first |
| The web page doesn't load | `uv run mynotes serve` must be running in a terminal; use http://127.0.0.1:8765/ |
| Port 8765 or 9200 already in use | Another program uses it. Use `uv run mynotes serve --port 8800`, or change `MYNOTES_COMPOSE_OS_PORT` / `MYNOTES_COMPOSE_PG_PORT` in `.env` (and the matching address) |
| The knowledge graph is empty or slow | It is built after indexing, by the LLM, at about 10 documents per minute. Check `uv run mynotes graph extract-status` |
| More detail | Add `-v` before the command for debug logs: `uv run mynotes -v scan` |

---

## 17. Shared (team) use: a server and agents

mynotes can also run as one **server** for several people. The server (PostgreSQL, OpenSearch, the worker and the web interface, in Docker) never sees a notes folder: on each person's machine a small **agent** watches their folder and uploads the files, and the server keeps a copy, indexes it and shows each person only their own notes. People sign in to the web interface with a personal token. The setup is described in [deployment/server/README.md](../deployment/server/README.md); in short, the administrator runs on the server:

```bash
mynotes token create --user alice --name laptop --root alice-notes   # her agent's token; creates the root
mynotes token create --user alice --name web --web                   # her sign-in token
mynotes token create --user admin --name me --admin                  # your own
mynotes root list                                          # roots and when each agent was last seen
mynotes token list / mynotes token revoke <id>
```

and alice runs on her machine, with the server's address, her agent token, the root's name and her folder in `MYNOTES_AGENT_SERVER_URL`, `MYNOTES_AGENT_TOKEN`, `MYNOTES_AGENT_ROOT` and `MYNOTES_AGENT_PATH`:

```bash
uv run mynotes agent run        # one full sync, then it watches the folder; Ctrl+C stops it
uv run mynotes agent sync       # one sync, then exit
uv run mynotes agent status     # what the server knows about this root
```

On the server the notes of a remote root appear under the root's name (`alice-notes/ideas/plan.md`), and alice's read access is granted under that prefix when the root is created. **Open in app** is not offered for uploaded notes; **Open** and **Download** serve the server's copy.

The older building blocks remain: sign-in through an OpenID Connect provider (`auth.mode: oidc`), and per-folder read permissions for users and groups:

```bash
uv run mynotes acl grant alice "projects/alfa"
uv run mynotes acl grant "group:team-a" "projects"
uv run mynotes acl grant admin ""              # "" = everything
uv run mynotes acl revoke alice "projects/alfa"
uv run mynotes acl list
```

OIDC needs an identity provider and a sign-in proxy in front of the web interface; the token sign-in of server mode needs neither. To run everything in Docker, see [deployment/README.md](../deployment/README.md).

---

## 18. Command reference

Every command starts with `uv run mynotes` and is run from the mynotes folder. Add `--help` to any of them for details.

### Setup and status

| Command | What it does |
|---|---|
| `uv run mynotes --version` | Print the mynotes version |
| `uv run mynotes init` | Create or upgrade the database tables, register roots, create the search indexes. `--no-migrate` skips the table upgrade |
| `uv run mynotes check-endpoints` | Test the embedding, reranker and LLM services. `--no-llm` skips the LLM |
| `uv run mynotes status` | Show the state of everything. `--json` |

### Asking and searching

| Command | What it does |
|---|---|
| `uv run mynotes ask "QUESTION"` | Answer with citations. Options: `--tag`, `--in`, `--format`, `--since`, `--history`, `--as-of`, `--no-stream`, `--json` |
| `uv run mynotes search "TEXT"` | Ranked passages. Options: `--tag`, `--in`, `--format`, `--since`, `--history`, `-n/--limit`, `--raw`, `--json` |
| `uv run mynotes open "[N]"` | Open citation N of the last answer in its application. Also takes a path, a title or an id |

### Running

| Command | What it does |
|---|---|
| `uv run mynotes worker` | Run the background worker until Ctrl+C. Options: `--concurrency N` (parallel jobs, default 4), `--types extract,chunk` (only some job types), `--no-schedule` (no scheduled scans) |
| `uv run mynotes serve` | Run the web interface and HTTP API until Ctrl+C. Options: `--port N`, `--host ADDRESS` (default `127.0.0.1`), `--reload` (for development) |
| `uv run mynotes mcp` | MCP server for Claude; Claude starts it by itself |
| `uv run mynotes scan` | Scan now. Options: `--root NAME`, `--paths PATH` (repeatable), `--full` |
| `uv run mynotes watch status` | Show the file watcher's state per root |
| `uv run mynotes service install` | Start the worker (or `--component web` / `all`) automatically at login. `--dry-run` shows what would be installed |
| `uv run mynotes service status` | Is the service installed and running? `--component worker|web|all` |
| `uv run mynotes service logs` | Follow the worker's log, or `--component web` |
| `uv run mynotes service uninstall` | Stop and remove the service(s). `--component worker|web|all` |

### Tags

| Command | What it does |
|---|---|
| `uv run mynotes tags list` | All tags with counts. Options: `--prefix TEXT`, `--source folder\|frontmatter\|hashtag\|llm\|manual` (only tags from that source), `--json` |
| `uv run mynotes tags show "PATH"` | One document's tags |
| `uv run mynotes tags review` | Review LLM suggestions interactively |
| `uv run mynotes tags accept "PATH" TAG` | Accept one suggestion |
| `uv run mynotes tags reject "PATH" TAG` | Reject one suggestion |
| `uv run mynotes tags merge FROM INTO` | Merge tag FROM into INTO |

### Versions

| Command | What it does |
|---|---|
| `uv run mynotes versions list` | Links between documents. Options: `--status proposed\|auto\|confirmed\|rejected`, `--json` |
| `uv run mynotes versions review` | Review proposed links interactively |
| `uv run mynotes versions confirm ID` | Confirm a link |
| `uv run mynotes versions reject ID` | Reject a link |
| `uv run mynotes versions swap ID` | Reverse a link's direction and confirm it |
| `uv run mynotes versions add "NEWER" "OLDER"` | Add a link by hand |
| `uv run mynotes versions history "PATH"` | A document's snapshots |

### Knowledge graph

| Command | What it does |
|---|---|
| `uv run mynotes graph extract-status` | Extraction progress. `--json` |
| `uv run mynotes graph backfill` | Queue documents missing extractions. `--limit N`, `--retry-errors` |
| `uv run mynotes graph entities` | List entities. Options: `--type TYPE`, `--limit N` (default 50) |
| `uv run mynotes graph entity "NAME"` | An entity's card. Options: `--type TYPE`, `--json` |
| `uv run mynotes graph neighbors "NAME"` | Connected entities. Options: `--hops N` (default 2), `--json` |
| `uv run mynotes graph timeline "NAME"` | Timeline of an entity, or of `project:NAME`. `--json` |
| `uv run mynotes graph doc "PATH"` | A document's folder, tags, links and versions. `--json` |
| `uv run mynotes graph merge "KEEP" "DROP"` | Merge duplicate entities |
| `uv run mynotes graph merge-variants` | Merge entities that differ only by a generic word ("X plugin" into "X"). `--dry-run` lists them |
| `uv run mynotes graph hubs` | The most connected entities, for review. Options: `--type TYPE`, `--limit N` (default 20), `--json` |
| `uv run mynotes graph fold-dates` | Turn bare-date entities into date attributes of what they date. `--dry-run` counts them |
| `uv run mynotes graph link-mentions` | Add mentions of known entities named verbatim in notes where the LLM missed them. `--dry-run` counts them |
| `uv run mynotes graph unmerge "NAME"` | Undo a merge |

### Maintenance

| Command | What it does |
|---|---|
| `uv run mynotes backup` | Back up the database and settings. `--out FOLDER` |
| `uv run mynotes restore FILE.dump` | Restore a backup and rebuild the indexes. `--yes` skips the confirmation |
| `uv run mynotes index sync` | Send queued changes to the search index now |
| `uv run mynotes index reconcile` | Compare database and index and queue repairs. `--no-fix` only reports |
| `uv run mynotes index rebuild` | Rebuild the search index. Options: `--entities` (entity index instead), `--reembed` (recalculate embeddings), `--yes` |
| `uv run mynotes maintenance run` | Housekeeping now. `--dry-run` only reports |

### Shared use

| Command | What it does |
|---|---|
| `uv run mynotes acl grant USER "FOLDER"` | Let a user (or `group:NAME`) read under a folder (`""` = everything) |
| `uv run mynotes acl revoke USER "FOLDER"` | Remove that permission |
| `uv run mynotes acl list` | List all permissions |
| `uv run mynotes root add NAME --remote --owner USER` | Create a root fed by a user's agent (server mode) |
| `uv run mynotes root list` / `root remove NAME` | List the roots / take a remote root out of the index |
| `uv run mynotes token create --user USER [--root NAME] [--web] [--admin]` | Create a personal token (printed once) |
| `uv run mynotes token list` / `token revoke ID` | List / revoke tokens |
| `uv run mynotes agent run` / `agent sync` / `agent status` | The agent on a person's machine (server mode) |

### Global options

| Option | Effect |
|---|---|
| `uv run mynotes -v COMMAND` | Show detailed (debug) logs |
| `uv run mynotes --install-completion` | Enable Tab completion of commands in your shell |

---

## 19. Glossary

| Term | Meaning |
|---|---|
| **Chunk** | A passage of a document (a few paragraphs). Documents are split into chunks so the relevant part can be found and cited |
| **Citation** | A numbered reference (`[1]`) in an answer, pointing to the chunk it came from |
| **Docker** | Software that runs programs (here, the databases) in isolated containers |
| **Embedding** | A list of numbers representing a text's meaning, so texts with similar meanings can be found even with different words |
| **Entity** | Something the knowledge graph knows about: a project, idea, person, organisation, technology, decision, milestone… |
| **Frontmatter** | A block at the top of a Markdown file, between `---` lines, holding properties such as `tags:` or `supersedes:` |
| **Index** | The search database (OpenSearch) mynotes searches in. It is rebuilt from PostgreSQL when needed |
| **Knowledge graph** | The map of entities and the relations between them, extracted from your notes |
| **LLM** | Large language model: the AI that writes answers, suggests tags and builds the knowledge graph |
| **MCP** | Model Context Protocol: lets AI assistants such as Claude use mynotes as a tool |
| **Reranker** | An AI model that re-sorts search results by how well they answer the question |
| **Root** | A folder you told mynotes to read (with all its subfolders), set in `mynotes.yaml` |
| **Scan** | A pass over your roots to find new, changed, moved and deleted files |
| **Snapshot** | A saved earlier state of a document you edited |
| **Superseded** | Replaced by a newer document; only used for history questions |
| **Tag** | A label grouping documents; used to filter |
| **Worker** | The background program that scans, reads, indexes and builds the knowledge graph |
