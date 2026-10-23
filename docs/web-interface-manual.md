# mynotes web interface manual

This manual explains the mynotes web interface: the pages you open in your browser to ask questions about your notes, search them, read documents, review mynotes's suggestions, explore the knowledge graph and check that everything is working. It assumes no prior knowledge of mynotes.

For installation, the command line and configuration, see the main [user manual](user-manual.md).

## Contents

1. [What the web interface is](#1-what-the-web-interface-is)
2. [Opening the web interface](#2-opening-the-web-interface)
3. [Finding your way around](#3-finding-your-way-around)
4. [Chat: asking questions](#4-chat-asking-questions)
5. [Search: finding passages](#5-search-finding-passages)
6. [History: past questions](#6-history-past-questions)
7. [The document page](#7-the-document-page)
8. [Reviews: checking mynotes's suggestions](#8-reviews-checking-mynotess-suggestions)
9. [Graph: exploring entities and connections](#9-graph-exploring-entities-and-connections)
10. [Status: is everything working?](#10-status-is-everything-working)
11. [Privacy and security](#11-privacy-and-security)
12. [Troubleshooting](#12-troubleshooting)
13. [Quick reference](#13-quick-reference)

---

## 1. What the web interface is

mynotes reads the documents in the folders you chose (Markdown notes, PDFs, Word, PowerPoint and Excel files, web pages and e-books) and lets you ask questions about them. The web interface does this in your browser:

- **Chat** answers questions in plain language, with numbered references to the documents it used.
- **Search** lists the passages that best match what you type.
- **Document pages** show a document's text, tags and version history.
- **Reviews** let you confirm or correct what mynotes guessed by itself: which documents replace which, which tags fit, which names refer to the same thing.
- **Graph** shows the people, projects, technologies and other things your notes mention, and how they connect.
- **Status** shows whether scanning, indexing and the AI services are working.

Some terms used throughout:

| Term | Meaning |
|---|---|
| **Citation** | A numbered reference such as `[1]` in an answer, pointing to the document and place it came from |
| **Tag** | A label grouping documents (from folder names, from tags written in the documents, or suggested by AI) |
| **Superseded** / **historical** | A document replaced by a newer one. mynotes answers from the newer one, and uses the older one only for questions about history |
| **Snapshot** | A saved earlier state of a document you edited |
| **Entity** | Something the knowledge graph knows about: a project, person, organisation, technology, decision, milestone… |
| **Root** | One of the folders mynotes reads, with all its subfolders |
| **Worker** | The background program that reads new and changed files. The web interface shows its results |

---

## 2. Opening the web interface

The web interface is served by mynotes on your own computer. If someone else set mynotes up for you, they may already have it running: just open the address below.

### Start it

Open a terminal, go to the mynotes folder and run:

```bash
cd ~/mynotes
uv run mynotes serve
```

Leave that terminal open while you use the web interface. To stop it, press **Ctrl+C** in the terminal.

### Open it

Open this address in your browser (Firefox, Chrome, Edge, Safari):

**http://127.0.0.1:8765/**

It opens the Chat page. Bookmark it for next time.

If the address doesn't work, see [Troubleshooting](#12-troubleshooting). If mynotes was started on another port (for example with `uv run mynotes serve --port 8800`), use that number instead of `8765`. With the all-in-Docker setup ([deployment/full-stack](../deployment/full-stack/README.md)), the web interface runs automatically on the same address.

### What else must be running

- **The databases** must be running (`docker compose up -d` in the mynotes folder), otherwise pages show errors.
- **The worker** must be running for mynotes to notice new or changed notes, and for the **Scan now** button to have an effect. Without it you can still ask about everything already indexed. See [Status](#10-status-is-everything-working) to check.

---

## 3. Finding your way around

Every page has the same menu bar at the top:

| Menu | Page | Use it to |
|---|---|---|
| **Chat** | `/ui/chat` | Ask questions and get answers with sources |
| **Search** | `/ui/search` | Find where something is written, with filters |
| **History** | `/ui/history` | Reread your past questions and answers, and delete old ones |
| **Reviews** | `/ui/reviews` | Confirm or correct mynotes's suggestions |
| **Graph** | `/ui/graph` | Explore entities and their connections |
| **Status** | `/ui/status` | Check scanning, the queue, the search index and the AI services |
| **Help** | `/ui/help` | Read this manual, the command-line manual and the Claude (MCP) guide inside the app |

Document pages have no menu entry: you reach them from search results, citations, reviews and the graph.

On a phone the menu folds behind a **Menu** button at the top. The switch at the right end of the bar chooses the colour theme: **System** follows your device's light or dark setting, **Light** and **Dark** force one. The choice is remembered in that browser. Keyboard users can press Tab once on any page to reach a "Skip to content" link.

---

## 4. Chat: asking questions

### Ask a question

1. Click **Chat** in the menu bar.
2. Type your question in the box at the bottom, for example *What is the deadline for the Alfa project?*
3. Click **Ask**.

Pressing **Enter** sends the question; **Shift+Enter** adds a new line. The question box stays at the bottom of the page while you scroll through earlier answers. Its options (old note versions, tags) sit under **Options** next to the **Ask** button; on a phone they start folded.

The answer appears word by word as it is written. While it is being written, the **Ask** button is greyed out.

### Reading the answer

Each question you ask appears as a block with three parts:

1. **Your question.**
2. **The answer.** Numbers in square brackets, such as `[1]` or `[2]`, are citations: click one to jump to that source in the list below.
3. **The sources**, numbered to match the citations. Each line shows:
   - the number, such as `[1]`;
   - the document's title and its path inside your notes folder;
   - the page (`p. 4`) for PDFs, or the slide (`slide 7`) for presentations;
   - the document's date, when known;
   - a **historical** label when the source is an older, replaced version;
   - links to open the document (see the next table).

What the links do depends on the type of file:

| File type | Links | What they do |
|---|---|---|
| PDF | **Open** | Opens the PDF in a new browser tab, at the cited page |
| Markdown, text | **View note** | Opens the [document page](#7-the-document-page) at the cited section, in a new window |
| Web page (HTML) | **Open** | Opens the original page in a new tab (with its scripts disabled, for safety) |
| Word, PowerPoint, Excel, e-book | **Open in app** | Opens the original file in its usual application on the computer running mynotes |
| | **Download** | Downloads the original file |
| | **View text** | Opens the document page at the cited section, in a new window, showing the text mynotes extracted |

If your notes don't contain the answer, mynotes says so in its answer, and a line under it may read *No relevant notes were found for this question.* If something went wrong, the line reads *The request failed. Please try again.*

### Answer language and spelling

- Ask in **Portuguese or English**: the answer comes in the language of your question, whatever the language of the documents.
- **Upper/lower case and accents don't matter**: *prazo*, *Prazo* and *PRAZO* find the same documents, and so do *informação* and *informacao*.

### Each question stands alone

mynotes doesn't remember earlier questions. Earlier questions and answers stay on the page for reference, but a follow-up such as *And who leads it?* doesn't know what *it* is. Write each question in full: *Who leads the Alfa project?*

Reloading the page clears the conversation.

### Narrowing a question

Under the question box:

- **Include old notes versions**: ticked by default, so answers can use older, replaced versions of documents. Untick it for questions about the current state of things only. Your choice is remembered in this browser (see below).
- Press **Enter** to ask; **Shift+Enter** adds a new line.
- **Follow-up questions** work: after "O que é o Portainer?" you can ask "e qual é a porta?". mynotes rewrites a follow-up into a complete question before searching your notes, and shows it under yours in italics, for example *Searched as: Qual é a porta do Portainer?*. If that line misread you, ask again in full. The answer also sees the last three questions and answers of the chat (each earlier answer cut to about 1,200 characters, so at most about 1,000 tokens in all); anything said more than three questions ago is forgotten. A chat ends when you reload or leave the page; **Continue this chat** in History picks it up again.
- Answers are formatted as they arrive: headings, lists, tables, code blocks, bold and italic text. `[1]` markers link to the numbered sources below the answer. Code blocks have a **Copy** button in the corner that puts the block's text on the clipboard.
- **tags, comma-separated**: only use documents with these tags, for example `clienta, proposal`.

You can also write filters inside the question itself:

| Write in the question | Effect | Example |
|---|---|---|
| `#tag` | Only documents with that tag | *What is the budget? #clienta* |
| `in:folder` | Only documents under that folder (use quotes if the name has spaces) | *Who is responsible? in:projects/alfa* or *in:"Project Notes/alfa"* |
| `format:type` | Only one file type: `md`, `txt`, `pdf`, `docx`, `pptx`, `xlsx`, `html`, `epub` | *What does the contract say about payment? format:pdf* |
| `since:date` | Only documents dated on or after it (`YYYY-MM` or `YYYY-MM-DD`) | *What did we decide? since:2026-01* |
| `until:date` | Only documents dated on or before it | *What was planned? until:2025-12-31* |
| `history:` | Include older versions (same as the checkbox) | *How did the plan change? history:* |
| `as-of:date` | Answer as things stood on that date | *Who led the project? as-of:2025-06* |

Questions containing words such as *previous*, *earlier*, *history*, *changed*, *used to*, *antes*, *anterior*, *histórico*, *mudou* or *alterações* include older versions automatically.

### Tips for good questions

- Be specific: *What is the delivery date for the Alfa pilot?* works better than *Dates?*
- Name the project, client or document when you know it.
- If an answer seems out of date, check whether the sources carry the **historical** label, or ask with the history option to see how things changed.
- To see everything written about a topic rather than one answer, use [Search](#5-search-finding-passages).

---

## 5. Search: finding passages

Search shows the passages that best match your words, ranked, without writing an answer. Use it to find where something is written.

### Search

1. Click **Search** in the menu bar.
2. Type words or a question in the search box. Inline filters (`#tag`, `in:folder`, and the others listed [above](#narrowing-a-question)) work here too.
3. Press **Enter** or click **Search**. *Searching…* appears while it works.

### Filters

All filters are optional and combine: a result must match all of them.

| Filter | How to use it |
|---|---|
| **Folder** | Type the start of a folder path, relative to your notes folder, for example `projects/alfa` |
| **Since** | Pick a month: only documents dated from that month on |
| **Formats** | Tick file types to search only those. None ticked = all types |
| **include notes history** | Also search older, replaced versions |
| **rerank** | On by default: an AI model re-sorts the results by relevance. Untick it for slightly faster but less precise results |
| **Tags** | At the bottom of the form (see below) |

Once you have searched, the results update by themselves whenever you change a filter. After editing **Folder**, click elsewhere or press **Search**.

#### Choosing tags

The **Tags** section at the bottom lists the tags that are in use, most used first, each with the number of documents that have it.

- **Click a tag** to select it, and click it again to deselect it. Selected tags are outlined and in bold.
- **Select several** to find documents that have **all** of them.
- **Type in the "Filter tags…" box** to show only the tags whose name contains what you typed (case and accents don't matter). Selected tags stay visible. Press **Escape** to empty the box. If nothing matches, it says *No tags match.*
- The section title shows how many tags are selected, and a **Clear** button deselects them all at once.

*No tags yet.* means none of your documents has a tag.

### Results

Up to 20 results are found, best first. With **rerank** ticked (the default), results that score far below the best one (under 5% of its score) are hidden behind a **Show N weaker results** line at the end: a search for a word that only one note contains shows that note, not twenty loosely related ones. Click the line to see the rest. Without rerank, all results are shown. Each result shows:

- an **excerpt** of the matching passage, formatted like chat answers (lists, tables, code blocks, bold and italic text), though cut at about 300 characters, so a long code block or table may end abruptly (a cut code block has no Copy button; open the note for the full text);
- the **section** it belongs to, when the document has headings;
- the **source line**: title, path, page or slide, date, the **historical** label for older versions, and the same open links as in [Chat](#reading-the-answer);
- the document's **tags**.

*No results.* means nothing matched: try other words, fewer filters, or tick **include notes history**.

---

## 6. History: past questions

**History** (`/ui/history`) lists your chats, most recently used first. A chat is everything you asked on the Chat page before reloading or leaving it: two follow-up questions show up as **one** entry, titled by the first question and marked "2 questions" with the time of the first and the last. Click an entry to see every question with its answer, formatted as in Chat, and the sources each answer cited, with the same open links. Only finished answers are kept; a question that failed is not. In shared (team) use, each person sees only their own chats.

**Picking a chat up again.** Open a chat and click **Continue this chat**: the Chat page opens with its earlier questions and answers, and anything you ask next is added to that same chat in History. **Start a new chat** (under the conversation) or opening Chat from the menu starts a fresh one. Chats saved before 7 October 2026 can be continued too.

Follow-up questions understand the chat they belong to, including a chat continued from History: see **Follow-up questions** in [Chat](#4-chat-asking-questions).

**Deleting old chats.** The box at the top deletes:

- chats **not used for N days** (30 by default): a chat is judged by its last question, so one you continued yesterday is kept even if it started a month ago;
- **all** chats.

Click **Delete** and confirm. This cannot be undone. The page shows how many chats were deleted.

The page shows the 200 most recently used chats; older ones are kept until you delete them. Questions asked before chats were grouped (7 October 2026) appear as one-question chats.

---

## 7. The document page

A document page shows one document as mynotes knows it. Open it with **View note** or **View text** (both open in a new window) from a citation or search result, or by clicking a document's title in Reviews or the Graph.

From top to bottom:

### Header

- The **title** (or the path, if the document has no title).
- The **path** inside your notes folder.
- The **status**: *current*; *superseded* (replaced by a newer document); *archived* (marked obsolete in the document itself).
- The document's **date**, its **language**, and its **version number** (for example `v3` after two edits).
- Links to the original file: **Open** (PDF, HTML), **Raw** (Markdown and text, at the top of this page), **Download** or **Open in app**, depending on the file type (see the table in [Chat](#reading-the-answer)).

### Tags

Tags are grouped by where they came from:

| Group | Meaning |
|---|---|
| **folder** | From the folders the file is in |
| **frontmatter** | Written in the properties block at the top of a Markdown file |
| **hashtag** | Written in the text as `#tag` |
| **llm** | Suggested by the AI after reading the document |
| **manual** | Added or chosen by you during a review |

AI suggestions waiting for your decision have **Accept** and **Reject** buttons next to them. Click one and the tags refresh straight away. A rejected tag isn't suggested again for that document.

### Version chain

Shown when mynotes knows this document replaces another, or is replaced by one:

- **Newer:** the documents that replace this one. Answers use those.
- **Older:** the documents this one replaces.

Click a title to go to that document. To correct a wrong link, use the **Versions** tab in [Reviews](#versions-tab) (for proposed links) or the command line (`uv run mynotes versions reject ID`).

### Snapshots

The saved earlier states of the document after edits: the version number, its date, a **latest** label on the current one, and which sections changed.

### Content

The document's text as mynotes extracted it, with headings, lists and tables. Code blocks written between triple backticks (```) have a **Copy** button in their top-right corner. When you arrive from a citation, the page opens at the cited section. For PDFs, Word and other files this is the extracted text, not the original layout: use the open links in the header for the original.

---

## 8. Reviews: checking mynotes's suggestions

mynotes makes three kinds of guesses by itself. Clear cases are applied automatically; uncertain ones wait here for you. Nothing is lost if you leave them: they wait until you decide.

Click **Reviews** in the menu bar, then choose a tab: **Versions** (shown first), **Tags** or **Entities**. Each list refreshes after every decision; the item you decided on disappears.

### Versions tab

Pairs of documents where mynotes thinks one replaces the other (for example `plan-v2.md` and `plan.md`). For each pair you see:

- **newer** and **older**: each document's title (click to open it), path, date and the start of its text, so you can compare;
- the **evidence**: how the pair was found and how confident mynotes is (`confidence` from 0 to 1), how similar the texts are (`jaccard` for wording, `cosine` for meaning, both from 0 to 1), and why it put them in this order (`order_reason`).

Choose one:

| Button | Meaning | Effect |
|---|---|---|
| **Confirm** | Yes, the newer replaces the older | The older becomes *superseded*: answers use the newer one, history questions can still reach the older |
| **Reject** | No, these are different documents | Both stay current; this pair isn't proposed again |
| **Swap** | Yes, but the other way round | The direction is reversed, then confirmed |

*No proposed version relations.* means there is nothing to review.

### Tags tab

AI-suggested tags that weren't certain enough to apply automatically, grouped by document. For each suggestion you see the tag, the AI's confidence (0 to 1) and its reason. Choose one:

| Action | Effect |
|---|---|
| **Accept** | The document gets the tag |
| **Reject** | The tag is dropped and not suggested again for this document |
| **Merge into…** | Type an existing tag name in the box, then click **Merge into…**. The suggested tag is merged into that tag everywhere, and future occurrences of it become the existing tag |

Use **Merge into…** for spelling variants and synonyms, such as merging *machine-learning* into *ml*.

*No proposed tags.* means there is nothing to review.

### Entities tab

Pairs of names in the knowledge graph that may refer to the same thing, such as *Projeto Alfa* and *Alfa project*. Each pair shows both names (click one to open it in the [Graph](#9-graph-exploring-entities-and-connections)), their types and creation dates, which one would be kept (**keep**) and which merged into it (**merge**), and how similar the names are (0 to 1).

| Button | Effect |
|---|---|
| **Merge (keep the older)** | The two become one entity, keeping the older one's name; the other name becomes an alias |
| **Keep separate** | They are recorded as different and not proposed again |

A merge can be undone from the command line: `uv run mynotes graph unmerge "NAME"`.

*No entity merge candidates.* means there is nothing to review.

---

## 9. Graph: exploring entities and connections

The knowledge graph is a map of the entities your notes mention and how they relate ("Joana works on Alfa", "Alfa uses Kubernetes"). It is built in the background after documents are indexed, so it fills up over time.

### Find an entity

1. Click **Graph** in the menu bar.
2. Type a name in the search box. Results appear after you pause typing; exact spelling isn't required.
3. If several entities match (for example a project and a topic with the same name), a list appears: click the one you mean.

You can also link straight to an entity by name, `http://127.0.0.1:8765/ui/graph?entity=Projeto%20Alfa`, or by its id, `...?entity=%23519` (`#519`).

### The entity card

- The **name**, its **type** (Project, Idea, Person, Organization, Technology, Concept, Decision, Requirement, Milestone, Risk or Metric) and, if it isn't active, its status.
- A **summary** written by the AI from what your notes say.
- **Also known as**: other names for the same thing.
- **Facts**: current values such as a status, owner or date, with the date of the document they came from.
- **Mentioned in**: the documents that mention it most. Click one to open its document page.

### The diagram

Below the card, the entity and its connections are drawn as a network:

- each **circle** is an entity, coloured by type; the entity you are looking at has a highlighted border;
- each **line** is a relation, labelled with its kind, such as `works_on`, `uses`, `part_of`, `depends_on`, `owned_by` or `replaces`, or with a fact such as `status=active`; a **dashed** line is a relation that no longer holds (it ended or was replaced);
- **Hops** chooses how far out to look: **1** shows direct connections only, **2** also shows the connections of those connections (busier, but reveals indirect links);
- **click a circle** to re-centre the diagram on that entity and see its connections (the card and timeline above stay on the first entity; search for the new one to see its card);
- **click a line** to see, under the diagram, the documents that support that relation;
- **Expand** makes the diagram fill the whole window; **Restore** (or the Esc key) brings the page back.

Drag circles to untangle the drawing, scroll to zoom, and drag the background to move around.

### Timeline

Under the diagram: the dated facts about the entity, oldest first (relations that started or ended, values that changed). Items that no longer hold carry a status label. *No dated facts for this entity yet.* means the notes don't give dates for it.

---

## 10. Status: is everything working?

The Status page shows the health of mynotes at a glance. It refreshes by itself every 30 seconds.

### Scan now

The **Scan now** button looks through all your folders for new, changed, moved and deleted files. The worker then reads and indexes what changed, so it must be running. You don't normally need the button: the worker scans every 15 minutes by default, and within seconds if the file watcher is on.

### Roots

One row per notes folder:

| Column | Meaning |
|---|---|
| **Root** | The folder's short name |
| **Path** | Where it is on disk |
| **Watcher** | Whether changes are noticed immediately: *native* (the system reports changes) or *polling* (mynotes checks every 2 seconds) when the watcher is on; *down* when the folder can't be watched (for example, missing); *no state* when the watcher is off or the worker isn't running. **stale** means the worker hasn't reported for over 2 minutes: it has probably stopped |
| **Last scan** | When the last scan ran, its kind (*incremental*: only changes; *full*: re-reads everything; *targeted*: specific files), whether it finished (*done*), and what it found, for example `new=3 modified=1 unchanged=560` |

### Queue and KG backlog

- **queued**, **running**, **failed**: background jobs waiting, in progress, and failed after several attempts. A large *queued* number right after adding many files is normal; it goes down as the worker processes them. If it never goes down, the worker isn't running.
- **outbox**: changes waiting to be copied into the search index. This should be close to 0 most of the time.
- **queued by type**: which kinds of jobs are waiting (reading files, embedding, tagging, graph extraction…).
- **KG backlog**: progress of the knowledge graph. *Pending groups* are document sections still to be analysed; *cached* of *total* are done; the **ETA** estimates the time left (*unknown* until enough work has been timed).

### Index

The search index and how many passages it holds. Problems with it are listed in red. The **drift report** compares the search index with the database: *missing*, *extra* and *stale* should be 0. The report is updated every night; if the numbers aren't 0, run `uv run mynotes index reconcile` in the mynotes folder.

### Model endpoints

The three AI services mynotes uses (embeddings, reranker, LLM):

| Column | Meaning |
|---|---|
| **Process** | Which mynotes process made the calls: *web* (this web interface: your questions and searches) or *worker* (indexing, tagging, the knowledge graph). Worker rows show how long ago the worker last reported, which it does every 30 seconds while it runs |
| **State** | *closed*: working normally. *open*: paused after repeated failures; mynotes retries automatically after a short wait. *half_open*: trying again after a pause |
| **Calls** | Requests made |
| **p50 ms / p95 ms** | Typical and slowest response times, in milliseconds |
| **Errors** | Share of requests that failed |
| **Last error** | The most recent error message |

Web rows count the requests made since the web interface was started; worker rows are the worker's own counts since it started, refreshed every 30 seconds (an old "ago" value means the worker stopped). A worker row appears only after the worker has used that service at least once. *no endpoint used yet* means nothing has called the AI services since the processes started.

### Unsupported files

How many files in your folders mynotes can't read (unsupported formats, such as old `.doc` files), with examples, and how many online-only cloud files (iCloud, OneDrive) are skipped until they are downloaded.

### What "healthy" looks like

- every root shows a recent **Last scan** with status *done*;
- **queued** is low or going down, **failed** is 0 and **outbox** is about 0;
- the drift report shows 0 *missing*, 0 *extra*, 0 *stale*;
- every model endpoint shows *closed*.

---

## 11. Privacy and security

- **Everything stays on your computer**, except the text sent to the three AI services to embed passages, rank results, write answers and suggest tags.
- **No login is needed** in the standard single-user setup. For that reason the web interface only accepts connections from the computer it runs on (`127.0.0.1`). Don't make it reachable from other computers unless whoever runs your installation has set up sign-in: anyone who can reach it can read your notes.
- **mynotes never changes your notes.** The web interface reads them; all its own data (tags, reviews, versions) lives in its database.
- **Your chats are kept** in the mynotes database (question, answer and sources) for the **History** page, until you delete them there.
- **Your checkbox choices and the colour theme are remembered in your browser**, not on the server: Chat's **Include old notes versions**, Search's **include notes history** and **rerank**, and the System/Light/Dark switch keep the state you last chose, in that browser only. Tags and file types are not remembered, so a new search never starts with hidden filters. Clearing the browser's site data brings back the defaults.
- **Open in app** opens the file on the computer running mynotes. That is your computer in the standard setup; it is not offered for notes uploaded by an agent (server mode), whose originals are on the owner's machine.
- **On a shared server** you sign in with a personal token (the Sign in page); the cookie that keeps you signed in lives in your browser, and **Sign out** in the top bar removes it.

---

## 12. Troubleshooting

| Problem | What to do |
|---|---|
| The page doesn't load ("unable to connect") | The web interface isn't running. In a terminal: `cd ~/mynotes` then `uv run mynotes serve`, and keep the terminal open. Check the address: http://127.0.0.1:8765/ |
| Pages show an error about the database or OpenSearch | The databases aren't running. In a terminal: `cd ~/mynotes` then `docker compose up -d`, wait a minute, reload |
| A new or edited note doesn't appear | The worker must be running (on Status, **Watcher** shows *stale* or *no state* and the queue doesn't move when it isn't). Start it with `uv run mynotes worker`, or ask whoever runs your installation. Then click **Scan now** |
| An answer uses an outdated document | Look for a **historical** label on the sources. If a newer document should replace an older one, confirm the proposal in **Reviews → Versions**, or link them from the command line (`uv run mynotes versions add NEWER OLDER`) |
| The answer says the notes don't contain it, but they do | Try **Search** with other words to see what is found; remove filters; rephrase the question with the project or document name |
| **Open in app** does nothing or is missing | It opens the file on the computer running mynotes, with that computer's default application. It isn't available when mynotes runs in Docker or for notes uploaded by an agent. Use **Download** instead |
| A PDF opens at the first page instead of the cited one | Some browser PDF viewers ignore the page in the link; scroll to the page shown in the citation (`p. N`) |
| The Graph is empty or an entity isn't found | The graph is built in the background after indexing, at roughly 10 documents per minute. Check **KG backlog** on the Status page |
| A model endpoint shows *open* | That AI service is failing or unreachable (network, VPN, key). mynotes retries by itself; check **Last error**. Questions fail until it recovers |
| The **Ask** button stays grey | An answer is still being written; wait for it to finish |

---

## 13. Quick reference

| Page | Address |
|---|---|
| Chat | http://127.0.0.1:8765/ui/chat |
| Search | http://127.0.0.1:8765/ui/search |
| Reviews | http://127.0.0.1:8765/ui/reviews |
| Graph | http://127.0.0.1:8765/ui/graph |
| Graph, for one entity | http://127.0.0.1:8765/ui/graph?entity=NAME |
| Status | http://127.0.0.1:8765/ui/status |
| Help (this manual, the command-line manual, the MCP guide) | http://127.0.0.1:8765/ui/help |
| A document | http://127.0.0.1:8765/ui/documents/ID |
| HTTP API documentation (for connecting other programs) | http://127.0.0.1:8765/docs |

| Task | Where |
|---|---|
| Ask a question | Chat → type → **Ask** |
| Include older versions | Chat: tick **Include old notes versions**; Search: tick **include notes history**; or write `history:` |
| Reread or delete past chats | **History** |
| Limit to a tag | Chat: **tags** box; Search: click tags in the **Tags** section; or write `#tag` |
| Limit to a folder | Search: **Folder**; or write `in:folder` |
| Open a cited document | The links on its source line: **Open**, **View note**, **Download**, **Open in app** |
| Accept or reject a suggested tag | The document page's tags, or **Reviews → Tags** |
| Confirm that a document replaces another | **Reviews → Versions** → **Confirm** |
| Merge duplicate entities | **Reviews → Entities** → **Merge (keep the older)** |
| See how things connect | **Graph** → search a name → click circles and lines |
| Check that mynotes is up to date | **Status** |
| Look for changes now | **Status** → **Scan now** |
