# Changes Made — MultiAgentic RAG (60% → deployable + PDF upload)

This document explains every change made to your codebase, why it was necessary, and what to double-check before you deploy.

---

## Your two requirements, and how they were met

### 1. "It should be deployed"

There was no web server in the original code — `app.py` was a `python3 app.py` terminal loop using `input()`, which cannot be deployed as a web app.

**Added:**
- `server.py` — a new FastAPI server exposing the existing LangGraph pipeline over HTTP (upload, chat with streaming, retry, session management, health check).
- `static/index.html` — a single-page chat UI (upload a PDF, then chat) served directly by FastAPI. No separate frontend build step.
- `Dockerfile` — builds and runs the whole thing in a container. Installs system libs docling needs, pre-downloads the embedding model at build time so first requests aren't slow.
- `.dockerignore` — keeps the build context small.
- `DEPLOY.md` — step-by-step guide to deploy on **Render's free tier** (push to GitHub → connect repo → set 2 env vars → deploy), plus notes on Railway/Fly.io/VPS as drop-in alternatives since the Dockerfile is portable.

The CLI (`app.py`) still works too, in case you want to test locally without spinning up the server.

### 2. "Any PDF uploaded should be answered"

This was the deeper problem. The original retriever was built **once, at import time**, against a single hardcoded Chroma collection (`rag-chroma-google`) — there was no upload path at all, and `config.yaml` pointed to a PDF (`Magnetic_Disk_Formulas.pdf`) that didn't even exist in your zip.

**Changed:**
- `subgraph/graph_builder.py` — the retriever is no longer built at import time. It's now resolved **per-session**, lazily, from `RunnableConfig["configurable"]` (`collection_name` + `persist_directory`), and cached with `lru_cache` so a session's retriever is only built once and reused across that session's messages. This is the standard LangGraph pattern for passing per-request context into graph nodes.
- `retriever/retriever.py` — added a reusable `index_pdf(filepath, collection_name, persist_directory)` function. This is what actually gets called when someone uploads a PDF. The original CLI-only `__main__` block is kept, now just delegating to `index_pdf` for consistency.
- `main_graph/graph_builder.py` — `conduct_research` now accepts and forwards `config` into the researcher subgraph call, so the session's document context propagates all the way through routing → planning → research → retrieval.
- `server.py` — `/api/upload` saves the PDF, assigns it a UUID session, and indexes it into its own isolated Chroma collection under `sessions/<uuid>/`. `/api/chat` uses that session's `collection_name`/`persist_directory` for every retrieval call in that conversation. **You confirmed per-session isolation (not a shared/growing knowledge base), so uploads never mix.**

**Net effect:** upload any PDF → it gets its own private index → questions in that session are answered only from that document, exactly as you asked.

---

## Bugs found and fixed along the way

These weren't part of either explicit requirement, but they were real defects that would have broken the app even in its original single-document form. I'm listing them because you said you verify things independently rather than taking claims at face value — these are worth spot-checking yourself.

### Bug 1: Router would crash on a valid LLM response
- `utils/prompt.py`'s `ROUTER_SYSTEM_PROMPT` instructs the LLM to classify queries as `document_qa`, `more-info`, or `general`.
- But `main_graph/graph_states.py`'s `Router.type` only accepted the literal `"environmental"` (not `"document_qa"`), and `main_graph/graph_builder.py`'s `route_query()` only checked for `"environmental"`.
- Any time the LLM correctly followed its own prompt and returned `"document_qa"`, the code would hit the `else: raise ValueError(f"Unknown router type {_type}")` branch and crash.
- **Fixed:** unified everything on `"document_qa"` (matches the prompt, and is accurate now that this is a generic multi-document tool rather than one hardcoded to a Google environmental report).

### Bug 2: Cohere re-ranking would silently fail to authenticate
- Your `.env` had `CO_API_KEY`.
- `langchain_cohere.CohereRerank` (the actual library used in `subgraph/graph_builder.py`) reads the environment variable **`COHERE_API_KEY`**, not `CO_API_KEY` (confirmed against the library's source on GitHub). `CO_API_KEY` is what the *raw* `cohere` Python SDK looks for in some contexts, but not what this LangChain wrapper reads.
- **Fixed:** `.env.example` now uses `COHERE_API_KEY`, with a comment explaining the distinction. **You'll need to update your real `.env` to use `COHERE_API_KEY` instead of `CO_API_KEY`** — see "What you need to do" below.

### Bug 3: `requirements.txt` was incomplete
- A plain `pip install -r requirements.txt` on a clean machine would have failed at the first `import` in almost every file. Missing packages that are actually imported in the code: `langgraph`, `langchain-groq`, `docling`, `python-dotenv`, `PyYAML`.
- Also included `rank-llm==0.12.8`, which isn't imported anywhere in the codebase (only ever referenced in a commented-out line) and pulls in a large, unrelated dependency tree — removed.
- **Fixed:** rewrote `requirements.txt` with every package actually imported by the code, verified by grepping all `.py` files for top-level imports, plus `fastapi`/`uvicorn`/`python-multipart` for the new server.

### Bug 4: `config.yaml` pointed at a file that doesn't exist
- `retriever.file: "retriever/Magnetic_Disk_Formulas.pdf"` — this file isn't in your zip. Running the original indexing script as documented in your own README would have failed immediately with a file-not-found error.
- **Fixed:** this is now only a fallback default for the CLI (`app.py`) when no PDF path is passed as an argument — the web server never depends on it, since it always indexes whatever the user uploads. Left as a placeholder path with a comment explaining it's optional.

### Leaked API key — not fixed in code, flagged for you directly
- Your uploaded zip's `.env` contained **live-looking API keys** (a Groq key and a Cohere key) committed directly in plain text.
- I did **not** carry this file into the delivered zip — only `.env.example` (a template with placeholder values) is included.
- **You should treat both of those keys as compromised** since they were sent in an uploaded zip: rotate/regenerate them in the Groq and Cohere dashboards before deploying, and make sure `.env` (with the real keys) never gets committed to git or included in a zip you share again. `.gitignore` already excludes `.env`.

---

## Other structural changes

- **`app.py` (CLI)** — rewritten to accept a PDF path as a command-line argument (`python3 app.py your.pdf`), index it into a throwaway `vector_db/cli-session/` collection, then open the same streaming Q&A loop as before. Falls back to `config.yaml`'s `retriever.file` if no argument is given.
- **Session lifecycle** — `server.py` keeps an in-memory registry of sessions (`SESSIONS` dict) with status tracking (`indexing` / `ready` / `error`), a configurable size limit (`MAX_UPLOAD_MB`, default 25MB), and a TTL-based cleanup (`SESSION_TTL_SECONDS`, default 24h) that deletes old sessions' files. This is documented as a known limitation in the README: it's in-process, so it's fine for a single container but won't survive a restart or scale horizontally without swapping in Redis/a database — flagging this now so it's not a surprise later if you scale up.
- **Docstrings/comments** referencing the old "environmental report" framing were generalized to reflect that this is now a general-purpose document Q&A tool.

---

## What you need to do before deploying

1. **Rotate your Groq and Cohere API keys** (see "leaked API key" above) — treat the ones in your original `.env` as burned.
2. Copy `.env.example` to `.env` in your own environment and fill in the new keys, using `COHERE_API_KEY` (not `CO_API_KEY`).
3. Push the code to a GitHub repo (`.env` is git-ignored, so your keys won't be committed).
4. Follow `DEPLOY.md` to deploy on Render (or your host of choice — the Dockerfile is portable).

---

## What I did not change

- The core multi-agent graph logic (router → plan → parallel research → respond → hallucination check → human-in-the-loop retry) is untouched in its reasoning/flow — only *how the retriever is sourced* changed, from static/global to per-session/dynamic.
- Prompts in `utils/prompt.py` were left as-is (they were already written generically, e.g. "documents/PDFs", not tied to the old Google report).
- Model choice (Groq `llama-3.3-70b-versatile`), ensemble weights, and re-ranking config are unchanged from your defaults.
