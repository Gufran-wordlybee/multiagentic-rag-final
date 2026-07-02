# MultiAgentic RAG

**A self-correcting, multi-agent Retrieval-Augmented Generation system built with LangGraph — upload any PDF, ask questions about it, deployed as a web app.**

## Overview

MultiAgentic RAG answers questions over **any PDF you upload** using a graph of cooperating LLM agents rather than a single retrieve-then-generate call.

Instead of naively embedding a question and stuffing the top-k chunks into a prompt, the system:

1. **Routes** the query — deciding if it needs more information, is answerable from the document, or is a general/off-topic question.
2. **Plans** a short multi-step research strategy for on-topic questions.
3. **Researches** each step in parallel — expanding it into several search queries, retrieving with an ensemble of retrievers, and re-ranking with Cohere.
4. **Responds** using only the retrieved evidence, with inline citations.
5. **Grades itself** for hallucinations and — if the answer isn't well-grounded — pauses for **human-in-the-loop approval** before deciding whether to retry.

This turns RAG from a single hop into an auditable, self-correcting pipeline.

### What's new in this version

- **Upload any PDF, from a browser.** No more editing `config.yaml` and re-running a script — drop a file into the web UI and start asking questions within seconds.
- **Per-session isolation.** Every upload gets its own UUID, its own Chroma collection, and its own persist directory. Two people (or two documents) never see each other's data.
- **Deployable.** A FastAPI server (`server.py`) + Dockerfile replace the terminal-only `app.py` loop. See [`DEPLOY.md`](../DEPLOY.md) for a step-by-step guide to deploying on Render's free tier.
- **CLI still works.** `python3 app.py path/to/file.pdf` still gives you the original terminal experience, for local testing.

### Key Features

- **LLM-based Query Router** — classifies each query as `document_qa`, `more-info`, or `general` before doing any retrieval work.
- **Multi-Step Research Planning** — an LLM breaks the question into a short (1–3 step) research plan.
- **Query Fan-Out** — each research step is expanded into multiple sub-queries and retrieved **in parallel** via LangGraph's `Send` API.
- **Hybrid Ensemble Retrieval** — combines dense similarity search, MMR search, and BM25 (sparse/keyword) search.
- **Cohere Contextual Re-Ranking** — re-ranks the ensemble's candidates for higher precision before they reach the LLM.
- **Hallucination Grading + Human-in-the-Loop** — a grader LLM checks whether the final answer is supported by the retrieved documents; ungrounded answers trigger a LangGraph `interrupt()` so a human can approve a retry.
- **Streaming responses** — token-by-token streaming, both in the terminal (`app.py`) and over Server-Sent Events (`server.py` + web UI).
- **Config-Driven** — model names, chunk headers, and retrieval weights live in `config.yaml`.
- **Per-Session Document Ingestion** — PDF → Markdown (via Docling) → header-aware chunking → Chroma vector store, built automatically the moment a user uploads a file.

---

## Quick Start (local)

### Prerequisites

- Python 3.10+ (3.11 recommended)
- A [Groq](https://groq.com/) API key (LLM inference)
- A [Cohere](https://cohere.com/) API key (re-ranking)

### Installation

```bash
cd MultiAgenticRAG

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```bash
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
```

> **Note:** the API key env var is `COHERE_API_KEY` (that's what `langchain_cohere` reads), not `CO_API_KEY`.

### Run the web app (recommended)

```bash
python3 server.py
```

Open **http://localhost:8000**, drop in a PDF, wait for it to finish indexing, then ask questions about it.

### Run the CLI (optional)

```bash
python3 app.py path/to/your.pdf
```

```
Indexing 'path/to/your.pdf'...
Ready.

Enter your query (type '-q' to quit):
> What does this document say about X?
```

If you run `python3 app.py` with no argument, it falls back to `config.yaml`'s `retriever.file`.

---

## Deployment

See **[`DEPLOY.md`](../DEPLOY.md)** for full step-by-step instructions to deploy this as a live web app on Render's free tier (Docker-based, ~10 minutes, no credit card required for the free instance type).

In short: this repo ships a `Dockerfile`. Any container host (Render, Railway, Fly.io, a VPS with Docker, etc.) can build and run it directly — set the two environment variables (`GROQ_API_KEY`, `COHERE_API_KEY`) in the host's dashboard and deploy.

---

## Architecture

Two LangGraph graphs: a **main conversational graph** that owns routing, planning and response generation, and a **researcher subgraph** it calls into for each research step. Sitting in front of both is a **FastAPI server** that manages PDF uploads and per-session isolation.

```
                    Browser (static/index.html)
                              │
                              │  POST /api/upload (PDF)
                              ▼
                    ┌───────────────────┐
                    │   server.py        │  creates session_id,
                    │  (FastAPI)          │  indexes PDF into its own
                    └─────────┬───────────┘  Chroma collection
                              │
                              │  POST /api/chat {session_id, message}
                              ▼
                    ┌────────────────────────────┐
                    │     analyze_and_route_query  │
                    │  (Router: more-info /        │
                    │   document_qa / general)      │
                    └───────────────┬──────────────┘
                     ┌────────────────────────┼───────────────────────┐
                     ▼                        ▼                       ▼
          ┌────────────────────┐  ┌─────────────────────┐  ┌────────────────────────┐
          │ ask_for_more_info   │  │ create_research_plan │  │ respond_to_general_query│
          │ (asks 1 follow-up)  │  │ (LLM plans ≤3 steps) │  │ (politely declines)     │
          └─────────┬───────────┘  └───────────┬──────────┘  └────────────┬────────────┘
                    END                         ▼                        END
                                    ┌────────────────────────┐
                                    │    conduct_research     │◄────────────┐
                                    │ (calls researcher_graph │             │
                                    │  for one step at a time,│             │
                                    │  session config passed  │             │
                                    │  through)                │             │
                                    └────────────┬─────────────┘            │
                                                 ▼                         │
                                      steps remaining? ─── yes ────────────┘
                                                 │ no
                                                 ▼
                                    ┌────────────────────────┐
                                    │         respond          │
                                    │ (answers using retrieved │
                                    │  docs, cites sources)    │
                                    └────────────┬─────────────┘
                                                 ▼
                                    ┌────────────────────────┐
                                    │   check_hallucinations   │
                                    │ (LLM grades groundedness)│
                                    └────────────┬─────────────┘
                                                 ▼
                                                 ▼
                                    ┌────────────────────────┐
                                    │     human_approval       │
                                    │ (binary_score==1: auto- │
                                    │  approve; else pauses    │
                                    │  via interrupt() for a   │
                                    │  human decision)         │
                                    └────────────┬─────────────┘
                                                 ▼
                                    retry_decision == "y"? ── yes ── respond
                                                 │ no/approved
                                                 ▼
                                                END
```

### Researcher Subgraph (per research step)

```
┌──────────────────┐        ┌──────────────────────────────┐
│  generate_queries │ ─────▶ │  retrieve_and_rerank_documents │  (fanned out in parallel
│  (LLM expands the │  Send  │  • resolves this session's     │   via LangGraph Send,
│   step into N      │  x N   │    retriever from config       │   one call per query)
│   search queries)   │        │  • Ensemble retriever:         │
└──────────────────┘        │    - similarity + MMR + BM25   │
                              │  • Cohere re-rank (top_k)      │
                              └──────────────────────────────┘
```

The retriever used to be built once at import time against one hardcoded PDF/collection. It's now resolved per-request from `RunnableConfig["configurable"]` (`collection_name`, `persist_directory`), which `server.py` sets per session — so every uploaded document gets its own isolated retriever, built lazily and cached for the life of that session.

### Document Ingestion Pipeline (`retriever/retriever.py`, runs on upload)

```
Uploaded PDF (Docling, CPU accelerator)
   │  convert → export_to_markdown()
   ▼
Markdown
   │  MarkdownHeaderTextSplitter (splits on #, ##)
   ▼
Chunks
   │  HuggingFaceEmbeddings ("BAAI/bge-small-en-v1.5")
   ▼
Chroma vector store (persisted to sessions/<session_id>/)
```

---

## How the Router & Decision Logic Work

| Classification | Meaning | Next Node |
|---|---|---|
| `more-info` | The question is ambiguous or missing a key detail | `ask_for_more_info` — asks exactly one clarifying follow-up |
| `document_qa` | The question can be answered from the uploaded document | `create_research_plan` → `conduct_research` |
| `general` | Off-topic / unrelated to the document | `respond_to_general_query` — politely declines |

### Retrieval Strategy

Each generated query is retrieved through an **ensemble** of three retrievers, weighted and combined, then compressed/re-ranked:

| Retriever | Role | Weight (default) |
|---|---|---|
| Similarity (dense) | Semantic nearest-neighbour search | 0.3 |
| MMR (dense) | Diversity-aware semantic search | 0.3 |
| BM25 (sparse) | Keyword/lexical matching | 0.4 |

The combined candidate pool is then passed through **Cohere Rerank** (`rerank-english-v3.0`) to select the most relevant `top_k_compression` documents, configurable in `config.yaml`.

### Groundedness Check

After `respond` generates an answer, `check_hallucinations` asks a grader LLM to output a binary score (`1` = grounded, `0` = not grounded). The result is passed to a dedicated `human_approval` node: a `1` score auto-approves and ends the turn; a `0` score calls a LangGraph `interrupt()`, surfaced in the web UI as a "Retry the generation?" prompt. `interrupt()` is called from a node (never from a routing/conditional-edge function) because LangGraph resumes an interrupt by replaying the node that raised it — that's what makes `/api/chat/retry` reliable.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/upload` | POST | Upload a PDF (`multipart/form-data`, field `file`). Returns `{session_id, filename, status}`. |
| `/api/sessions/{id}` | GET | Poll indexing status: `indexing`, `ready`, or `error`. |
| `/api/chat` | POST | `{session_id, message}` → streams the answer as Server-Sent Events. |
| `/api/chat/retry` | POST | `{session_id, retry: true\|false}` → resumes after a hallucination-check interrupt. |
| `/api/sessions/{id}` | DELETE | Delete a session and its indexed data. |
| `/health` | GET | Liveness check. |

Interactive API docs are auto-generated at `/docs` (Swagger UI) once the server is running.

---

## Configuration

`config.yaml` holds non-secret tunables (model names, chunk headers, retrieval weights). `retriever.file` is only used as the CLI's default PDF when no path is passed — the web server always uses whatever PDF the user uploads.

```yaml
retriever:
  file: "retriever/sample.pdf"     # CLI fallback only
  headers_to_split_on:
    - ["#", "Header 1"]
    - ["##", "Header 2"]
  top_k: 3
  top_k_compression: 3
  ensemble_weights: [0.3, 0.3, 0.4]
  cohere_rerank_model: rerank-english-v3.0
llm:
  groq_model: llama-3.3-70b-versatile
  temperature: 0
```

By default the system runs on **Groq** (`llama-3.3-70b-versatile`) for all LLM calls; OpenAI client code is present but commented out and can be swapped back in per-node.

---

## Project Structure

```
MultiAgenticRAG/
├── server.py                     # FastAPI web server — upload + chat endpoints (NEW)
├── static/index.html             # Built-in single-page chat UI (NEW)
├── app.py                        # CLI entrypoint — now indexes any PDF path passed as an argument
├── config.yaml                   # Non-secret configuration
├── requirements.txt              # Now includes langgraph, langchain-groq, docling, fastapi (previously missing)
├── Dockerfile                    # For containerized deployment (NEW)
├── .env.example                  # Template for required API keys (NEW)
├── main_graph/
│   ├── graph_builder.py          # Router, planner, respond, hallucination check, human approval
│   └── graph_states.py           # AgentState, Router, GradeHallucinations, InputState
├── subgraph/
│   ├── graph_builder.py          # generate_queries + retrieve_and_rerank_documents (now per-session)
│   └── graph_states.py           # ResearcherState, QueryState
├── retriever/
│   └── retriever.py              # PDF -> Markdown -> Chroma indexing, now exposes index_pdf() for reuse
├── utils/
│   ├── prompt.py                 # All system prompts
│   └── utils.py                  # config loader, UUID helpers, reduce_docs state reducer
└── sessions/                     # Per-upload Chroma collections (generated, git-ignored)
```

---

## Known Limitations

- Sessions are stored on local disk (`sessions/`) and held in an in-process dict — fine for a single-container deployment, but won't survive a restart or scale across multiple server instances without swapping in Redis/S3-backed session storage.
- Very large PDFs (100+ pages) can take a minute or two to index on the free tier of most hosts, since Docling's conversion is CPU-bound.
- Sessions expire after 24h by default (`SESSION_TTL_SECONDS`), deleting the uploaded PDF and its index.

## License
This project is private. All rights reserved by [Gufran](https://github.com/Gufran-wordlybee)

## Contact
**Gufran Alam**
- **Email:** <a href="mailto:justgufran07@gmail.com">justgufran07@gmail.com</a>
- **LinkedIn:** <a href="https://www.linkedin.com/in/gufran-alam-a25717321/" target="_blank">linkedin.com/in/gufran-alam-a25717321</a>
