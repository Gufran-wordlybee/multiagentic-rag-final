# MultiAgentic RAG — Project Status (this pass)

This continues from the previous `PROJECT_STATUS.md` (the docling/Dockerfile/
router fixes described there are still in place and I re-verified them by
re-reading every changed line — see "Re-verified from the last pass" below).
This pass focused on your two hard requirements:

1. It should be deployed.
2. Any PDF uploaded should be answered.

I did **not** have internet access in this sandbox (same constraint as last
time — Hugging Face, PyPI, Groq, and Cohere are all unreachable here), so
nothing involving a live LLM call, a live model download, or an actual
`pip install` could be run end-to-end. What I *did* do this pass, since I
couldn't re-run the server: a full manual trace of every module — reading
every function, every graph edge, every state field — against how LangGraph,
docling, and FastAPI actually behave, rather than re-skimming the previous
summary. That's how I found the issues below; they're not visible from a
casual read-through, only from tracing what happens on resume/retry and on a
second question after a flagged answer.

---

## 🐛 Bugs found and fixed this pass (beyond the previous pass)

### 1. `interrupt()` was called from a routing function, not a node — fixed
This is the important one. The hallucination-retry flow used to look like:

```python
def human_approval(state):
    if state.hallucination.binary_score == "1":
        return "END"
    else:
        retry_generation = interrupt({...})   # <-- called inside a conditional-edge function
        return "respond" if retry_generation == "y" else "END"

builder.add_conditional_edges("check_hallucinations", human_approval, {...})
```

LangGraph's `interrupt()` works by **replaying the node that called it** from
the top when you resume with `Command(resume=...)` — that's the mechanism
that lets it return the resume value in place of the paused call. Routing
functions passed to `add_conditional_edges` are meant to be plain/pure — they
aren't checkpointed and replayed the way nodes are. Calling `interrupt()`
there leaves `/api/chat/retry`'s resume behavior undefined: it might work by
accident, or it might not, depending on LangGraph internals I can't fully
verify without running it live against your actual `langgraph==0.2.62` pin.

**Fix**: split this into a real node (`human_approval_node`) that calls
`interrupt()` and just writes the outcome into a new `retry_decision` state
field, plus a separate, ordinary routing function (`check_retry_decision`)
that only reads that field — no side effects, safe to be whatever LangGraph
needs it to be. Graph wiring is now:

```
respond → check_hallucinations → human_approval → (check_retry_decision) → respond | END
```

Behavior from the outside (the API contract, the SSE events, the UI) is
**unchanged** — `/api/chat/retry` still takes `{session_id, retry: bool}` and
streams the same events. This was purely an internal correctness fix so the
retry flow that's central to requirement #2 actually behaves the way the
code implies it does.

I also reset `retry_decision` back to `None` at the start of every fresh
`create_research_plan` call (same place `documents` already gets cleared),
so a leftover decision from a previous turn can't bleed into a new one.

### 2. A new question could collide with an unresolved retry — fixed
Related to #1: if a user got a flagged answer and just... asked a new
question instead of clicking "Retry" or "Keep it," the graph thread was left
paused mid-interrupt. The old code had no guard for this, so the next
`/api/chat` call would send fresh input into a thread that LangGraph still
considered "waiting to be resumed" — undefined behavior, not a clean new
turn. Fixed: `/api/chat` now checks for a pending interrupt first and, if
one exists, silently auto-resumes it with "n" (equivalent to the user
clicking "keep it") before starting the new question. This can't happen via
the web UI's own retry box (it only ever sends `retry: true`), but it's a
real gap for direct API callers, the CLI, or just impatient users.

### 3. `InputState(messages=req.message)` relied on implicit string coercion
`messages` is typed `Annotated[list[AnyMessage], add_messages]` — a list.
Passing a bare string worked only because `add_messages` happens to coerce a
lone string into a `HumanMessage`. That's not documented, load-bearing
behavior to depend on. Changed both `server.py` and `app.py` to build
`[HumanMessage(content=...)]` explicitly.

### 4. Docling's build-time model cache wasn't explicitly wired to runtime
The Dockerfile change from the last pass (calling
`StandardPdfPipeline.download_models_hf()` at build time) is correct and
still in place. But the runtime `DocumentConverter()` call in
`retriever.py` never told docling *where* those baked-in weights were — it
was relying on docling's default `~/.cache` resolving to the same path at
build time and at runtime, which only holds because neither Docker stage
sets a `USER` (both run as root, so `~` is `/root` in both places). That's
true today, but it's an implicit assumption, not a guarantee — it would
silently break if anyone later hardens the image with a non-root `USER`.
Fixed: the Dockerfile now captures the actual path `download_models_hf()`
downloaded to, writes it to a file baked into the image, and
`retriever.py` reads that file and passes it to `PdfPipelineOptions`
explicitly. If the file isn't present (e.g. running outside Docker), it
falls back to docling's normal default behavior — no change for local dev.

### 5. Interrupt-detection code used opaque tuple indexing
`state[-1][0].interrupts` (indexing into `graph.get_state()`'s return value)
worked but was fragile and unreadable. Replaced with the named field it's
actually reaching for: `state_snapshot.tasks`, then `task.interrupts` per
task. Same information, no behavior change, but much less likely to break
silently if the tuple shape ever shifts.

---

## Re-verified from the last pass (still correct, re-checked this session)

- `docling-core==2.14.0` pin alongside `docling==2.15.1` in
  `requirements.txt` — still the fix for the `ImportError` that would
  otherwise break every single PDF upload on a fresh install.
- Router prompt/enum consistency (`more-info` with a hyphen, matching the
  `Literal["more-info", "document_qa", "general"]` in `graph_states.py`) —
  confirmed still aligned.
- `COHERE_API_KEY` (not `CO_API_KEY`) in `.env.example` — confirmed correct
  against `langchain_cohere.CohereRerank`'s actual env var.
- `requirements.txt` has `langgraph`, `langchain-groq`, `docling`,
  `python-dotenv`, `PyYAML` all present.
- `config.yaml`'s `retriever.file` is still correctly ignored by
  `server.py`, which always indexes whatever's uploaded.

---

## ⚠️ Still-true limitations (unchanged from last pass — not fixed, by design/scope)

These weren't part of your two stated requirements, so I left them as
documented last time rather than doing an unrequested architecture change:

1. **Sessions and the LangGraph checkpointer are in-memory.** A restart
   (redeploy, crash, Render free-tier spin-down) loses all active sessions.
   Fine for a demo; would need `SqliteSaver`/`PostgresSaver` + a persistent
   `SESSIONS` store to survive restarts. Say the word and I'll do this next.
2. **Single-process only** — the in-memory session registry and the
   `lru_cache`-based retriever cache in `subgraph/graph_builder.py` assume
   exactly one server process. Fine for the single free-tier instance
   `DEPLOY.md` sets up; would need Redis/shared storage to scale out.
3. **No background session-cleanup timer** — `_cleanup_expired_sessions()`
   only runs when someone hits `/api/upload`. Not a functional bug.
4. **First upload after a fresh deploy is still a bit slower** than
   subsequent ones (Python-side warm caches), even with build-time model
   downloading.
5. **I still could not run a real end-to-end answer** (upload → question →
   cited answer) in this sandbox — no internet access to Hugging Face, and
   I don't have Groq/Cohere keys (nor would I want you pasting live ones
   here). Everything I could verify statically — the graph wiring, the
   state machine, the interrupt/resume mechanics, the retriever construction
   path, the Docker build steps — checks out. I want to be upfront that the
   LLM-calling nodes themselves (`router`, `planner`, `respond`) are
   unchanged from your working version and I have no reason to think
   they're broken, but "no reason to think it's broken" isn't the same as
   "I watched it work."
6. **Rotate your Groq/Cohere keys** if you haven't yet — same note as last
   time, still applies, still not in this zip (only `.env.example` is).

---

## Files changed in this pass

| File | Change |
|---|---|
| `MultiAgenticRAG/main_graph/graph_states.py` | Added `retry_decision: Optional[str]` field to `AgentState` |
| `MultiAgenticRAG/main_graph/graph_builder.py` | Split `human_approval` (interrupt-in-conditional-edge) into `human_approval_node` (interrupt-in-node) + `check_retry_decision` (pure routing fn); rewired graph edges accordingly; reset `retry_decision` in `create_research_plan`; widened its return type hint |
| `MultiAgenticRAG/server.py` | Build `InputState.messages` explicitly as `[HumanMessage(...)]`; added `_has_pending_interrupt` guard in `/api/chat` to auto-resolve a stale interrupt before starting a new question; replaced tuple-indexing interrupt check with named `.tasks`/`.interrupts` access |
| `MultiAgenticRAG/app.py` | Same `HumanMessage` fix; same named-field interrupt check; CLI now always resumes the interrupt with whatever the user typed (previously, typing anything other than `'y'` left the thread stuck mid-interrupt forever) |
| `MultiAgenticRAG/Dockerfile` | Model-priming step now captures the actual artifacts path from `download_models_hf()` and bakes it into the image via a file, instead of relying on `~/.cache` implicitly matching between build and runtime |
| `MultiAgenticRAG/retriever/retriever.py` | Reads that baked-in artifacts path (if present) and passes it to `PdfPipelineOptions.artifacts_path` explicitly; falls back to docling's normal default behavior if not present (e.g. local dev) |
| `MultiAgenticRAG/README.md` | Updated the architecture diagram and groundedness-check section to describe the node-based interrupt flow accurately |

Everything else — prompts, retrieval strategy, ensemble weights, config,
the researcher subgraph, the frontend — is unchanged from what you gave me
this pass. I read through all of it looking for other issues in the same
"looks fine on a skim, breaks on a careful trace" category and didn't find
more; the two items above (interrupt/node mismatch, stale-interrupt
collision) were the real ones.

## What I'd want to double check with real keys, if you run it

The retry flow (#1/#2 above) is the part I'm most confident is *now*
correct by design, but least able to prove without actually clicking
through: upload a PDF, ask something the document can't answer, see
whether the "Retry the generation?" box appears, click **Retry**, confirm
you get a fresh answer streamed back — then repeat but click **Keep it** (or
just ask a different question without responding to the box) and confirm
that works cleanly too. If either of those misbehaves once you have live
keys, tell me exactly what you saw and I'll dig in from there.
