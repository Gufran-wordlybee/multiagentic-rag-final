"""FastAPI web server for MultiAgentic RAG.

Exposes the existing LangGraph multi-agent pipeline (router -> planner ->
parallel research -> response -> hallucination check -> human-in-the-loop
retry) over HTTP, with per-session PDF upload and isolated retrieval.

Endpoints:
    POST /api/upload         Upload a PDF -> creates a new session, indexes it.
    POST /api/chat           Ask a question within a session (SSE streaming).
    POST /api/chat/retry     Resume a session after a hallucination-check interrupt.
    GET  /api/sessions/{id}  Session status (ready / indexing / error).
    DELETE /api/sessions/{id} Delete a session and its data.
    GET  /health             Liveness check for the deploy platform.
    GET  /                   Minimal built-in chat UI.

Each upload gets its own UUID, its own Chroma persist directory under
SESSIONS_DIR, and its own LangGraph thread_id, so concurrent users (or
multiple documents from the same user) never share data or context.
"""

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from main_graph.graph_builder import InputState, graph
from retriever.retriever import index_pdf
from utils.utils import new_uuid

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).parent
SESSIONS_DIR = Path(os.environ.get("SESSIONS_DIR", APP_DIR / "sessions"))
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "25"))
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", str(24 * 60 * 60)))

# In-memory session registry. Fine for a single-process deployment; if this
# is ever scaled to multiple workers/machines, swap this for Redis/SQLite.
# {session_id: {"status": "indexing"|"ready"|"error", "filename": str,
#                "persist_directory": str, "collection_name": str,
#                "created_at": float, "error": Optional[str]}}
SESSIONS: dict[str, dict] = {}


def _session_paths(session_id: str) -> tuple[Path, str]:
    persist_directory = SESSIONS_DIR / session_id
    collection_name = f"session-{session_id}"
    return persist_directory, collection_name


def _session_config(session_id: str) -> dict:
    """Build the RunnableConfig for a session: thread_id for the graph
    checkpointer, plus collection_name/persist_directory for the retriever.
    """
    persist_directory, collection_name = _session_paths(session_id)
    return {
        "configurable": {
            "thread_id": session_id,
            "collection_name": collection_name,
            "persist_directory": str(persist_directory),
        }
    }


def _require_ready_session(session_id: str) -> dict:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Upload a PDF first.")
    if session["status"] == "indexing":
        raise HTTPException(status_code=409, detail="This document is still being indexed. Try again shortly.")
    if session["status"] == "error":
        raise HTTPException(status_code=422, detail=f"Indexing failed: {session.get('error', 'unknown error')}")
    return session


def _cleanup_expired_sessions() -> None:
    now = time.time()
    expired = [sid for sid, s in SESSIONS.items() if now - s["created_at"] > SESSION_TTL_SECONDS]
    for sid in expired:
        _delete_session(sid)


def _delete_session(session_id: str) -> None:
    persist_directory, _ = _session_paths(session_id)
    shutil.rmtree(persist_directory, ignore_errors=True)
    SESSIONS.pop(session_id, None)


app = FastAPI(title="MultiAgentic RAG", description="Upload a PDF and ask questions about it.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Liveness check used by deploy platforms (Render, Railway, etc.)."""
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and index it into a brand-new, isolated session.

    Returns a session_id that must be passed to /api/chat for every
    question about this document.
    """
    _cleanup_expired_sessions()

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    session_id = new_uuid()
    persist_directory, collection_name = _session_paths(session_id)
    persist_directory.mkdir(parents=True, exist_ok=True)

    upload_path = persist_directory / "source.pdf"
    size = 0
    with open(upload_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                shutil.rmtree(persist_directory, ignore_errors=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max size is {MAX_UPLOAD_MB} MB.",
                )
            f.write(chunk)

    SESSIONS[session_id] = {
        "status": "indexing",
        "filename": file.filename,
        "persist_directory": str(persist_directory),
        "collection_name": collection_name,
        "created_at": time.time(),
        "error": None,
    }

    async def _index():
        try:
            await asyncio.to_thread(
                index_pdf,
                filepath=str(upload_path),
                collection_name=collection_name,
                persist_directory=str(persist_directory),
            )
            SESSIONS[session_id]["status"] = "ready"
            logger.info(f"Session {session_id} ready ({file.filename}).")
        except Exception as e:
            logger.exception(f"Failed to index session {session_id}")
            SESSIONS[session_id]["status"] = "error"
            SESSIONS[session_id]["error"] = str(e)

    asyncio.create_task(_index())

    return {"session_id": session_id, "filename": file.filename, "status": "indexing"}


@app.get("/api/sessions/{session_id}")
async def session_status(session_id: str):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "session_id": session_id,
        "filename": session["filename"],
        "status": session["status"],
        "error": session.get("error"),
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    _delete_session(session_id)
    return {"deleted": True}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class RetryRequest(BaseModel):
    session_id: str
    retry: bool = True


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_graph_response(session_id: str, astream_iter):
    """Consume a graph.astream(..., stream_mode='messages') iterator and
    yield Server-Sent Events of streamed tokens, then a final 'done' event
    that tells the client whether a hallucination-check retry is pending.
    """
    async for chunk, metadata in astream_iter:
        if chunk.content:
            yield _sse("token", {"text": chunk.content})

    session_config = _session_config(session_id)
    state_snapshot = graph.get_state(session_config)
    needs_retry = False
    # StateSnapshot.tasks holds any pending PregelTask for this thread; a
    # task has a non-empty .interrupts tuple exactly when a node (here,
    # human_approval_node) called interrupt() and is paused waiting for
    # /api/chat/retry to resume it.
    if state_snapshot and state_snapshot.tasks:
        needs_retry = any(task.interrupts for task in state_snapshot.tasks)

    yield _sse("done", {"needs_retry_confirmation": needs_retry})


def _has_pending_interrupt(session_config: dict) -> bool:
    snapshot = graph.get_state(session_config)
    return bool(snapshot and snapshot.tasks and any(t.interrupts for t in snapshot.tasks))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Ask a question about the session's document. Streams the answer via SSE."""
    _require_ready_session(req.session_id)

    session_config = _session_config(req.session_id)

    # If the previous turn is still paused on the hallucination-check
    # interrupt (the user never called /api/chat/retry), starting a new
    # question with fresh `input` on the same thread_id would collide with
    # that pending interrupt rather than cleanly beginning the new turn.
    # Auto-decline it ('n' -> check_retry_decision routes to END) so the
    # thread is clean before we send the new question.
    if _has_pending_interrupt(session_config):
        async for _ in graph.astream(Command(resume="n"), stream_mode="messages", config=session_config):
            pass

    # messages is typed as list[AnyMessage]; build the list explicitly
    # instead of passing a bare string, so this doesn't depend on
    # add_messages' implicit string coercion.
    input_state = InputState(messages=[HumanMessage(content=req.message)])

    astream_iter = graph.astream(input=input_state, stream_mode="messages", config=session_config)
    return StreamingResponse(
        _stream_graph_response(req.session_id, astream_iter),
        media_type="text/event-stream",
    )


@app.post("/api/chat/retry")
async def chat_retry(req: RetryRequest):
    """Resume a session after the hallucination grader flagged the answer
    as ungrounded, per the graph's human-in-the-loop interrupt().
    """
    _require_ready_session(req.session_id)

    session_config = _session_config(req.session_id)
    resume_value = "y" if req.retry else "n"

    astream_iter = graph.astream(Command(resume=resume_value), stream_mode="messages", config=session_config)
    return StreamingResponse(
        _stream_graph_response(req.session_id, astream_iter),
        media_type="text/event-stream",
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = APP_DIR / "static" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>MultiAgentic RAG</h1><p>UI not found. See /docs for the API.</p>"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
