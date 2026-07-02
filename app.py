#!/usr/bin/env python3
"""CLI entrypoint for MultiAgentic RAG.

Usage:
    python3 app.py path/to/document.pdf
    python3 app.py                          # falls back to config.yaml's retriever.file

Indexes the given PDF into its own throwaway session (vector_db/cli-session/)
and then opens the same streaming Q&A loop as before. For a persistent,
multi-user, uploadable experience, run `python3 server.py` instead (see README).
"""

import asyncio
import sys
import time
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from main_graph.graph_builder import InputState, graph
from retriever.retriever import index_pdf
from utils.utils import config, new_uuid

CLI_SESSION_DIR = Path("vector_db") / "cli-session"


def _resolve_pdf_path() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    return config["retriever"]["file"]


async def process_query(query: str, thread: dict):
    input_state = InputState(messages=[HumanMessage(content=query)])

    async for c, metadata in graph.astream(input=input_state, stream_mode="messages", config=thread):
        if c.additional_kwargs.get("tool_calls"):
            print(c.additional_kwargs.get("tool_calls")[0]["function"].get("arguments"), end="", flush=True)
        if c.content:
            time.sleep(0.05)
            print(c.content, end="", flush=True)

    state = graph.get_state(thread)
    needs_retry = bool(state and state.tasks and any(t.interrupts for t in state.tasks))
    if needs_retry:
        response = input("\nThe response may contain uncertain information. Retry the generation? If yes, press 'y': ")
        async for c, metadata in graph.astream(Command(resume=response.lower()), stream_mode="messages", config=thread):
            if c.additional_kwargs.get("tool_calls"):
                print(c.additional_kwargs.get("tool_calls")[0]["function"].get("arguments"), end="")
            if c.content:
                time.sleep(0.05)
                print(c.content, end="", flush=True)


async def main():
    pdf_path = _resolve_pdf_path()
    if not Path(pdf_path).exists():
        print(f"PDF not found: {pdf_path}")
        print("Pass a path: python3 app.py path/to/document.pdf")
        sys.exit(1)

    print(f"Indexing '{pdf_path}'... (first run may take a minute)")
    CLI_SESSION_DIR.mkdir(parents=True, exist_ok=True)
    index_pdf(
        filepath=pdf_path,
        collection_name="cli-session",
        persist_directory=str(CLI_SESSION_DIR),
    )
    print("Ready.\n")

    thread = {
        "configurable": {
            "thread_id": new_uuid(),
            "collection_name": "cli-session",
            "persist_directory": str(CLI_SESSION_DIR),
        }
    }

    print("Enter your query (type '-q' to quit):")
    while True:
        query = input("> ")
        if query.strip().lower() == "-q":
            print("Exiting...")
            break
        await process_query(query, thread)


if __name__ == "__main__":
    asyncio.run(main())
