"""
server.py
=========
FastAPI backend for the data analytics AI agent.

Endpoints
---------
GET  /                  Serve frontend.html
GET  /api/files         List CSV files in the datasets/ directory
POST /api/stream        Stream agent execution via Server-Sent Events

SSE Event Schema
----------------
Every SSE message is a JSON object with a `type` field:

  { "type": "node_start",    "node": str }
  { "type": "node_end",      "node": str }
  { "type": "token",         "text": str }
  { "type": "step",          "index": int, "reasoning": str, "code": str }
  { "type": "step_result",   "index": int, "stdout": str, "error": str | null }
  { "type": "plot",          "title": str, "figure_json": dict }
  { "type": "validation",    "verdict": str, "reasoning": str }
  { "type": "final_answer",  "answer": str }
  { "type": "done" }
  { "type": "error",         "message": str }

Usage
-----
  python server.py
  # or with auto-reload:
  uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Analytics Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASETS_DIR = Path("datasets")
DATASETS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class InvokeRequest(BaseModel):
    question: str
    csv_files: list[str]  # Bare filenames from datasets/


# ---------------------------------------------------------------------------
# File listing endpoint
# ---------------------------------------------------------------------------


@app.get("/api/files")
async def list_files() -> dict:
    """Return all CSV filenames found in the datasets/ directory."""
    files = sorted((f.name for f in DATASETS_DIR.iterdir() if f.suffix == ".csv"), reverse=True)
    return {"files": files}


# ---------------------------------------------------------------------------
# Streaming endpoint
# ---------------------------------------------------------------------------


def _emit(event_type: str, **data) -> str:
    """Format one SSE message."""
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


# Nodes shown in the progress timeline (order matters for display).
TRACKED_NODES = {
    "orchestrate",
    "dataset_fan",
    "analytics",
    "plan_execute",
    "validate",
    "retry",
    "generic",
    "finalize",
}


async def _event_stream(question: str, csv_files: list[str]) -> AsyncIterator[str]:
    """
    Invoke the compiled LangGraph graph and forward events as SSE.

    Uses graph.astream_events(version="v2") for fine-grained streaming:
      - on_chain_start / on_chain_end  → node lifecycle
      - on_chat_model_stream           → LLM token streaming
    """
    try:
        # Lazy import — avoids loading the heavy graph at startup.
        from graph import graph  # type: ignore
        from schemas import GraphInput  # type: ignore
        from langchain_core.runnables import RunnableConfig  # type: ignore
    except ImportError as exc:
        yield _emit(
            "error",
            message=f"Import error: {exc}. Make sure graph.py and schemas.py are in the Python path.",
        )
        return

    try:
        async for event in graph.astream_events(
            GraphInput(question=question, csv_paths=csv_files),
            version="v2",
            config=RunnableConfig(tags=["analytics-agent-ui"]),
        ):
            kind: str = event["event"]
            name: str = event.get("name", "")
            data: dict = event.get("data", {})

            # ── Node lifecycle ────────────────────────────────────────────
            if kind == "on_chain_start" and name in TRACKED_NODES:
                yield _emit("node_start", node=name)

            elif kind == "on_chain_end" and name in TRACKED_NODES:
                yield _emit("node_end", node=name)
                output = data.get("output", {}) or {}

                # Emit plots as soon as they appear in state.
                for plot in output.get("plots", []):
                    yield _emit(
                        "plot", title=plot["title"], figure_json=plot["figure_json"]
                    )

                # Emit validation verdict.
                if name == "validate":
                    vr = output.get("validation_result")
                    if vr and isinstance(vr, dict):
                        yield _emit(
                            "validation",
                            verdict=vr.get("verdict", ""),
                            reasoning=vr.get("reasoning", ""),
                        )

                # Emit final answer + final plots from finalize_node.
                if name == "finalize":
                    answer = output.get("answer", "")
                    if answer:
                        yield _emit("final_answer", answer=answer)
                    for plot in output.get("final_plots", []):
                        yield _emit(
                            "plot", title=plot["title"], figure_json=plot["figure_json"]
                        )

            # ── Analytics step details ────────────────────────────────────
            elif kind == "on_chain_end" and name == "analytics_step":
                output = data.get("output", {}) or {}
                steps: list = output.get("steps", [])
                if steps:
                    latest = steps[-1]
                    yield _emit(
                        "step",
                        index=latest.get("step_index", 0),
                        reasoning=latest.get("reasoning", ""),
                        code=latest.get("code", ""),
                    )
                    ex = latest.get("execution", {})
                    yield _emit(
                        "step_result",
                        index=latest.get("step_index", 0),
                        stdout=ex.get("stdout", ""),
                        error=ex.get("error"),
                    )

            # ── LLM token streaming ───────────────────────────────────────
            elif kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield _emit("token", text=chunk.content)

        yield _emit("done")

    except Exception as exc:
        import traceback

        yield _emit("error", message=str(exc), traceback=traceback.format_exc())


@app.post("/api/stream")
async def stream_agent(request: InvokeRequest) -> StreamingResponse:
    """
    Stream agent execution for the given question + CSV file list.
    Returns an SSE stream consumed by the frontend.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    # if not request.csv_files:
    #     raise HTTPException(
    #         status_code=400, detail="At least one CSV file must be selected."
    #     )
    # Validate files exist in datasets/.
    for name in request.csv_files:
        if not (DATASETS_DIR / name).exists():
            raise HTTPException(status_code=404, detail=f"Dataset not found: {name}")

    return StreamingResponse(
        _event_stream(request.question, request.csv_files),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Frontend serving
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def serve_frontend() -> str:
    """Serve the single-page frontend."""
    frontend = Path("frontend.html")
    if not frontend.exists():
        return "<h1>frontend.html not found — place it next to server.py</h1>"
    return frontend.read_text()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True, log_level="info")
