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
    files = sorted(
        (f.name for f in DATASETS_DIR.iterdir() if f.suffix == ".csv"), reverse=True
    )
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
    # "analytics",
    "analytics_init",
    "analytics_step",
    "analytics_answer",
    "plan",
    "plan_execute",
    "plan_respond",
    "validate",
    "retry",
    # "generic",
    "generic_search",
    "generic_respond",
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

    nodes_seen: dict[str, int] = {}

    try:
        async for event in graph.astream_events(
            GraphInput(question=question, csv_paths=csv_files),
            version="v2",
            config=RunnableConfig(tags=["analytics-agent-ui"]),
        ):
            kind: str = event["event"]
            name: str = event.get("name", "")
            data: dict = event.get("data", {})

            if name == "LangGraph":
                continue

            if kind != "on_chain_stream":
                print(f"{"="*5} {name}: {kind} {"="*5}")
                # print(data)

            # ── Node lifecycle ────────────────────────────────────────────
            if kind == "on_chain_start":
                if name not in nodes_seen:
                    nodes_seen[name] = 0
                yield _emit("node_start", node=str(nodes_seen[name]) + "--" + name)

            elif kind == "on_chain_end":
                yield _emit("node_end", node=str(nodes_seen[name]) + "--" + name)
                nodes_seen[name] += 1
                output = data.get("output", {}) or {}

                # Emit plots as soon as they appear in state.
                # for plot in output.get("plots", []):
                #     print(plot)
                #     yield _emit(
                #         "plot", title=plot["title"], figure_json=plot["figure_json"]
                #     )

                # Emit validation verdict.
                if name == "validate":
                    vr = output.get("validation_result")
                    if vr:
                        yield _emit(
                            "validation",
                            verdict=vr.verdict,
                            reasoning=vr.reasoning,
                        )

                if name == "analytics_step":
                    steps: list = output.get("steps", [])
                    if steps:
                        latest = steps[-1]
                        index = (
                            str(output.get("retry_count", 0))
                            + ":"
                            + str(latest.get("step_index", 0))
                        )
                        yield _emit(
                            "step",
                            index=index,
                            reasoning=latest.get("reasoning", ""),
                            code=latest.get("code", ""),
                        )
                        ex = latest.get("execution", {})
                        yield _emit(
                            "step_result",
                            index=index,
                            stdout=ex.get("stdout", ""),
                            error=ex.get("error"),
                        )

                if name == "plan":
                    plan_execute_result = output.get("plan_execute_result", {})
                    if plan_execute_result:
                        plan = json.loads(
                            plan_execute_result.get("plan", {}).model_dump_json()
                        )
                        index = str(output.get("retry_count", 0)) + ":" + "Verify"
                        yield _emit(
                            "plan",
                            index=index,
                            reasoning=plan.get("reasoning", ""),
                            code=json.dumps(
                                plan.get("steps", []), indent=2, ensure_ascii=False
                            ),
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
    uvicorn.run(
        "server:app", host="127.0.0.1", port=8000, reload=True, log_level="info"
    )
