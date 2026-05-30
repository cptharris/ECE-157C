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
import re
from typing import AsyncIterator
import plotly.graph_objects as go

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


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
REPORT_DIR = Path("report-data")
REPORT_DIR.mkdir(exist_ok=True)


def convertName(question: str) -> str:
    return re.sub(r"[^\w,\d]+", "-", question.lower().strip()).strip("-").strip()


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
    retry_count = 0
    question_dir = REPORT_DIR / convertName(question)
    question_dir.mkdir(exist_ok=True)
    (question_dir / "input.json").write_text(
        json.dumps(GraphInput(question=question, csv_paths=csv_files), indent=2)
    )

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
            if name == "retry" and kind == "on_chain_start":
                retry_count += 1

            if kind != "on_chain_stream":
                print(
                    f"{bcolors.HEADER}{"="*5} {retry_count} {name.center(28, " ")}: {kind.replace("on_chain_", "").center(10, " ")} {"="*5}{bcolors.ENDC}"
                )
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

                # Emit validation verdict.
                if name == "validate":
                    vr = output.get("validation_result")
                    if vr:
                        yield _emit(
                            "validation",
                            verdict=vr.verdict,
                            reasoning=vr.reasoning,
                        )
                        (question_dir / f"validation-{retry_count}.json").write_text(
                            vr.model_dump_json(indent=2)
                        )

                if name == "analytics_step":
                    steps: list = output.get("steps", [])
                    if steps:
                        latest = steps[-1]
                        index = (
                            str(retry_count) + "." + str(latest.get("step_index", 0))
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
                        (question_dir / f"analytics-{index}.json").write_text(
                            json.dumps(latest, indent=2)
                        )

                if name == "plan_execute":
                    plan_execute_result = output.get("plan_execute_result", {})
                    if plan_execute_result:
                        plan = json.loads(
                            plan_execute_result.get("plan", {}).model_dump_json()
                        )
                        index = str(retry_count) + ":" + "Verify"
                        steps = plan.get("steps", [])
                        for trace in plan_execute_result.get("trace", {}):
                            steps[trace.step_index]["step_index"] = str(
                                trace.step_index
                            )
                            steps[trace.step_index]["input_shape"] = str(
                                trace.input_shape
                            )
                            steps[trace.step_index]["output_shape"] = str(
                                trace.output_shape
                            )
                            steps[trace.step_index]["error"] = trace.error

                        yield _emit(
                            "plan",
                            index=index,
                            reasoning=plan.get("reasoning", ""),
                            code=json.dumps(steps, indent=2, ensure_ascii=False),
                        )
                        yield _emit(
                            "plan_result",
                            index=index,
                            stdout=plan_execute_result.get("execution_result", ""),
                            error="",
                        )
                        plan["steps"] = steps
                        (question_dir / f"plan_execute-{retry_count}.json").write_text(
                            json.dumps(plan, indent=2)
                        )

                # Emit final answer + final plots from finalize_node.
                if name == "finalize":
                    answer = output.get("answer", "")
                    if answer:
                        yield _emit("final_answer", answer=answer)
                        (question_dir / "final_answer.md").write_text(answer)
                    already_plotted = []
                    for plot in output.get("final_plots", []):
                        if plot not in already_plotted:
                            yield _emit(
                                "plot",
                                title=plot["title"],
                                figure_json=plot["figure_json"],
                            )
                            (
                                question_dir / f"plot-{convertName(plot["title"])}.json"
                            ).write_text(json.dumps(plot["figure_json"], indent=2))
                            fig = go.Figure(plot["figure_json"])
                            fig.write_html(
                                question_dir / f"plot-{convertName(plot["title"])}.html"
                            )
                            fig.write_image(
                                format="png",
                                scale=2,
                                file=(
                                    question_dir
                                    / f"plot-{convertName(plot["title"])}.png"
                                ),
                            )
                        already_plotted.append(plot)

                if name == "analytics_answer":
                    result = output.get("analytics_result", {})
                    if result:
                        (
                            question_dir / f"analytics_answer-{retry_count}.md"
                        ).write_text(result["final_answer"])
                        (question_dir / f"analytics_code-{retry_count}.py").write_text(
                            result["overall_plan"]
                        )
                if name == "plan_respond":
                    result = output.get("plan_execute_result", {})
                    if result:
                        (question_dir / f"plan_answer-{retry_count}.md").write_text(
                            result["final_answer"]
                        )
                if name == "generic_search":
                    result = output.get("generic_result", {}).get("search_result", {})
                    if result:
                        (question_dir / f"generic-{retry_count}.json").write_text(
                            json.dumps(result, indent=2)
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
    finally:
        print(f"{bcolors.HEADER}{"="*10} DONE {"="*10}{bcolors.ENDC}")


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
