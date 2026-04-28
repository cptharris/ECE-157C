from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import uuid
from fastapi.responses import FileResponse

from agent import agent
from nodes.memory import get_memory, update_memory, build_recent_context, MEMORY_STORE

from pprint import pprint

from utils import make_json_safe


# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="Dataset Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Request schema
# -----------------------------
class QueryRequest(BaseModel):
    session_id: str
    question: str
    dataset_name: str


# -----------------------------
# Dataset listing endpoint
# -----------------------------
@app.get("/datasets")
def list_datasets():
    """
    Returns CSV files inside backend/datasets
    """
    dataset_dir = "datasets"

    if not os.path.exists(dataset_dir):
        return []

    files = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]

    return files


# -----------------------------
# Main agent endpoint
# -----------------------------
@app.post("/query")
def query(req: QueryRequest):
    """
    Runs full LangGraph pipeline:
    summarize → codegen → execute → visualize → respond
    """

    # -----------------------------
    # Memory
    # -----------------------------
    memory = get_memory(req.session_id)

    # -----------------------------
    # Build initial artifact
    # -----------------------------
    artifact = {
        "session_id": req.session_id,
        "run_id": str(uuid.uuid4()),
        "step": "start",
        "input_question": req.question,
        "dataset_name": req.dataset_name,
        "recent_context": "",
        "dataset_context": {},
        "code": "",
        "execution": {},
        "visualization": {"should_visualize": False, "reason": "", "plotly_code": ""},
        "final_answer": None,
    }

    # Enforce one dataset per session and inject recent conversational context
    if memory.dataset_name is None:
        memory.dataset_name = req.dataset_name
    elif memory.dataset_name != req.dataset_name:
        artifact["step"] = "error"
        artifact["final_answer"] = (
            "This session is already tied to a different dataset. "
            "Please start a new session for a new dataset."
        )
        artifact["visualization"] = {"should_visualize": False}
        return make_json_safe(artifact)

    artifact["recent_context"] = build_recent_context(req.session_id)

    # -----------------------------
    # Run LangGraph agent
    # -----------------------------
    try:
        result = agent.invoke(artifact)

        # Convert all numpy / pandas / non-JSON-safe objects
        result = make_json_safe(result)

    except Exception as e:
        artifact["step"] = "error"
        artifact["execution"] = {"result": None, "error": str(e)}
        artifact["final_answer"] = f"Agent execution failed: {str(e)}"
        artifact["visualization"] = {"should_visualize": False, "error": str(e)}
        return make_json_safe(artifact)

    # -----------------------------
    # Update memory
    # -----------------------------

    update_memory(req.session_id, result)

    # -----------------------------
    # Response payload
    # -----------------------------
    return result


# -----------------------------
# Artifact retrieval endpoints
# -----------------------------
@app.get("/runs/{run_id}")
def get_run(run_id: str):
    for session_memory in MEMORY_STORE.values():
        for artifact in session_memory.artifacts:
            if artifact.get("run_id") == run_id:
                return artifact
    return {"error": "run_id not found"}


@app.get("/sessions/{session_id}/runs")
def get_session_runs(session_id: str):
    memory = get_memory(session_id)
    return memory.artifacts


# -----------------------------
# Frontend entrypoint
# -----------------------------
@app.get("/")
def serve_frontend():
    frontend_path = os.path.join(
        os.path.dirname(os.path.abspath(".")), "frontend/index.html"
    )
    return FileResponse(frontend_path)


# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=os.path.dirname(os.path.realpath(os.path.abspath("."))),
    )
