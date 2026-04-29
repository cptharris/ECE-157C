from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import uuid
from fastapi.responses import FileResponse
import json

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
# Session listing endpoint (lazy sidebar population)
# -----------------------------
@app.get("/sessions")
def list_sessions():
    """
    Returns available session IDs by scanning memory/*.json
    Does NOT load full session content.
    """
    memory_dir = "memory"

    if not os.path.exists(memory_dir):
        return []

    sessions = []

    for file in os.listdir(memory_dir):
        if file.endswith(".json"):
            sessions.append(file.replace(".json", ""))

    sessions.sort(reverse=True)
    return sessions


# -----------------------------
# Full session load endpoint
# -----------------------------
@app.get("/sessions/{session_id}")
def load_session(session_id: str):
    """
    Loads full session only when selected.
    """
    path = os.path.join("memory", f"{session_id}.json")

    if not os.path.exists(path):
        return {"error": "session not found"}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Main agent endpoint
# -----------------------------
@app.post("/query")
def query(req: QueryRequest):
    """
    Runs full LangGraph pipeline
    """

    # -----------------------------
    # Memory
    # -----------------------------
    memory = get_memory(req.session_id)

    # -----------------------------
    # Build initial artifact
    # -----------------------------
    artifact = {
        "metadata": {
            "session_id": req.session_id,
            "run_id": str(uuid.uuid4()),
            "prompt": req.question,
            "dataset_name": req.dataset_name,
            "status": "running",
        },
        "recent_context": None,
        "previous": {"data": {}, "data_desc": None},
        "dataset_desc": None,
        "plan": {
            "is_follow_up": False,
            "do_response": True,
            "do_vis": False,
            "question": None,
            "viz_spec": None,
            "data_spec": None,
        },
        "execution": {"data_code": None, "data": {}, "data_desc": None, "error": None},
        "vis": {"vis_code": "", "fig": {}, "error": None},
        "response": None,
    }

    # Enforce one dataset per session
    if memory.dataset_name is None:
        memory.dataset_name = req.dataset_name
    elif memory.dataset_name != req.dataset_name:
        artifact["metadata"]["status"] = "error"
        artifact["response"] = (
            "This session is already tied to a different dataset. "
            "Please start a new session for a new dataset."
        )
        return make_json_safe(artifact)

    # -----------------------------
    # Run LangGraph agent
    # -----------------------------
    try:
        result = agent.invoke(artifact)
        result["metadata"]["status"] = "complete"

        # Convert all numpy / pandas / non-JSON-safe objects
        result = make_json_safe(result)

    except Exception as e:
        artifact["metadata"]["status"] = "error"
        artifact["response"] = f"Agent execution failed: {str(e)}"
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
# Run server
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=os.path.dirname(os.path.realpath(os.path.abspath("."))),
        reload_excludes="**/memory/*, llm_cache.json"
    )
