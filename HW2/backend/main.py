from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import uuid
from fastapi.responses import FileResponse

from agent import agent
from nodes.memory import get_memory, update_memory

from pprint import pprint

from utils import make_json_safe


# -----------------------------
# Artifact versioning store
# -----------------------------
RUN_STORE: dict[str, list[dict]] = {}
RUN_INDEX: dict[str, dict] = {}


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

    files = [
        f for f in os.listdir(dataset_dir)
        if f.endswith(".csv")
    ]

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
        "run_id": str(uuid.uuid4()),
        "input_question": req.question,
        "dataset_name": req.dataset_name,
        "dataset_context": {},
        "code": "",
        "execution": {},
        "visualization": {
            "should_visualize": False,
            "reason": "",
            "plotly_code": ""
        },
        "final_answer": None
    }

    # Store context in memory
    memory.last_question = req.question
    memory.last_dataset = req.dataset_name

    # -----------------------------
    # Run LangGraph agent
    # -----------------------------
    try:
        result = agent.invoke(artifact)
        print("\n\n===== RESULT =====")
        pprint(result)
        print("\n\n===== END =====")

        # Convert all numpy / pandas / non-JSON-safe objects
        result = make_json_safe(result)

        # -----------------------------
        # Artifact versioning
        # -----------------------------
        result_dict = result

        RUN_INDEX[result_dict["run_id"]] = result_dict

        if req.session_id not in RUN_STORE:
            RUN_STORE[req.session_id] = []

        RUN_STORE[req.session_id].append(result_dict)

    except Exception as e:
        return {
            "final_answer": f"Agent execution failed: {str(e)}",
            "artifact": make_json_safe(artifact),
            "visualization": {
                "should_visualize": False,
                "error": str(e)
            }
        }

    # -----------------------------
    # Update memory
    # -----------------------------
    update_memory(req.session_id, result_dict)

    memory.chat_history.append({
        "question": req.question,
        "answer": result["final_answer"],
        "dataset": req.dataset_name
    })

    # -----------------------------
    # Response payload
    # -----------------------------
    return {
        "run_id": result["run_id"],
        "final_answer": result["final_answer"],
        "visualization": result["visualization"],
        "run_history": make_json_safe(RUN_STORE.get(req.session_id, [])),
        "artifact": make_json_safe(result),
        "session_id": req.session_id
    }


# -----------------------------
# Artifact retrieval endpoints
# -----------------------------
@app.get("/runs/{run_id}")
def get_run(run_id: str):
    return RUN_INDEX.get(run_id, {"error": "run_id not found"})


@app.get("/sessions/{session_id}/runs")
def get_session_runs(session_id: str):
    return RUN_STORE.get(session_id, [])


# -----------------------------
# Frontend entrypoint
# -----------------------------
@app.get("/")
def serve_frontend():
    frontend_path = os.path.join(os.path.dirname(os.path.abspath('.')), "frontend/index.html")
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
        reload_dirs=os.path.dirname(os.path.realpath(os.path.abspath('.')))
    )
