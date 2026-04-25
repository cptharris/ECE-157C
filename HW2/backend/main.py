"""
backend/main.py
---------------
FastAPI backend for the HW2 interactive data analysis agent.

Endpoints:
  POST /chat          - send a question, get answer + optional viz
  POST /reset         - clear conversation memory for a session
  GET  /history       - return the turn history for a session
  GET  /datasets      - list available CSV files in ../datasets/

Sessions are stored in-process (keyed by session_id).
For a real deployment you'd persist to Redis/DB, but this is enough for demo.
"""

import os
import uuid
import glob
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add parent dir to path so we can import agent/nodes
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import run_turn, ConversationMemory


app = FastAPI(title="ECE 157C HW2 — Data Analysis Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the frontend directory as static files
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ---------------------------------------------------------------------------
# In-process session store
# ---------------------------------------------------------------------------

# session_id → {"memory": ConversationMemory, "csv_path": str}
_sessions: dict[str, dict] = {}

DATASETS_DIR = Path(__file__).parent.parent / "datasets"


def get_or_create_session(session_id: str, csv_path: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "memory": ConversationMemory(),
            "csv_path": csv_path,
        }
    else:
        # Allow switching datasets mid-session by resetting memory
        if _sessions[session_id]["csv_path"] != csv_path:
            _sessions[session_id] = {
                "memory": ConversationMemory(),
                "csv_path": csv_path,
            }
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    session_id: Optional[str] = None  # if None, a new session is created
    question: str
    csv_path: str  # relative path under datasets/
    is_followup: bool = False


class ChatResponse(BaseModel):
    session_id: str
    final_answer: str
    viz_json: Optional[str] = None
    generated_code: str
    evaluation: str
    turn_index: int


class ResetRequest(BaseModel):
    session_id: str


class HistoryResponse(BaseModel):
    session_id: str
    turns: list[dict]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def serve_frontend():
    index = frontend_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Frontend not found. Place index.html in frontend/"}


@app.get("/datasets")
def list_datasets():
    """Return the list of CSV files available under datasets/."""
    if not DATASETS_DIR.exists():
        return {"datasets": []}
    csvs = sorted(DATASETS_DIR.glob("**/*.csv"))
    return {"datasets": [str(p.relative_to(DATASETS_DIR)) for p in csvs]}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Resolve CSV path
    csv_path = str(DATASETS_DIR / req.csv_path)
    if not Path(csv_path).exists():
        # Allow absolute paths too (for testing)
        if Path(req.csv_path).exists():
            csv_path = req.csv_path
        else:
            raise HTTPException(
                status_code=404, detail=f"CSV not found: {req.csv_path}"
            )

    # Session management
    session_id = req.session_id or str(uuid.uuid4())
    session = get_or_create_session(session_id, csv_path)
    memory: ConversationMemory = session["memory"]

    # Run one agent turn
    result = run_turn(
        question=req.question,
        csv_path=csv_path,
        memory=memory,
        is_followup=req.is_followup,
    )

    return ChatResponse(
        session_id=session_id,
        final_answer=result["final_answer"],
        viz_json=result.get("viz_json"),
        generated_code=result.get("generated_code", ""),
        evaluation=result.get("evaluation", ""),
        turn_index=len(memory.turns) - 1,
    )


@app.post("/reset")
def reset_session(req: ResetRequest):
    if req.session_id in _sessions:
        _sessions.pop(req.session_id)
    return {"status": "ok", "session_id": req.session_id}


@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    memory: ConversationMemory = _sessions[session_id]["memory"]
    turns = [
        {"question": t["question"], "final_answer": t["final_answer"]}
        for t in memory.turns
    ]
    return HistoryResponse(session_id=session_id, turns=turns)


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=os.path.dirname(os.path.realpath(__file__)) + "/..",
    )
