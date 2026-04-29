from typing import Dict, Any, List
from pydantic import BaseModel, Field
import uuid
import os
import json


MEMORY_DIR = "memory"
os.makedirs(MEMORY_DIR, exist_ok=True)


class SessionMemory(BaseModel):
    # One session == one dataset-specific conversation thread
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset_name: str | None = None
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)


# In-memory cache of session objects
MEMORY_STORE: Dict[str, SessionMemory] = {}


def _session_path(session_id: str) -> str:
    return os.path.join(MEMORY_DIR, f"{session_id}.json")


def save_memory(session: SessionMemory):
    with open(_session_path(session.session_id), "w", encoding="utf-8") as f:
        json.dump(session.model_dump(), f, indent=2, ensure_ascii=False)


def load_memory(session_id: str) -> SessionMemory:
    path = _session_path(session_id)

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SessionMemory(**data)

    return SessionMemory(session_id=session_id)


def get_memory(session_id: str) -> SessionMemory:
    if session_id not in MEMORY_STORE:
        MEMORY_STORE[session_id] = load_memory(session_id)
    return MEMORY_STORE[session_id]


def update_memory(session_id: str, artifact: dict):
    mem = get_memory(session_id)
    mem.artifacts.append(artifact)
    MEMORY_STORE[session_id] = mem
    save_memory(mem)


def get_previous_artifact(session_id: str):
    mem = get_memory(session_id)
    if not mem.artifacts:
        return None
    return mem.artifacts[-1]


def build_recent_context(session_id: str, max_turns: int = 3) -> str:
    mem = get_memory(session_id)

    if not mem.artifacts:
        return ""

    recent = mem.artifacts[-max_turns:]

    lines = []
    for i, run in enumerate(recent, start=1):
        lines.append(
            f"Turn {i}:\nQuestion: {run.get('metadata').get('prompt')}\nAnswer: {run.get('response')}"
        )

    return "\n\n".join(lines)
