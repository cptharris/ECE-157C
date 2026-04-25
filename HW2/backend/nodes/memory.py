from typing import Dict, Any, List
from pydantic import BaseModel, Field
import uuid


class SessionMemory(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    chat_history: List[Dict[str, Any]] = []

    datasets_used: List[str] = []

    artifacts: List[Dict[str, Any]] = []

    last_question: str | None = None

    last_dataset: str | None = None


# in-memory store (swap with Redis later if needed)
MEMORY_STORE: Dict[str, SessionMemory] = {}

def get_memory(session_id: str) -> SessionMemory:
    if session_id not in MEMORY_STORE:
        MEMORY_STORE[session_id] = SessionMemory(session_id=session_id)
    return MEMORY_STORE[session_id]


def update_memory(session_id: str, artifact: dict):
    mem = get_memory(session_id)
    mem.artifacts.append(artifact)
    MEMORY_STORE[session_id] = mem
