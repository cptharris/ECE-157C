from typing import Dict, Any, List
from pydantic import BaseModel, Field
import uuid


class SessionMemory(BaseModel):
    # One session == one dataset-specific conversation thread
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    dataset_name: str | None = None

    chat_history: List[Dict[str, Any]] = Field(default_factory=list)

    artifacts: List[Dict[str, Any]] = Field(default_factory=list)


def build_recent_context(session_id: str, max_turns: int = 3) -> str:
    mem = get_memory(session_id)

    if not mem.chat_history:
        return ""

    recent = mem.chat_history[-max_turns:]

    lines = []
    for i, turn in enumerate(recent, start=1):
        lines.append(
            f"Turn {i}:\nQuestion: {turn.get('question')}\nAnswer: {turn.get('answer')}"
        )

    return "\n\n".join(lines)


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
