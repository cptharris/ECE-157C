from typing import Dict, Any
import json
from .call_llm import call_llm


SYSTEM_PROMPT = """You are a clear, concise data analyst. Answer only from provided data. Be specific, under 150 words, and do not mention code."""

USER_PROMPT = """
Question:
{question}

Captured data:
{data}
"""


def respond_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state["response"] = call_llm(
        SYSTEM_PROMPT,
        USER_PROMPT.format(
            question=state["plan"]["question"],
            data=json.dumps(state["execution"]["data"], indent=None)[:3000],
        ),
    )

    return state
