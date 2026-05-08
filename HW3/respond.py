"""
respond.py
"""

from schemas import *
import json
from call_llm import call_llm


SYSTEM_PROMPT = """\
You are the Analyzer for a structured data-analysis agent.
Your sole job is to use captured data to answer the question.

Follow these rules:
1) Do not make any claims that are not directly supported by the captured data.
2) Assume the captured data is the answer to the question. Infer units and round where appropriate.
3) Answer in under 150 words. Do not mention code or "the captured data".
4) Be professional, clear, concise, and specific. Use natural language.
"""

USER_PROMPT = """\
Question:
{question}

Captured data:
{data}
"""


def respond_node(state: AgentState) -> AgentState:
    state["final_answer"] = call_llm(
        SYSTEM_PROMPT,
        USER_PROMPT.format(
            question=state["question"],
            data=json.dumps(state["execution_result"], indent=None)[:10000],
        ),
    )

    return state
