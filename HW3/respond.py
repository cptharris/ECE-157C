"""
respond.py
"""

from schemas import *
import json
from call_llm import call_llm


SYSTEM_PROMPT = """\
You are the Analyzer for a structured data-analysis agent.
Your sole job is to use captured data to answer the question.
Do not make any claims that are not directly supported by the captured data.
Be clear, concise, and specific. Answer in under 150 words and do not mention code.
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
