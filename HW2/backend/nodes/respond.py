from .call_llm import call_llm
from typing import Dict, Any


SYSTEM_PROMPT = """You are a clear, concise data analyst.
Given a question, the raw execution result from a pandas analysis, and a
summary of the dataset, write a final human-readable answer.

Rules:
- Answer only from the execution result — do not invent numbers.
- Be specific: include key values, percentages, or rankings.
- Keep the answer under 150 words.
- Do not mention Python, pandas, or code.
"""


def respond_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state["step"] = "respond"

    user_prompt = f"""
Recent conversation context:
{state.get("recent_context", "None")}

Current Question:
{state["input_question"]}

Result:
{state["execution"].get("result")}
"""

    response = call_llm(SYSTEM_PROMPT, user_prompt)

    state["final_answer"] = response
    return state
