from .call_llm import call_llm
from typing import Dict, Any


SYSTEM_PROMPT = """
You are a data analysis reporting assistant.

You explain results clearly and concisely.

Do not fabricate numbers.
Only use provided data.
"""


def respond_node(state: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = f"""
Question:
{state["input_question"]}

Result:
{state["execution"].get("result")}
"""

    response = call_llm(SYSTEM_PROMPT, user_prompt)

    state["final_answer"] = response
    return state
