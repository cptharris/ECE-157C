from .call_llm import call_llm
from typing import Dict, Any


SYSTEM_PROMPT = """You are a Python data analysis code generator.

You write correct, executable pandas code.

Rules:
- You are given a dataframe called `df`
- You MUST return a variable named `result`
- result should be a dict or dataframe summary (not print statements)
- return enough information to answer follow-up questions
- Do NOT include explanations
- Output ONLY Python code
"""


def codegen_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state["step"] = "codegen"

    user_prompt = f"""
Recent conversation context:
{state.get("recent_context", "None")}

Current user question:
{state["input_question"]}

Columns and data types (with {state["dataset_context"]["row_count"]} rows):
{state["dataset_context"]["column_types"]}

Write Python code to answer the current question while using prior context when relevant.
"""

    code = call_llm(SYSTEM_PROMPT, user_prompt)

    state["code"] = code

    return state
