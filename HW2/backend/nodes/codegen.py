from .call_llm import call_llm
from typing import Dict, Any


SYSTEM_PROMPT = """You are a Python data analysis code generator.

You write correct, executable pandas code.

Rules:
- You are given a dataframe called `df`
- You MUST return a variable named `result`
- result should be a dict or dataframe summary (not print statements)
- Do NOT include explanations
- Output ONLY Python code
"""


def codegen_node(state: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = f"""
User question:
{state["input_question"]}

Columns:
{state["context"]["columns"]}

Types:
{state["context"]["column_types"]}

Sample rows:
{state["context"]["sample_rows"]}

Write pandas code to answer the question.
"""

    code = call_llm(SYSTEM_PROMPT, user_prompt)

    state["code"] = {
        "language": "python",
        "snippet": code
    }

    return state
