from typing import Dict, Any
from .call_llm import call_llm


CODEGEN_SYSTEM_PROMPT = """You are a Python data analysis code generator.
Rules:
- You are given a dataframe called df
- You MUST return a variable named data
- data must be JSON-safe dict or list of dicts
- do not include raw values
- do NOT make visualizations or figures
- Do NOT include explanations
- Output ONLY Python code
"""

CODEGEN_USER_PROMPT = """
Question: {question}
Data specification: {data_spec}
Dataset description:
{dataset_desc}
"""


def codegen_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state["execution"]["data_code"] = call_llm(
        CODEGEN_SYSTEM_PROMPT,
        CODEGEN_USER_PROMPT.format(
            question=state["plan"]["question"],
            data_spec=state["plan"]["data_spec"],
            dataset_desc=state["dataset_desc"] if state["plan"]["is_follow_up"] is False else state["previous"]["data_desc"],
        ),
    )

    return state
