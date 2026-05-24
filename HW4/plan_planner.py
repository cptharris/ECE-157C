"""
planner.py
==========
LLM-driven planning node for the deterministic plan-and-execute sub-agent.
"""

from langchain_core.runnables import RunnableConfig
from typing import Any

from schemas import PlanToExecute, GraphState
from utilities import call_llm


PLANNER_SYSTEM_PROMPT = """\
You are the planning node of a deterministic data analytics agent.

Your responsibility is to convert a natural-language analytics question into
an ordered sequence of strictly-typed execution Steps.

Execution constraints
---------------------
- Every operation must be expressed using the provided Step schemas.
- Do not write Python code.
- Plans must be deterministic and executable without ambiguity.
- Prefer the smallest correct sequence of Steps.
- Use filter_rows before aggregation whenever possible.
- Use derive_columns for arithmetic computations.
- Use snapshot / restore / join when branching intermediate computations.
- Use group_aggregate for all grouped statistics.
- Use sort_rows + limit_rows for ranking or top-k queries.
- The final DataFrame produced by the plan should directly contain the answer.
- Start by loading a DataFrame 

Dataset assumptions
-------------------
- The runtime loads one active pandas DataFrame at a time—start by loading a DataFrame.
- Column names must exactly match the dataset schema provided in the prompt.
- Do not invent columns.

Output requirements
-------------------
Return a valid PlanToExecute object.
- reasoning must explain the intended computation.
- steps must contain only valid Step objects.
- description must describe what the final execution result represents.
"""


def plan_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:

    plan = call_llm(
        system_prompt=PLANNER_SYSTEM_PROMPT,
        user_prompt=f"""\
Question:
{state["question"]}

Dataset Specifications:
{state["dataset_schema"]}
    """,
        response_model=PlanToExecute,
    )

    partial: PlanExecuteResult = {
        "plan": plan,
        "trace": [],
        "execution_result": "",
        "final_answer": "",
    }

    return {
        "plan_execute_result": partial,
    }
