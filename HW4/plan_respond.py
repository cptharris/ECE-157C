"""
respond.py
"""

from langchain_core.runnables import RunnableConfig
from typing import Any

from schemas import GraphState, PlanExecuteFinalAnswer
import json
from utilities import call_llm


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

Plan description:
{description}

Captured data:
{data}
"""


def plan_respond_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    result: PlanExecuteResult = state["plan_execute_result"]

    answer: PlanExecuteFinalAnswer = call_llm(
        SYSTEM_PROMPT,
        USER_PROMPT.format(
            question=state["question"],
            description=result["plan"].description,
            data=json.dumps(result["execution_result"], indent=None),
        ),
        PlanExecuteFinalAnswer,
    )

    updated: PlanExecuteResult = {
        **result,
        "final_answer": answer.final_answer,
    }

    return {
        "plan_execute_result": updated
    }
