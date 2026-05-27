from langchain_core.runnables import RunnableConfig
from typing import Any

from schemas import GraphState, OrchestrationDecision
from utilities import call_llm


def orchestrate_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    prompt = f"""
Question:
{state["question"]}
"""

    decision: OrchestrationDecision = call_llm(
        system_prompt="",
        user_prompt=prompt,
        who="orchestrate",
        response_model=OrchestrationDecision,
    )

    state["csv_paths"].sort()

    return {
        "csv_paths": state["csv_paths"],
        "orchestration_decision": decision,
        "retry_count": 0,
    }
