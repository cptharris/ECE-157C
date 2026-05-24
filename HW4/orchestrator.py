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

CSV paths:
{state["csv_paths"]}
"""

    decision: OrchestrationDecision = call_llm("", prompt, OrchestrationDecision)

    return {
        "agent_type": decision.agent_type,
    }
