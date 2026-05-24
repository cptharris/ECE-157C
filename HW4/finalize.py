from langchain_core.runnables import RunnableConfig
from typing import Any

from schemas import GraphState, GraphOutput, GenericResult, AnalyticsResult


def finalize_node(
    state: GraphState,
    config: RunnableConfig,
) -> GraphOutput:
    if state["agent_type"] == "generic":
        generic_result: GenericResult = state["generic_result"]

        return GraphOutput(
            answer=generic_result["response"],
            final_plots=[],
        )

    analytics_result: AnalyticsResult = state["analytics_result"]

    return GraphOutput(
        answer=analytics_result["final_answer"],
        final_plots=analytics_result["plots"],
    )
