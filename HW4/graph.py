"""
graph.py
========
LangGraph StateGraph implementation for the analytics agent architecture
defined in schemas.py, with LangSmith tracing / visualization support.

Requirements
------------
pip install langgraph langchain langsmith pydantic

Environment
-----------
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=...
export LANGCHAIN_PROJECT="analytics-agent"

Optional:
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"

Notes
-----
- This file focuses on graph construction and state transitions.
- LLM/tool internals are intentionally stubbed.
- Every node reads/writes the shared GraphState.
- Reducer fields (steps, plots) are append-only accumulators.
"""

from typing import Any, Literal, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from utilities import call_llm

# ---------------------------------------------------------------------------
# Schemas and Nodes
# ---------------------------------------------------------------------------


from schemas import (
    MAX_RETRY_CYCLES,
    MAX_ANALYTICS_STEPS,
    GraphInput,
    GraphOutput,
    GraphState,
)


# ---------------------------------------------------------------------------
# LangSmith configuration
# ---------------------------------------------------------------------------

# LangSmith tracing is enabled automatically when:
#
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_API_KEY=...
#
# are present in the environment.
#
# The compiled graph will appear visually in LangSmith traces.
#
# Optional per-run metadata can also be passed via RunnableConfig:
#
# graph.invoke(
#     input,
#     config=RunnableConfig(
#         tags=["analytics-agent"],
#         metadata={"user_id": "123"}
#     )
# )


# ---------------------------------------------------------------------------
# Routing Helpers
# ---------------------------------------------------------------------------


def agent_task_router(
    state: GraphState,
) -> Literal["analytics", "generic"]:
    return cast(
        Literal["analytics", "generic"], state["orchestration_decision"].agent_type
    )


def analytics_continue_router(
    state: GraphState,
) -> Literal["continue", "answer"]:
    if state["is_complete"]:
        return "answer"

    if state["current_step_index"] >= MAX_ANALYTICS_STEPS:
        return "answer"

    return "continue"


def validation_router(
    state: GraphState,
) -> Literal["approved", "retry"]:
    if state["validation_result"].verdict == "approved":
        return "approved"

    if state["retry_count"] >= MAX_RETRY_CYCLES:
        return "approved"

    return "retry"


# /------------------------------------------------------------------------------------------------------------\
# |   ____  _    _ _____ _      _____      _____ _    _ ____   _____ _____       __     _____  _    _  _____   |
# |  |  _ \| |  | |_   _| |    |  __ \    / ____| |  | |  _ \ / ____|  __ \     /  \   |  __ \| |  | |/ ____|  |
# |  | |_) | |  | | | | | |    | |  | |  | (___ | |  | | |_) | |  __| |__) |   /    \  | |__) | |__| | (___    |
# |  |  _ <| |  | | | | | |    | |  | |   \___ \| |  | |  _ <| | |_ |  _  /   /  /\  \ |  ___/|  __  |\___ \   |
# |  | |_) | |__| |_| |_| |____| |__| |   ____) | |__| | |_) | |__| | | \ \  /  ____  \| |    | |  | |____) |  |
# |  |____/ \____/|_____|______|_____/   |_____/ \____/|____/ \_____|_|  \_\/__/    \__\_|    |_|  |_|_____/   |
# \------------------------------------------------------------------------------------------------------------/


# ---------------------------------------------------------------------------
# Build analytics subgraph
# ---------------------------------------------------------------------------


from analytics_agent import (
    analytics_init_node,
    analytics_step_node,
    analytics_answer_node,
)

analytics_builder = StateGraph(GraphState)

analytics_builder.add_node("analytics_init", analytics_init_node)
analytics_builder.add_node("analytics_step", analytics_step_node)
analytics_builder.add_node("analytics_answer", analytics_answer_node)

analytics_builder.add_edge(START, "analytics_init")
analytics_builder.add_edge("analytics_init", "analytics_step")

analytics_builder.add_conditional_edges(
    "analytics_step",
    analytics_continue_router,
    {
        "continue": "analytics_step",
        "answer": "analytics_answer",
    },
)

analytics_builder.add_edge("analytics_answer", END)

analytics_subgraph = analytics_builder.compile()


# ---------------------------------------------------------------------------
# Build plan-execute subgraph
# ---------------------------------------------------------------------------

from plan_planner import plan_node
from plan_executor import plan_execute_node
from plan_respond import plan_respond_node

plan_execute_builder = StateGraph(GraphState)

plan_execute_builder.add_node("plan", plan_node)
plan_execute_builder.add_node("plan_execute", plan_execute_node)
plan_execute_builder.add_node("plan_respond", plan_respond_node)

plan_execute_builder.add_edge(START, "plan")
plan_execute_builder.add_edge("plan", "plan_execute")
plan_execute_builder.add_edge("plan_execute", "plan_respond")
plan_execute_builder.add_edge("plan_respond", END)

plan_execute_subgraph = plan_execute_builder.compile()


# ---------------------------------------------------------------------------
# Build generic subgraph
# ---------------------------------------------------------------------------


from generic_search import (
    generic_search_node,
    generic_respond_node,
)

generic_builder = StateGraph(GraphState)

generic_builder.add_node("generic_search", generic_search_node)
generic_builder.add_node("generic_respond", generic_respond_node)

generic_builder.add_edge(START, "generic_search")
generic_builder.add_edge("generic_search", "generic_respond")
generic_builder.add_edge("generic_respond", END)

generic_subgraph = generic_builder.compile()


# /------------------------------------------------------------------------------------------------------------------\
# |   ____  _    _ _____ _      _____     __  __     __     _____ _   _     _____ _____       __     _____  _    _   |
# |  |  _ \| |  | |_   _| |    |  __ \   |  \/  |   /  \   |_   _| \ | |   / ____|  __ \     /  \   |  __ \| |  | |  |
# |  | |_) | |  | | | | | |    | |  | |  | \  / |  /    \    | | |  \| |  | |  __| |__) |   /    \  | |__) | |__| |  |
# |  |  _ <| |  | | | | | |    | |  | |  | |\/| | /  /\  \   | | | . ` |  | | |_ |  _  /   /  /\  \ |  ___/|  __  |  |
# |  | |_) | |__| |_| |_| |____| |__| |  | |  | |/  ____  \ _| |_| |\  |  | |__| | | \ \  /  ____  \| |    | |  | |  |
# |  |____/ \____/|_____|______|_____/   |_|  |_/__/    \__\_____|_| \_|   \_____|_|  \_\/__/    \__\_|    |_|  |_|  |
# \------------------------------------------------------------------------------------------------------------------/

# ---------------------------------------------------------------------------
# Build main graph
# ---------------------------------------------------------------------------


from orchestrator import orchestrate_node
from dataset_node import dataset_node
from validator import validation_node, retry_node
from finalize import finalize_node

builder = StateGraph(GraphState, input_schema=GraphInput, output_schema=GraphOutput)

# Main nodes
builder.add_node("orchestrate", orchestrate_node)
builder.add_node("dataset_fan", dataset_node)
builder.add_node("analytics", analytics_subgraph)
builder.add_node("plan_execute", plan_execute_subgraph)
builder.add_node("validate", validation_node)
builder.add_node("retry", retry_node)
builder.add_node("generic", generic_subgraph)
builder.add_node("finalize", finalize_node)

# Start
builder.add_edge(START, "orchestrate")

# Route generic requests directly
# Analytics requests enter the dataset fan-out node
builder.add_conditional_edges(
    "orchestrate",
    agent_task_router,
    {
        "generic": "generic",
        "analytics": "dataset_fan",
    },
)

# Dataset node fans out into both analytics subgraphs
builder.add_edge("dataset_fan", "analytics")
# builder.add_edge("dataset_fan", "plan_execute")

# Validation waits for both branches to complete
builder.add_edge("analytics", "plan_execute")
builder.add_edge("plan_execute", "validate")

builder.add_conditional_edges(
    "validate",
    validation_router,
    {
        "approved": "finalize",
        "retry": "retry",
    },
)

builder.add_edge("retry", "analytics")

# Generic path
builder.add_edge("generic", "finalize")

# Finish
builder.add_edge("finalize", END)

# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

# In-memory checkpointing for local development.
# Replace with Redis/Postgres/etc in production.
# checkpointer = MemorySaver()

# ---------------------------------------------------------------------------
# Compile graph
# ---------------------------------------------------------------------------

graph = builder.compile(
    # checkpointer=checkpointer,
)

# ---------------------------------------------------------------------------
# Mermaid visualization
# ---------------------------------------------------------------------------

# Save graph visualization:
#
# with open("graph_mermaid.md", "w") as f:
#     f.write("```mermaid\n")
#     f.write(graph.get_graph(xray=True).draw_mermaid())
#     f.write("```")
#
# Or:
#
# png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
# with open("graph.png", "wb") as f:
#     f.write(png_bytes)

# ---------------------------------------------------------------------------
# Example invocation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = graph.invoke(
        GraphInput(
            question="Find undervalued technology companies from the dataset.",
            # question="Which individual stocks had the top 10 highest and lowest price returns in each year?",
            csv_paths=[
                "2017_Financial_Data.csv",
                "2018_Financial_Data.csv",
            ],
        ),
        config=RunnableConfig(
            tags=["analytics-agent"],
            metadata={
                "environment": "dev",
            },
            thread_id="thread-1",
        ),
    )

    print(result)
