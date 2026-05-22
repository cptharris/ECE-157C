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

from __future__ import annotations

import copy
from typing import Any, Literal, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from utilities.call_llm import call_llm
from utilities.call_ddg import call_ddg

from schemas import (
    MAX_ANALYTICS_STEPS,
    MAX_RETRY_CYCLES,
    PLOTLY_NAMESPACE_KEY,
    AnalyticsAction,
    AnalyticsFinalAnswer,
    AnalyticsResult,
    AnalyticsStep,
    GenericResponse,
    GenericResult,
    GraphInput,
    GraphOutput,
    GraphState,
    OrchestrationDecision,
    PlanExecuteFinalAnswer,
    PlanExecuteResult,
    PlanToExecute,
    Plot,
    SearchResult,
    ValidationDecision,
    ValidationResult,
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
# Stubbed external dependencies
# ---------------------------------------------------------------------------


def execute(code: str, namespace: dict[str, Any]) -> dict[str, Any]:
    """
    Stub for sandboxed Python execution.

    Must return ExecutionResult-compatible dict.
    """
    raise NotImplementedError


def execute_plan_steps(steps: list[Any]) -> tuple[list[Any], str]:
    """
    Stub for deterministic Step dispatcher.

    Returns:
        trace,
        execution_result
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _empty_plotly_namespace() -> dict[str, Any]:
    return {PLOTLY_NAMESPACE_KEY: {}}


def _harvest_plots(namespace: dict[str, Any]) -> list[Plot]:
    plots_dict = namespace.get(PLOTLY_NAMESPACE_KEY, {})

    harvested: list[Plot] = []

    for title, fig_json in plots_dict.items():
        harvested.append(
            Plot(
                title=title,
                figure_json=copy.deepcopy(fig_json),
            )
        )

    # Clear after harvest to avoid duplicate accumulation
    namespace[PLOTLY_NAMESPACE_KEY] = {}

    return harvested


# ---------------------------------------------------------------------------
# Orchestration node
# ---------------------------------------------------------------------------


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


def route_agent(
    state: GraphState,
) -> Literal["analytics", "generic"]:
    return cast(Literal["analytics", "generic"], state["agent_type"])


# ---------------------------------------------------------------------------
# Dataset fan-out node
# ---------------------------------------------------------------------------


def dataset_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """
    Dummy synchronization / fan-out node.

    Exists solely to make the top-level graph structure cleaner before
    branching into the analytics and plan-execute subgraphs.
    """
    return {}


# ---------------------------------------------------------------------------
# Analytics subgraph
# ---------------------------------------------------------------------------


def analytics_init_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    return {
        "namespace": _empty_plotly_namespace(),
        "current_step_index": 0,
        "is_complete": False,
    }


def analytics_step_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    prompt = f"""
Question:
{state["question"]}

Retry feedback:
{state.get("validation_feedback")}

Prior steps:
{state["steps"]}
"""

    action: AnalyticsAction = call_llm("", prompt, AnalyticsAction)

    execution_result = execute(
        action.code,
        state["namespace"],
    )

    harvested_plots = _harvest_plots(state["namespace"])

    step = AnalyticsStep(
        step_index=state["current_step_index"],
        reasoning=action.reasoning,
        code=action.code,
        execution=execution_result,
        plots_captured=[p["title"] for p in harvested_plots],
    )

    return {
        "steps": [step],
        "plots": harvested_plots,
        "namespace": state["namespace"],
        "current_step_index": state["current_step_index"] + 1,
        "is_complete": action.is_final_step,
    }


def analytics_continue_router(
    state: GraphState,
) -> Literal["continue", "answer"]:
    if state["is_complete"]:
        return "answer"

    if state["current_step_index"] >= MAX_ANALYTICS_STEPS:
        return "answer"

    return "continue"


def analytics_answer_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    prompt = f"""
Question:
{state["question"]}

Full step history:
{state["steps"]}
"""

    answer: AnalyticsFinalAnswer = call_llm("", prompt, AnalyticsFinalAnswer)

    result = AnalyticsResult(
        final_answer=answer.final_answer,
        plots=state["plots"],
        steps=state["steps"],
    )

    return {
        "analytics_result": result,
    }


# ---------------------------------------------------------------------------
# Plan-execute subgraph
# ---------------------------------------------------------------------------


def plan_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    prompt = f"""
Question:
{state["question"]}

CSV paths:
{state["csv_paths"]}
"""

    plan: PlanToExecute = call_llm("", prompt, PlanToExecute)

    partial: PlanExecuteResult = {
        "plan": plan,
        "trace": [],
        "execution_result": "",
        "final_answer": "",
    }

    return {
        "plan_execute_result": partial,
    }


def execute_plan_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    result = cast(PlanExecuteResult, state["plan_execute_result"])

    trace, execution_result = execute_plan_steps(result["plan"].steps)

    updated: PlanExecuteResult = {
        **result,
        "trace": trace,
        "execution_result": execution_result,
    }

    return {
        "plan_execute_result": updated,
    }


def plan_answer_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    result = cast(PlanExecuteResult, state["plan_execute_result"])

    prompt = f"""
Question:
{state["question"]}

Plan description:
{result["plan"].description}

Execution result:
{result["execution_result"]}
"""

    answer: PlanExecuteFinalAnswer = call_llm("", prompt, PlanExecuteFinalAnswer)

    updated: PlanExecuteResult = {
        **result,
        "final_answer": answer.final_answer,
    }

    return {
        "plan_execute_result": updated,
    }


# ---------------------------------------------------------------------------
# Validation node
# ---------------------------------------------------------------------------


def validation_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    analytics = state["analytics_result"]
    plan_execute = state["plan_execute_result"]

    prompt = f"""
Question:
{state["question"]}

Analytics answer:
{analytics}

Plan-execute answer:
{plan_execute}
"""

    decision: ValidationDecision = call_llm("", prompt, ValidationDecision)

    result = ValidationResult(
        decision=decision,
    )

    return {
        "validation_result": result,
    }


def validation_router(
    state: GraphState,
) -> Literal["approved", "retry"]:
    decision = state["validation_result"]["decision"]

    if decision.verdict == "approved":
        return "approved"

    if state["retry_count"] >= MAX_RETRY_CYCLES:
        return "approved"

    return "retry"


def retry_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    decision = state["validation_result"]["decision"]

    return {
        "retry_count": state["retry_count"] + 1,
        "validation_feedback": decision.feedback,
        "analytics_result": None,
        # Preserve the deterministic validator baseline across retries.
        "plan_execute_result": state["plan_execute_result"],
        "current_step_index": 0,
        "is_complete": False,
    }


# ---------------------------------------------------------------------------
# Generic subgraph
# ---------------------------------------------------------------------------


def generic_search_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    raw_text = call_ddg(state["question"])

    search_result = SearchResult(
        query=state["question"],
        raw_text=raw_text,
    )

    partial: GenericResult = {
        "search_result": search_result,
        "response": "",
    }

    return {
        "generic_result": partial,
    }


def generic_respond_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    generic_result = cast(GenericResult, state["generic_result"])

    prompt = f"""
Question:
{state["question"]}

Search results:
{generic_result["search_result"]["raw_text"]}
"""

    response: GenericResponse = call_llm("", prompt, GenericResponse)

    updated: GenericResult = {
        **generic_result,
        "response": response.response,
    }

    return {
        "generic_result": updated,
    }


# ---------------------------------------------------------------------------
# Finalize node
# ---------------------------------------------------------------------------


def finalize_node(
    state: GraphState,
    config: RunnableConfig,
) -> GraphOutput:
    if state["agent_type"] == "generic":
        generic_result = cast(GenericResult, state["generic_result"])

        return GraphOutput(
            answer=generic_result["response"],
            final_plots=[],
        )

    analytics_result = cast(
        AnalyticsResult,
        state["analytics_result"],
    )

    return GraphOutput(
        answer=analytics_result["final_answer"],
        final_plots=analytics_result["plots"],
    )


# ---------------------------------------------------------------------------
# Build analytics subgraph
# ---------------------------------------------------------------------------


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


plan_execute_builder = StateGraph(GraphState)

plan_execute_builder.add_node("plan", plan_node)
plan_execute_builder.add_node("execute_plan", execute_plan_node)
plan_execute_builder.add_node("plan_answer", plan_answer_node)

plan_execute_builder.add_edge(START, "plan")
plan_execute_builder.add_edge("plan", "execute_plan")
plan_execute_builder.add_edge("execute_plan", "plan_answer")
plan_execute_builder.add_edge("plan_answer", END)

plan_execute_subgraph = plan_execute_builder.compile()

# ---------------------------------------------------------------------------
# Build generic subgraph
# ---------------------------------------------------------------------------


generic_builder = StateGraph(GraphState)

generic_builder.add_node("generic_search", generic_search_node)
generic_builder.add_node("generic_respond", generic_respond_node)

generic_builder.add_edge(START, "generic_search")
generic_builder.add_edge("generic_search", "generic_respond")
generic_builder.add_edge("generic_respond", END)

generic_subgraph = generic_builder.compile()


# ---------------------------------------------------------------------------
# Build main graph
# ---------------------------------------------------------------------------


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
    route_agent,
    {
        "generic": "generic",
        "analytics": "dataset_fan",
    },
)

# Dataset node fans out into both analytics subgraphs
builder.add_edge("dataset_fan", "analytics")
builder.add_edge("dataset_fan", "plan_execute")

# Validation waits for both branches to complete
builder.add_edge("analytics", "validate")
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
            question="What is RAG?",
            csv_paths=[
                # "datasets/sales.csv",
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
