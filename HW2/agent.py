"""
agent.py
--------
Defines the LangGraph StateGraph and compiles it into a runnable graph.

State keys
----------
question         : str      - the natural language question
csv_path         : str      - path to the CSV file
csv_summary      : str      - compact dataset summary (shape, dtypes, stats)
generated_code   : str      - code produced by codegen_node
execution_result : any      - the `result` variable after exec()
execution_error  : str|None - traceback string if exec() raised
evaluation       : str      - "PASS" or "FAIL"
final_answer     : str      - human-readable answer
retry_count      : int      - how many times we have retried (max 1)
"""

from typing import Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from nodes import (
    summarize_node,
    codegen_node,
    execute_node,
    evaluate_node,
    respond_node,
)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    question: str
    csv_path: str
    csv_summary: str
    generated_code: str
    execution_result: Any
    execution_error: Optional[str]
    evaluation: str
    final_answer: str
    retry_count: int


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------


def route_after_evaluate(state: AgentState) -> str:
    """
    After evaluation:
    - PASS → respond
    - FAIL and no retry used yet → retry (loops back to codegen)
    - FAIL and already retried → respond anyway (best-effort)
    """
    if state.get("evaluation") == "PASS":
        return "respond"
    if state.get("retry_count", 0) < 1:
        return "retry"
    return "respond"


def increment_retry(state: AgentState) -> AgentState:
    """Tiny pass-through node that bumps retry_count before looping back."""
    return {"retry_count": state.get("retry_count", 0) + 1}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------


def build_graph():
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("summarize", summarize_node)  # new: runs once before codegen
    graph.add_node("codegen", codegen_node)
    graph.add_node("execute", execute_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("respond", respond_node)
    graph.add_node("retry", increment_retry)

    # Entry point
    graph.set_entry_point("summarize")

    # Linear edges
    graph.add_edge("summarize", "codegen")
    graph.add_edge("codegen", "execute")
    graph.add_edge("execute", "evaluate")

    # Conditional edge after evaluate
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "respond": "respond",
            "retry": "retry",
        },
    )

    # Retry skips summarize (summary is already cached in state) and goes
    # straight back to codegen so the LLM tries again with the same context.
    graph.add_edge("retry", "codegen")

    # Terminal edge
    graph.add_edge("respond", END)

    return graph.compile()


# Compiled graph — import this in test_agent.py
app = build_graph()
