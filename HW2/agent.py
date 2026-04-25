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

HW2: Stateful, interactive LangGraph agent with memory and visualization.

New state keys vs. HW1:
  memory      : ConversationMemory  - stores turns + prior result
  is_followup : bool                - True when the question operates on a prior result
  viz_json    : str | None          - Plotly figure JSON for the frontend
  viz_decision: dict | None         - LLM's visualization decision metadata
"""

from typing import Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from nodes import (
    ConversationMemory,
    summarize_node,
    codegen_node,
    execute_node,
    evaluate_node,
    respond_node,
    viz_node,
)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    # Core (HW1)
    question: str
    csv_path: str
    csv_summary: str
    generated_code: str
    execution_result: Any
    execution_error: Optional[str]
    evaluation: str
    final_answer: str
    retry_count: int

    # New (HW2)
    memory: ConversationMemory
    is_followup: bool
    viz_json: Optional[str]
    viz_decision: Optional[dict]


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


def route_after_summarize(state: AgentState) -> str:
    """
    If this is a follow-up and we already have a prior result, skip codegen
    from CSV and go straight to memory-aware codegen. (Both paths hit the
    same codegen_node; the node itself checks is_followup internally.)
    """
    return "codegen"


# ---------------------------------------------------------------------------
# Memory commit node  (runs after viz, before END)
# ---------------------------------------------------------------------------


def commit_to_memory(state: AgentState) -> dict:
    """Save the completed turn into ConversationMemory."""
    memory: ConversationMemory = state["memory"]
    memory.add_turn(
        question=state["question"],
        final_answer=state.get("final_answer", ""),
        execution_result=state.get("execution_result"),
    )
    return {}  # memory is mutated in-place; no state key to update

# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------


def build_graph():
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("summarize", summarize_node)
    graph.add_node("codegen", codegen_node)
    graph.add_node("execute", execute_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("respond", respond_node)
    graph.add_node("retry", increment_retry)
    graph.add_node("visualize", viz_node)
    graph.add_node("commit", commit_to_memory)

    graph.set_entry_point("summarize")

    graph.add_edge("summarize", "codegen")
    graph.add_edge("codegen", "execute")
    graph.add_edge("execute", "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"respond": "respond", "retry": "retry"},
    )

    # Retry skips summarize (summary cached in memory) → back to codegen
    graph.add_edge("retry", "codegen")

    # After respond → visualize → commit to memory → done
    graph.add_edge("respond", "visualize")
    graph.add_edge("visualize", "commit")
    graph.add_edge("commit", END)

    return graph.compile()


app = build_graph()


# ---------------------------------------------------------------------------
# Convenience runner used by the backend
# ---------------------------------------------------------------------------


def run_turn(
    question: str,
    csv_path: str,
    memory: ConversationMemory,
    is_followup: bool = False,
) -> dict:
    """
    Run one conversational turn and return a result dict with:
      final_answer, viz_json, generated_code, evaluation
    """
    initial_state: AgentState = {
        "question": question,
        "csv_path": csv_path,
        "memory": memory,
        "is_followup": is_followup,
        "retry_count": 0,
        # carry forward cached summary if available
        "csv_summary": memory.csv_summary or "",
    }

    output = app.invoke(initial_state)

    return {
        "final_answer": output.get("final_answer", ""),
        "viz_json": output.get("viz_json"),
        "generated_code": output.get("generated_code", ""),
        "evaluation": output.get("evaluation", ""),
        "viz_decision": output.get("viz_decision"),
    }
