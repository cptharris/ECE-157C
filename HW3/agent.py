from langgraph.graph import StateGraph, END
from schemas import AgentState

from describe_dataset import describe_dataset_node
# from planner import planner_node
# from executor import execute_node
# from respond import respond_node
from schemas import format_node


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------


def route_after_execute(state: AgentState) -> str:
    """
    After execution:
      - If all steps succeeded → respond
      - If a step failed and retries remain → re-plan (planner will see the
        error-annotated trace and produce a corrected plan)
      - If a step failed and retries are exhausted → respond anyway so the
        responder can surface a graceful error message
    """
    if all(entry.error is None for entry in state["trace"]):
        return "respond"

    if state["retry_count"] < state["max_retries"]:
        return "planner"  # re-plan with error context in state

    return "respond"  # give up gracefully


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph():
    graph = StateGraph(state_schema=AgentState)

    graph.add_node("describe_dataset", describe_dataset_node)
    # graph.add_node("planner", planner_node)
    # graph.add_node("execute", execute_node)
    # graph.add_node("respond", respond_node)
    graph.add_node("format", format_node)

    graph.set_entry_point("describe_dataset")

    graph.add_edge("describe_dataset", "format")

    graph.set_finish_point("format")

    # graph.add_edge("describe_dataset", "planner")
    # graph.add_edge("planner", "execute")

    # graph.add_conditional_edges("execute", route_after_execute)

    # graph.add_edge("respond", "format")
    graph.add_edge("format", END)

    return graph.compile()


agent = build_graph()
