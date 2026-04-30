from langgraph.graph import StateGraph, END
from typing import Dict, Any

from nodes.history import history_node
from nodes.describe_dataset import describe_dataset_node
from nodes.planner import planner_node
from nodes.codegen import codegen_node
from nodes.execute import execute_node
from nodes.respond import respond_node
from nodes.visualize import visualize_codegen_node, visualize_execute_node


def build_graph():
    graph = StateGraph(Dict[str, Any])

    graph.add_node("history", traced(history_node))
    graph.add_node("describe_dataset", traced(describe_dataset_node))
    graph.add_node("planner", traced(planner_node))
    graph.add_node("codegen", traced(codegen_node))
    graph.add_node("execute", traced(execute_node))
    graph.add_node("respond", traced(respond_node))
    graph.add_node("visualize_codegen", traced(visualize_codegen_node))
    graph.add_node("visualize_execute", traced(visualize_execute_node))

    graph.set_entry_point("history")

    graph.add_edge("history", "describe_dataset")
    graph.add_edge("describe_dataset", "planner")
    graph.add_edge("planner", "codegen")
    graph.add_edge("codegen", "execute")

    graph.add_conditional_edges("execute", traced(route_post_execute))
    graph.add_edge("visualize_codegen", "visualize_execute")
    graph.add_conditional_edges("visualize_execute", traced(route_visualize_execute))

    graph.add_edge("respond", END)
    graph.add_edge("visualize_execute", END)

    return graph.compile()


def traced(fn):
    def wrapper(state):
        print(f"\n{'='*50}\n{' '*5}[NODE] {fn.__name__}\n{'='*50}")
        return fn(state)
    return wrapper


def route_post_execute(state: dict[str, Any]):
    if state["execution"].get("error"):
        return END
    if state["plan"].get("do_vis"):
        return "visualize_codegen"
    if state["plan"].get("do_response"):
        return "respond"
    return END


def route_visualize_execute(state: dict[str, Any]):
    if state["plan"].get("do_response"):
        return "respond"
    return END


agent = build_graph()
