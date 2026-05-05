from typing import Dict, Any
from langgraph.graph import StateGraph, END


from describe_dataset import describe_dataset_node
from planner import planner_node
from executor import execute_node
from respond import respond_node

from schemas import *


def build_graph():
    graph = StateGraph(State)

    graph.add_node("describe_dataset", traced(describe_dataset_node))
    graph.add_node("planner", traced(planner_node))
    graph.add_node("execute", traced(execute_node))
    graph.add_node("respond", traced(respond_node))

    graph.set_entry_point("describe_dataset")

    graph.add_edge("describe_dataset", "planner")
    graph.add_edge("planner", "execute")
    graph.add_edge("execute", "respond")

    graph.set_finish_point("respond")

    return graph.compile()


def traced(fn):
    def wrapper(state):
        print(f"\n{'='*50}\n{' '*5}[NODE] {fn.__name__}\n{'='*50}")
        return fn(state)

    return wrapper


agent = build_graph()
