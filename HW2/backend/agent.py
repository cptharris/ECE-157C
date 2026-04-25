from langgraph.graph import StateGraph, END
from nodes.schemas import Artifact

from nodes.summarize import summarize_node
from nodes.codegen import codegen_node
from nodes.execute import execute_node
from nodes.respond import respond_node
from nodes.visualize import visualization_node


def build_graph():

    graph = StateGraph(Artifact)

    graph.add_node("summarize", summarize_node)
    graph.add_node("codegen", codegen_node)
    graph.add_node("execute", execute_node)
    graph.add_node("respond", respond_node)
    graph.add_node("visualize", visualization_node)

    graph.set_entry_point("summarize")

    graph.add_edge("summarize", "codegen")
    graph.add_edge("codegen", "execute")
    graph.add_edge("execute", "respond")

    graph.add_edge("respond", "visualize")
    graph.add_edge("visualize", END)

    return graph.compile()


agent = build_graph()
