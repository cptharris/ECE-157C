"""
web_search.py
"""

from langchain_core.runnables import RunnableConfig
from typing import Any

from schemas import GraphState, SearchResult, GenericResult, GenericResponse
from utilities import call_llm, call_ddg


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
    generic_result: GenericResult = state["generic_result"]

    prompt = f"""
Question:
{state["question"]}

Search results:
{generic_result["search_result"]["raw_text"]}
"""

    response: GenericResponse = call_llm(
        system_prompt="",
        user_prompt=prompt,
        who="generic_respond",
        response_model=GenericResponse,
    )

    updated: GenericResult = {
        **generic_result,
        "response": response.response,
    }

    return {
        "generic_result": updated,
    }
