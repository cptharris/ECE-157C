from typing import Any, Dict
from nodes.memory import build_recent_context, get_previous_artifact


def history_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state["recent_context"] = build_recent_context(state["metadata"]["session_id"])

    previous = get_previous_artifact(state["metadata"]["session_id"])
    state["previous"] = {
        "data": previous.get("execution", {}).get("data") if previous else {},
        "data_desc": (
            previous.get("execution", {}).get("data_desc", "") if previous else ""
        ),
    }

    state["dataset_desc"] = previous.get("dataset_desc", None) if previous else None

    return state
