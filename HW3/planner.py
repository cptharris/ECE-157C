from typing import Dict, Any
import json


from call_llm import call_llm


def planner_node(state: State) -> State:
    raw = call_llm(
        PLANNER_SYSTEM_PROMPT,
        PLANNER_USER_PROMPT.format(
            recent_context=state.get("recent_context", "None"),
            previous_data_desc=state["previous"].get("data_desc", "None"),
            dataset_description=state.get("dataset_description", ""),
            prompt=state["metadata"]["prompt"],
        ),
    )
    plan = json.loads(raw)
    # Normalize to the new state shape: state["steps"] as a list
    if isinstance(plan, dict) and "steps" in plan:
        steps = plan["steps"]
    else:
        steps = plan
    state["steps"] = steps
    # Keep backward compatibility for any legacy consumers
    # but avoid overwriting existing state unless we have steps
    return state
