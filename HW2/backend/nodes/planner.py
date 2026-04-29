from typing import Dict, Any
import json
from .call_llm import call_llm

PLANNER_SYSTEM_PROMPT = """You are a dataset agent planner. Return ONLY valid JSON with keys:
is_follow_up
- given the recent context and previous data description, is the prompt a follow-up?
do_response
- should we generate a text response to the prompt? in most cases, this is true
- unless the prompt specifically asks for a visualization
do_vis
- should we create a visualization? be conservative
- only visualize if it improves understanding
question
- given the recent context (if follow-up) and the prompt, clearly and concisely state the question
viz_spec
- specify what to visualize
data_spec
- specify what data to extract to answer the question, generate the visualization (if applicable), and answer potential follow-up questions"""

PLANNER_USER_PROMPT = """
Recent context:
{recent_context}

Previous data description:
{previous_data_desc}

Dataset description:
{dataset_desc}

Prompt:
{prompt}
"""


def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    raw = call_llm(
        PLANNER_SYSTEM_PROMPT,
        PLANNER_USER_PROMPT.format(
            recent_context=state.get("recent_context", "None"),
            previous_data_desc=state["previous"].get("data_desc", "None"),
            dataset_desc=state.get("dataset_desc", ""),
            prompt=state["metadata"]["prompt"],
        ),
    )

    state["plan"] = json.loads(raw)
    return state
