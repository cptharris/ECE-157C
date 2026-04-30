from typing import Dict, Any
import json
import plotly
import traceback
from .call_llm import call_llm
from utils import sanitize_plotly


VIS_SYSTEM_PROMPT = """Generate Plotly code. result is preloaded. Use plotly.express as px, assign figure to fig, do not call fig.show(), output ONLY Python code."""

VIS_USER_PROMPT = """
Visualization goal:
{viz_spec}

Captured data:
{data}
"""


def visualize_codegen_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state["vis"]["vis_code"] = call_llm(
        VIS_SYSTEM_PROMPT,
        VIS_USER_PROMPT.format(
            viz_spec=state["plan"]["viz_spec"],
            data=state["execution"]["data"],
        ),
    )
    return state


def visualize_execute_node(state: Dict[str, Any]) -> Dict[str, Any]:
    import plotly.express as px

    local_vars = {"result": state["execution"]["data"], "px": px}

    try:
        exec(state["vis"]["vis_code"], local_vars)

        fig = local_vars.get("fig")

        state["vis"]["fig"] = (
            json.loads(
                json.dumps(
                    sanitize_plotly(fig.to_plotly_json()),
                    cls=plotly.utils.PlotlyJSONEncoder,
                )
            )
            if fig
            else None
        )

        state["vis"]["error"] = None

    except Exception as e:
        state["vis"]["fig"] = None
        state["vis"]["error"] = str(e)
        traceback.print_exception(e)

    return state
