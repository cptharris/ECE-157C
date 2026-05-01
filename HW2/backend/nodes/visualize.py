from typing import Dict, Any
import pandas as pd
import json
import plotly
import traceback
from .call_llm import call_llm
from utils import sanitize_plotly


VIS_SYSTEM_PROMPT = """You are a data visualization expert.
Rules:
- Generate Plotly code.
- you are given a dataframe called df
- you must return the figure as fig
- do not call fig.show()
- output ONLY Python code."""

VIS_USER_PROMPT = """
Visualization goal:
{viz_spec}
Dataset description:
{dataset_desc}
"""


def visualize_codegen_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state["vis"]["vis_code"] = call_llm(
        VIS_SYSTEM_PROMPT,
        VIS_USER_PROMPT.format(
            viz_spec=state["plan"]["viz_spec"],
            dataset_desc=state["dataset_desc"],
        ),
    )
    return state


def visualize_execute_node(state: Dict[str, Any]) -> Dict[str, Any]:
    import plotly.express as px

    df = pd.read_csv(f"datasets/{state['metadata']['dataset_name']}")

    local_vars = {"df": df, "px": px}

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
