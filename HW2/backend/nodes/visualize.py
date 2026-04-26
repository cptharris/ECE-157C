from .call_llm import call_llm
from typing import Dict, Any
import json


SYSTEM_PROMPT_DECISION = """You are a data visualization decision engine.

Your job is to determine whether a visualization is useful.

Rules:
- Be conservative
- Only visualize if it improves understanding
- Map:
  comparisons → bar
  trends → line
  relationships → scatter
  distributions → histogram

Return ONLY JSON:
{
  "should_visualize": true/false,
  "chart_type": "...",
  "reason": "..."
}
"""


SYSTEM_PROMPT_VIZ_GEN = """You are a senior data visualization engineer.

Generate Plotly code.

Rules:
- result is preloaded
- use plotly.express as px
- assign figure to `fig`
- do not call fig.show()
- output ONLY Python code
"""


def visualization_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # ----------------------------
    # 1. Decision step
    # ----------------------------
    decision_prompt = f"""User question:
{state["input_question"]}

Result:
{state["execution"].get("result")}
"""

    decision_raw = call_llm(SYSTEM_PROMPT_DECISION, decision_prompt)
    viz_decision = json.loads(decision_raw)

    state["visualization"] = viz_decision

    # ----------------------------
    # 2. Conditional generation
    # ----------------------------
    if not viz_decision.get("should_visualize"):
        return state

    generation_prompt = f"""User question:
{state["input_question"]}

Chart type:
{viz_decision.get("chart_type")}

Result:
{state["execution"].get("result")}
"""

    plotly_code = call_llm(SYSTEM_PROMPT_VIZ_GEN, generation_prompt)

    state["visualization"]["plotly_code"] = plotly_code

    import plotly.express as px
    import plotly.graph_objects as go

    local_vars = {"result": state["execution"]["result"], "px": px, "go": go}

    try:
        exec(plotly_code, {}, local_vars)
        fig = local_vars.get("fig")

        if fig:
            state["visualization"]["figure_json"] = fig.to_json()
        else:
            state["visualization"]["figure_json"] = None
            state["visualization"]["error"] = "Visualization code did not create fig"

    except Exception as e:
        state["visualization"]["figure_json"] = None
        state["visualization"]["error"] = str(e)

    return state
