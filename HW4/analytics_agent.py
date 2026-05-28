"""
analytics_agent.py
"""

from langchain_core.runnables import RunnableConfig
from typing import Any, Literal
import json
import copy

from schemas import (
    PLOTLY_NAMESPACE_KEY,
    MAX_ANALYTICS_STEPS,
    GraphState,
    Plot,
    AnalyticsStep,
    AnalyticsAction,
    AnalyticsResult,
    AnalyticsFinalAnswer,
)
from utilities import call_llm, execute


def analytics_init_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    import json

    dataset_schema = {}

    for csv_path, entry in state["dataset_thoughts"].schemas.items():
        dataset_schema[csv_path] = {"access": f"""dfs["{csv_path}"]""", **entry}

    state["namespace"] = {PLOTLY_NAMESPACE_KEY: {}}

    execute(
        f"""\
import pandas as pd
dfs = {{}}
for csv_path in {state["csv_paths"]}:
    dfs[csv_path] = pd.read_csv("datasets/" + csv_path)
    """,
        state["namespace"],
    )

    step = AnalyticsStep(
        step_index=0,
        reasoning="Reveal the shape and available columns of each dataset. 'dfs' is a dict of pandas DataFrames.",
        code="",
        execution=json.dumps(dataset_schema, indent=None),
        plots_captured=[],
    )

    return {
        "steps": [step],
        "namespace": state["namespace"],
        "current_step_index": 1,
        "is_complete": False,
    }


def analytics_step_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    prompt = f"\nQuestion:\n{state["question"]}"

    if "validation_feedback" in state:
        prompt += (
            "\n\nRetry feedback:\n" + state.get("plan_execute_result")["plan"].reasoning
        )

    prompt += "\n\nPrior steps:\n" + str(state["steps"]) + "\n"

    action: AnalyticsAction = call_llm(
        system_prompt="",
        user_prompt=prompt,
        who="analytics_step",
        response_model=AnalyticsAction,
    )

    execution_result = execute(
        action.code,
        state["namespace"],
    )

    harvested_plots = _harvest_plots(state["namespace"])

    step = AnalyticsStep(
        step_index=state["current_step_index"],
        reasoning=action.reasoning,
        code=action.code,
        execution=execution_result,
        plots_captured=[p["title"] for p in harvested_plots],
    )

    state["steps"].append(step)
    if "plots" not in state:
        state["plots"] = []
    state["plots"].append(harvested_plots)

    return {
        "steps": state["steps"],
        "plots": harvested_plots,
        "namespace": state["namespace"],
        "current_step_index": state["current_step_index"] + 1,
        "is_complete": action.is_final_step & (execution_result["error"] is None),
    }


def analytics_answer_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    prompt = f"""
Question:
{state["question"]}

Full step history:
{state["steps"]}
"""

    answer: AnalyticsFinalAnswer = call_llm(
        system_prompt="",
        user_prompt=prompt,
        who="analytics_answer",
        response_model=AnalyticsFinalAnswer,
    )

    result = AnalyticsResult(
        final_answer=answer.final_answer,
        overall_plan=answer.overall_plan,
        plots=state["plots"],
        steps=state["steps"],
    )

    return {
        "analytics_result": result,
    }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _harvest_plots(namespace: dict[str, Any]) -> list[Plot]:
    plots_dict = namespace.get(PLOTLY_NAMESPACE_KEY, {})

    harvested: list[Plot] = []

    for title, fig_json in plots_dict.items():
        harvested.append(
            Plot(
                title=title,
                figure_json=copy.deepcopy(json.loads(fig_json)),
            )
        )

    # Clear after harvest to avoid duplicate accumulation
    namespace[PLOTLY_NAMESPACE_KEY] = {}

    return harvested
