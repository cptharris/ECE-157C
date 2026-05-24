"""
analytics_agent.py
"""

from langchain_core.runnables import RunnableConfig
from typing import Any, Literal

from schemas import (
    GraphState,
    OrchestrationDecision,
    PLOTLY_NAMESPACE_KEY,
    Plot,
    AnalyticsStep,
    AnalyticsAction,
    AnalyticsFinalAnswer,
    AnalyticsResult,
    MAX_ANALYTICS_STEPS,
)
from utilities import call_llm, execute


def analytics_init_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:

    code = """\
import pandas as pd
import json

dataset_schemas = {{}}
dfs = {{}}
for csv_path in {csv_paths}:
    dfs[csv_path] = pd.read_csv("datasets/" + csv_path)

    dataset_schemas[csv_path] = {{
        "access": f"dfs['{{csv_path}}']",
        "shape": dfs[csv_path].shape,
        "columns": dfs[csv_path].columns.tolist(),
    }}
print(json.dumps(dataset_schemas, indent=0))
    """.format(
        csv_paths=state["csv_paths"]
    )

    state["namespace"] = {PLOTLY_NAMESPACE_KEY: {}}

    execution_result = execute(code, state["namespace"])

    step = AnalyticsStep(
        step_index=0,
        reasoning="Reveal the shape and schema of each dataset.",
        code="",
        execution=execution_result,
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
    prompt = f"""\
Question:
{state["question"]}

Retry feedback:
{state.get("validation_feedback")}

Prior steps:
{state["steps"]}
"""

    action: AnalyticsAction = call_llm("", prompt, AnalyticsAction)

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

    return {
        "steps": [step],
        "plots": harvested_plots,
        "namespace": state["namespace"],
        "current_step_index": state["current_step_index"] + 1,
        "is_complete": action.is_final_step,
    }


def analytics_continue_router(
    state: GraphState,
) -> Literal["continue", "answer"]:
    if state["is_complete"]:
        return "answer"

    if state["current_step_index"] >= MAX_ANALYTICS_STEPS:
        return "answer"

    return "continue"


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

    answer: AnalyticsFinalAnswer = call_llm("", prompt, AnalyticsFinalAnswer)

    result = AnalyticsResult(
        final_answer=answer.final_answer,
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
                figure_json=copy.deepcopy(fig_json),
            )
        )

    # Clear after harvest to avoid duplicate accumulation
    namespace[PLOTLY_NAMESPACE_KEY] = {}

    return harvested
