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
    AnalyticsReasoning,
    AnalyticsAction,
    AnalyticsResult,
    AnalyticsFinalAnswer,
    ExecutionResult,
)
from utilities import call_llm, execute


def analytics_init_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    code = """\
import pandas as pd
# dataFrames is pre-loaded
print(dataFrames.keys())"""

    execution_result: ExecutionResult = execute(
        code,
        state["namespace"],
    )

    execution_result["stdout"] += "\n" + json.dumps(
        state["dataset_thoughts"].schemas, indent=None
    )

    step = AnalyticsStep(
        step_index=0,
        reasoning="Reveal the shape and available columns of each dataset.",
        code=code,
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
    prompt = f"\nQuestion:\n{state['question']}"

    if "validation_feedback" in state:
        prompt += "\n\nRetry feedback:\n" + str(state.get("validation_feedback"))

    prompt += "\n\nPrior steps:\n" + str(state["steps"]) + "\n"

    reasoning_result: AnalyticsReasoning = call_llm(
        system_prompt="",
        user_prompt=prompt,
        who="analytics_reasoning",
        response_model=AnalyticsReasoning,
    )

    if not reasoning_result.requires_python_execution:
        step = AnalyticsStep(
            step_index=state["current_step_index"],
            reasoning=reasoning_result.reasoning,
            code="",
            execution={
                "stdout": "",
                "stderr": "",
                "error": None,
            },
            plots_captured=[],
        )

        state["steps"].append(step)

        return {
            "steps": state["steps"],
            "namespace": state["namespace"],
            "current_step_index": state["current_step_index"] + 1,
            "is_complete": True,
        }

    code_prompt = f"""\
Question:
{state['question']}

Prior steps:
{state['steps']}

Reasoning about the next analysis step:
{reasoning_result.reasoning}
"""

    action: AnalyticsAction = call_llm(
        system_prompt="",
        user_prompt=code_prompt,
        who="analytics_code_generation",
        response_model=AnalyticsAction,
    )

    execution_result = execute(
        action.code,
        state["namespace"],
    )

    harvested_plots = _harvest_plots(state["namespace"])

    step = AnalyticsStep(
        step_index=state["current_step_index"],
        reasoning=reasoning_result.reasoning,
        code=action.code,
        execution=execution_result,
        plots_captured=[p["title"] for p in harvested_plots],
    )

    state["steps"].append(step)

    return {
        "steps": state["steps"],
        "plots": harvested_plots,
        "namespace": state["namespace"],
        "current_step_index": state["current_step_index"] + 1,
        "is_complete": False,
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
