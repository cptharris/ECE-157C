"""
validator_agent.py

Inputs: original question, analytics agent output (final answer + execution trace)
Process:

1) Prompt the LLM independently: "Given this question and these results, does the analysis correctly and completely answer the question? Are the numbers consistent? Is anything missing?"
2) Extend operator coverage for multi-table ops: add join, union, cross_year_compare to whatever operators HW3 defined.
3) Return structured JSON: {"verdict": "pass" | "retry" | "flag", "reason": "...", "missing": [...]}.

Keep it as a pure function — no side effects, no state. The orchestrator decides what to do with the verdict.
"""

from langchain_core.runnables import RunnableConfig
from typing import Any
import copy

from schemas import GraphState, ValidationDecision
from utilities import call_llm


def validation_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    analytics = copy.deepcopy(state["analytics_result"])
    plan_execute = copy.deepcopy(state["plan_execute_result"])

    analytics.pop("overall_plan")
    analytics.pop("plots")
    plan_execute.pop("trace")

    prompt = f"""
Question:
{state["question"]}

Analytics answer:
{analytics["final_answer"]}

Plan-execute answer:
{plan_execute["final_answer"]}
"""

    decision: ValidationDecision = call_llm(
        system_prompt="",
        user_prompt=prompt,
        who="validation",
        response_model=ValidationDecision,
    )

    return {
        "validation_result": decision,
    }


def retry_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    return {
        "retry_count": state["retry_count"] + 1,
        "validation_feedback": state["validation_result"].feedback,
        "analytics_result": None,
        # Preserve the deterministic validator baseline across retries.
        "plan_execute_result": state["plan_execute_result"],
        "current_step_index": 0,
        "is_complete": False,
    }
