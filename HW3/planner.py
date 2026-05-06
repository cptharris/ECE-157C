"""
planner.py
"""

from schemas import *
import json


from call_llm import call_llm


PLANNER_SYSTEM_PROMPT = """TODO"""

PLANNER_USER_PROMPT = """TODO"""


def planner_node(state: State) -> State:
    raw = call_llm(
        PLANNER_SYSTEM_PROMPT,
        PLANNER_USER_PROMPT.format(
            dataset_description_shape=state["dataset_description"]["shape"],
            dataset_description_columns_types=state["dataset_description"]["columns"],
            user_question=state["question"],
        )
    )

    # TODO: extract plan from raw

    return state
