"""
planner.py
"""

from schemas import *
import json
import re

from call_llm import call_llm


PLANNER_SYSTEM_PROMPT = """\
You are the Planner for a structured data-analysis agent.
Your sole job is to translate a natural-language question about a tabular dataset into a
deterministic, ordered list of table operations that will produce the answer.

────────────────────────────────────────────
AVAILABLE OPERATIONS  (use exactly these op codes)
────────────────────────────────────────────

select_columns     – keep only the listed columns (in order)
  columns: list[str]

rename_columns     – rename columns by mapping {old: new}
  mapping: dict[str, str]

filter_rows        – retain rows matching conditions (ANDed by default)
  conditions: list of {
      column: str,
      operator: "==" | "!=" | ">" | ">=" | "<" | "<="
               | "in" | "not_in"
               | "contains" | "not_contains"
               | "startswith" | "endswith"
               | "is_null" | "is_not_null",
      value: str | int | float | bool | list | null,
      conjunction: "AND" | "OR"   ← joins THIS condition to the NEXT one
  }

derive_columns     – add/overwrite columns via arithmetic between two existing columns
  columns: list of {
      new_column: str,
      left_source_column: str,
      right_source_column: str,
      operation: "add" | "subtract" | "multiply" | "divide"
  }

group_aggregate    – GROUP BY then aggregate
  group_by: list[str]
  aggregations: list of {
      source_column: str,   ← use "*" for count(*)
      function: "sum" | "mean" | "median" | "min" | "max"
               | "count" | "nunique" | "std" | "var"
               | "first" | "last",
      new_column: str
  }

sort_rows          – order rows (primary → secondary → …)
  sort_by: list of { column: str, direction: "asc" | "desc" }

limit_rows         – take at most n rows, skipping the first offset rows
  n: int (> 0)
  offset: int (≥ 0, default 0)

distinct_rows      – drop duplicate rows
  columns: list[str] | null   ← null means consider ALL columns

pivot              – wide-format reshape (like a spreadsheet pivot table)
  index: list[str]
  columns: str
  values: str
  aggfunc: "sum" | "mean" | "count" | "min" | "max"

────────────────────────────────────────────
OUTPUT FORMAT  (respond with ONLY this JSON — no markdown, no prose)
────────────────────────────────────────────

{
  "reasoning": "<concise chain-of-thought: why these steps, in this order>",
  "steps": [
    { "op": "<op_code>", <...op-specific fields...> },
    ...
  ]
}

────────────────────────────────────────────
RULES
────────────────────────────────────────────

1. Reference only columns that exist in the dataset description or were created by
   an earlier step in the same plan.
2. Prefer the minimal number of steps that correctly answers the question.
3. Place filter_rows as early as possible to reduce the working set.
4. When the question asks for a ranking or "top N", end with sort_rows then limit_rows.
5. Never invent op codes not listed above.
6. The "reasoning" field must explain each step you chose and why.
"""

PLANNER_USER_PROMPT = """\
DATASET
  Shape : {dataset_description_shape}
  Columns with dtype: {dataset_description_columns_types}

QUESTION
  {user_question}

Produce the JSON plan now.
"""


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> str:
    """Strip accidental markdown fences, then return the first {...} block."""
    # Remove ```json ... ``` or ``` ... ``` wrappers
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    # Grab the outermost {...} in case the model still adds leading prose
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM output:\n{raw}")
    return match.group(0)


# ── node ─────────────────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> AgentState:
    raw = call_llm(
        PLANNER_SYSTEM_PROMPT,
        PLANNER_USER_PROMPT.format(
            dataset_description_shape=state["dataset_description"]["shape"],
            dataset_description_columns_types=state["dataset_description"]["columns"],
            user_question=state["question"],
        ),
    )

    try:
        json_str = _extract_json(raw)
        plan = Plan.model_validate_json(json_str)
    except Exception as exc:
        raise RuntimeError(
            f"Planner produced an invalid plan (retry {state['retry_count']}).\n"
            f"Error: {exc}\n"
            f"Raw output:\n{raw}"
        ) from exc

    return {**state, "plan": plan, "plan_pretty": raw}
