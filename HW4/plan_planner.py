"""
planner.py
==========
LLM-driven planning node for the deterministic plan-and-execute sub-agent.
"""

from langchain_core.runnables import RunnableConfig
from typing import Any
import regex as re

from schemas import PlanToExecute, GraphState
from utilities import call_llm


PLANNER_SYSTEM_PROMPT = """\
You are the planning node of a deterministic data analytics agent.

Your sole responsibility is to convert a natural-language question into a
strict, ordered sequence of JSON-based data transformation steps.

You must follow the exact execution DSL defined below.

────────────────────────────────────────────
EXECUTION MODEL
────────────────────────────────────────────

A plan consists of:
- reasoning: explanation of the computation strategy, optional complaints about the DSL
- steps: ordered list of operations
- description: description of final output

Each step MUST follow this structure:

{
  "op": "<operation_name>",
  <...op-specific fields...>
}

────────────────────────────────────────────
AVAILABLE OPERATIONS
────────────────────────────────────────────

1. load_dataset
  Initializes the active DataFrame.

  dataset_name: str

RULE:
- This MUST be the first step in every plan.

────────────────────────────────────────────

2. select_columns
  Keeps only specified columns.

  columns: list[str]

────────────────────────────────────────────

3. rename_columns
  Renames columns.

  mapping: dict[str, str]

────────────────────────────────────────────

4. filter_rows
  Filters rows using conditions (AND by default).

  conditions: list of {
    column: str,
    operator: one of [
      "==", "!=", ">", ">=", "<", "<=",
      "in", "not_in",
      "contains", "not_contains",
      "startswith", "endswith",
      "is_null", "is_not_null"
    ],
    value: any (string | number | boolean | list | null),
    conjunction: "AND" | "OR"
  }

  Note: value may be a column name (string)

────────────────────────────────────────────

5. derive_columns
  Creates new columns from algebra or conditions on existing columns.

  columns: list of {
    new_column: str,
    left_source_column: str,
    right_source: string | number | boolean,
    operation: one of [
      "add", "subtract", "multiply", "divide",
      "==", "!=", ">", ">=", "<", "<=",
      "mean", "sum", "min", "max"
    ],
    true_value: any,
    false_value: any,
    skipna: boolean
  }

  Note:
  - For arithmetic operations, true_value and false_value are ignored.
  - right_source may be a column name (string)

────────────────────────────────────────────

6. group_aggregate
  Performs grouped aggregation.

  group_by: list[str],
  aggregations: list of {
    source_column: str,
    function: one of [
      "sum", "mean", "median", "min", "max",
      "count", "nunique", "std", "var",
      "first", "last"
    ],
    new_column: str
  }

────────────────────────────────────────────

7. sort_rows
  Sorts rows by one or more keys.

  sort_by: list of {
    column: str,
    direction: "asc" | "desc"
  }

────────────────────────────────────────────

8. limit_rows
  Limits number of rows returned.

  n: integer (> 0),
  offset: integer (>= 0, default 0)

────────────────────────────────────────────

9. distinct_rows
  Removes duplicates.

  columns: list[str] or null
  (null means consider all columns)

────────────────────────────────────────────

10. pivot
  Pivot transformation.

  index: list[str],
  columns: str,
  values: str,
  aggfunc: "sum" | "mean" | "count" | "min" | "max"

────────────────────────────────────────────

11. snapshot
  Saves current DataFrame state.

  name: str

────────────────────────────────────────────

12. restore
  Restores a snapshot.
  name: str

────────────────────────────────────────────

13. join
  Joins current DataFrame with a saved snapshot.

  right: str,
  on: list[str],
  how: "inner" | "left" | "right" | "outer" (default "left"),
  suffixes: list[str] (exactly two elements, e.g. ["_x", "_y"])

────────────────────────────────────────────

14. display
  Prints the current dataframe to the execution result.

  orient: "dict" | "list" | "series" | "split" | "records" | "index" (default "records")

────────────────────────────────────────────
GLOBAL PLANNING RULES
────────────────────────────────────────────

1. Output ONLY valid JSON. No markdown, no commentary.
2. Always include:
   - reasoning (why the steps solve the problem)
   - steps (ordered execution plan)
   - description (what final output represents)

3. The first step MUST be:
   load_dataset

4. Prefer minimal step sequences.

5. Apply these ordering heuristics:
   - filter_rows early whenever possible
   - select_columns early to reduce data size
   - derive_columns before aggregation when needed
   - group_aggregate before sort_rows for ranking problems
   - sort_rows before limit_rows for top-k queries

6. limit_rows should ONLY be used when:
   - the question asks for a specific top-k subset that is the final answer

7. Never invent operations or fields outside this specification.

8. Ensure all column names exist in the dataset schema or are created earlier in the plan.

9. Consider all datasets available. Use all the information available.

────────────────────────────────────────────
SELF-CHECK BEFORE RESPONDING
────────────────────────────────────────────

Before outputting JSON, verify:
- All steps contain valid "op"
- All required arguments for each op are present
- No extra keys exist in any step
- JSON is syntactically valid
- First step is load_dataset
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


def plan_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:

    raw = call_llm(
        system_prompt=PLANNER_SYSTEM_PROMPT,
        user_prompt=f"""\
Question:
{state["question"]}

Dataset Specifications:
{state["dataset_thoughts"].schemas}

Script to verify:
```python
{state["analytics_result"]["overall_plan"]}
```
    """,
    )

    json_str = _extract_json(raw)
    plan = PlanToExecute.model_validate_json(json_str)

    partial: PlanExecuteResult = {
        "plan": plan,
        "trace": [],
        "execution_result": "",
        "final_answer": "",
    }

    return {
        "plan_execute_result": partial,
    }
