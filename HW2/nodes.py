"""
nodes.py
--------
Each function here is a LangGraph node. A node receives the full shared
state dict and returns a (partial) dict with the keys it wants to update.
LangGraph merges the returned dict back into state automatically.
"""

import os
import re
import traceback
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-5-mini"  # gpt-4o?


def call_llm(messages: list) -> str:
    return (
        client.chat.completions.create(model=MODEL, messages=messages)
        .choices[0]
        .message.content.strip()
    )


# ---------------------------------------------------------------------------
# Summarize node  (runs once before codegen)
# ---------------------------------------------------------------------------


def _build_csv_summary(csv_path: str) -> str:
    """
    Read the CSV and return a compact summary string:
      - shape
      - column names + dtypes
      - first 3 rows
      - numeric describe() (count/mean/std/min/max only)
      - null counts for columns that have any nulls
    """
    df = pd.read_csv(csv_path)
    rows, cols = df.shape

    lines = [
        f"Shape: {rows} rows x {cols} columns",
        "",
        "Columns and dtypes:",
    ]
    for col, dtype in df.dtypes.items():
        lines.append(f"  {col!r}: {dtype}")

    lines += ["", "First 3 rows:", df.head(3).to_string(index=False)]

    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        stats = numeric_df.describe().loc[["count", "mean", "std", "min", "max"]]
        lines += ["", "Numeric statistics:", stats.to_string()]

    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        lines += ["", "Columns with null values:", null_counts.to_string()]

    return "\n".join(lines)


# Cache so we don't re-read the CSV on retries within the same process.
_summary_cache: dict[str, str] = {}


def summarize_node(state: dict) -> dict:
    """Load the CSV and build a summary stored in state['csv_summary']."""
    csv_path = state["csv_path"]
    if csv_path not in _summary_cache:
        _summary_cache[csv_path] = _build_csv_summary(csv_path)
    return {"csv_summary": _summary_cache[csv_path]}


# ---------------------------------------------------------------------------
# Codegen node
# ---------------------------------------------------------------------------

CODEGEN_SYSTEM = """You are an expert Python / pandas data analyst.
Your job is to write a self-contained Python script that answers a question
about a CSV file.

Rules (follow exactly):
1. Use pandas to read the CSV from the exact path provided.
2. Store the final answer in a variable called `result`.
   `result` must be a pandas DataFrame, a pandas Series, a scalar, or a
   plain Python list/dict — something that can be printed or described.
3. Do NOT use print(), display(), or any I/O other than reading the CSV.
4. Do NOT include any import statements other than `import pandas as pd`
   and standard library modules.
5. Return ONLY raw Python code — no markdown fences, no explanation.
6. Use the dataset summary below to write correct column names and handle
   known null values or dtype quirks.
"""

CODEGEN_USER = """CSV path: {csv_path}

Dataset summary:
{csv_summary}

Question: {question}

Write the Python code now."""


def codegen_node(state: dict) -> dict:
    """Call the LLM to generate pandas code that answers the question."""
    prompt = CODEGEN_USER.format(
        csv_path=state["csv_path"],
        csv_summary=state.get("csv_summary", "(no summary available)"),
        question=state["question"],
    )
    code = call_llm(
        messages=[
            {"role": "system", "content": CODEGEN_SYSTEM},
            {"role": "user", "content": prompt},
        ]
    )
    # Strip accidental markdown fences if the model disobeys
    code = re.sub(r"^```(?:python)?\n?", "", code)
    code = re.sub(r"\n?```$", "", code)
    return {"generated_code": code}


# ---------------------------------------------------------------------------
# Execute node
# ---------------------------------------------------------------------------


def execute_node(state: dict) -> dict:
    """Run the generated code with exec() and capture `result`."""
    env: dict = {}
    try:
        exec(state["generated_code"], env)  # noqa: S102
        execution_result = env.get("result", None)
        execution_error = None
    except Exception:
        execution_result = None
        execution_error = traceback.format_exc()
    return {
        "execution_result": execution_result,
        "execution_error": execution_error,
    }


# ---------------------------------------------------------------------------
# Evaluate node
# ---------------------------------------------------------------------------

EVALUATE_SYSTEM = """You are a strict code-output evaluator.
Given a question and the execution result of some Python code, decide whether
the result correctly and meaningfully answers the question.

Reply with exactly one word: PASS or FAIL.

Return FAIL if any of the following are true:
- execution_result is None or empty
- the result is clearly unrelated to the question
- there was a runtime error
- the result is a trivially empty DataFrame or Series
"""

EVALUATE_USER = """Question: {question}

Execution result:
{execution_result}

Execution error (if any):
{execution_error}

Verdict (PASS or FAIL):"""


def evaluate_node(state: dict) -> dict:
    """Ask the LLM to judge whether the execution result answers the question."""
    result_str = str(state.get("execution_result", ""))[:3000]
    error_str = str(state.get("execution_error", "None"))

    verdict = call_llm(
        messages=[
            {"role": "system", "content": EVALUATE_SYSTEM},
            {
                "role": "user",
                "content": EVALUATE_USER.format(
                    question=state["question"],
                    execution_result=result_str,
                    execution_error=error_str,
                ),
            },
        ]
    ).upper()
    verdict = "PASS" if "PASS" in verdict else "FAIL"
    return {"evaluation": verdict}


# ---------------------------------------------------------------------------
# Respond node
# ---------------------------------------------------------------------------

RESPOND_SYSTEM = """You are a clear, concise data analyst.
Given a question, the raw execution result from a pandas analysis, and a
summary of the dataset, write a final human-readable answer.

Rules:
- Answer only from the execution result — do not invent numbers.
- Be specific: include key values, percentages, or rankings.
- Keep the answer under 150 words.
- Do not mention Python, pandas, or code.
"""

RESPOND_USER = """Dataset summary:
{csv_summary}

Question: {question}

Execution result:
{execution_result}

Final answer:"""


def respond_node(state: dict) -> dict:
    """Generate a human-readable final answer from the execution result."""
    result_str = str(state.get("execution_result", "No result available."))[:3000]

    final_answer = call_llm(
        messages=[
            {"role": "system", "content": RESPOND_SYSTEM},
            {
                "role": "user",
                "content": RESPOND_USER.format(
                    csv_summary=state.get("csv_summary", "(no summary available)"),
                    question=state["question"],
                    execution_result=result_str,
                ),
            },
        ]
    )
    return {"final_answer": final_answer}
