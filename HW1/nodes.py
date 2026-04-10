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
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-5-mini"  # gpt-4o?


def call_llm(messages):
    return (
        client.chat.completions.create(model=MODEL, messages=messages)
        .choices[0]
        .message.content.strip()
    )


# ---------------------------------------------------------------------------
# Codegen node
# ---------------------------------------------------------------------------

CODEGEN_SYSTEM = """You are an expert Python / pandas data analyst.
Your job is to write a self-contained Python script that answers a question
about a CSV file.

Rules (follow exactly):
1. Use pandas to read the CSV from the path provided.
2. Store the final answer in a variable called `result`.
   `result` must be a pandas DataFrame, a pandas Series, a scalar, or a
   plain Python list/dict — something that can be printed or described.
3. Do NOT use print(), display(), or any I/O other than reading the CSV.
4. Do NOT include any import statements other than `import pandas as pd`
   and standard library modules.
5. Return ONLY raw Python code — no markdown fences, no explanation.
"""

CODEGEN_USER = """CSV path: {csv_path}

Question: {question}

Write the Python code now."""


def codegen_node(state: dict) -> dict:
    """Call the LLM to generate pandas code that answers the question."""
    prompt = CODEGEN_USER.format(
        csv_path=state["csv_path"],
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
    result_str = str(state.get("execution_result", ""))[:3000]  # trim huge outputs
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
Given a question and the raw execution result from a pandas analysis,
write a final human-readable answer.

Rules:
- Answer only from the execution result — do not invent numbers.
- Be specific: include key values, percentages, or rankings.
- Keep the answer under 150 words.
- Do not mention Python, pandas, or code.
"""

RESPOND_USER = """Question: {question}

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
                    question=state["question"],
                    execution_result=result_str,
                ),
            },
        ]
    )
    return {"final_answer": final_answer}
