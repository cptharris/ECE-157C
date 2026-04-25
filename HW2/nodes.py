"""
nodes.py
--------
Each function here is a LangGraph node. A node receives the full shared
state dict and returns a (partial) dict with the keys it wants to update.
LangGraph merges the returned dict back into state automatically.

LangGraph nodes for the HW2 stateful agent.

New vs. HW1:
  - ConversationMemory  : stores turns, prior execution result, csv summary
  - codegen_node        : memory-aware; operates on prior result for follow-ups
  - viz_node            : LLM decides chart type, generates + executes Plotly code
"""

import os
import re
import json
import traceback
import pandas as pd
from openai import OpenAI

# ---------------------------------------------------------------------------
# LLM cache (toggle via env: LLM_CACHE=1)
# ---------------------------------------------------------------------------
from llm_cache import get_cached, set_cached


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-5-mini"


def call_llm(messages: list) -> str:
    print("===== MESSAGE =====")
    print(messages[1]["content"])

    cached = get_cached(messages)
    if cached is not None:
        print("===== RESPONSE (cached) =====")
        print(cached)
        return cached

    response = (
        client.chat.completions.create(model=MODEL, messages=messages)
        .choices[0]
        .message.content.strip()
    )

    set_cached(messages, response)

    print("===== RESPONSE (rqsted) =====")
    print(response)

    return response


# ===========================================================================
# ConversationMemory
# ===========================================================================


class ConversationMemory:
    """
    Stores the conversation history and the most recent execution result
    so follow-up queries can operate on prior results without re-reading
    the original CSV.

    Turn schema:
        {
            "question":         str,
            "final_answer":     str,
            "execution_result": <serialized result dict or None>,
        }

    Execution result serialization format:
        {
            "type":    "dataframe" | "series" | "scalar" | "list" | "other",
            "columns": [...] | None,
            "data":    <records list> | <value>,
        }
    """

    def __init__(self):
        self.turns: list[dict] = []
        self.csv_summary: str | None = None  # cached across turns

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def serialize_result(result) -> dict:
        """Convert an execution result into a JSON-serializable dict."""
        if result is None:
            return {"type": "other", "columns": None, "data": None}
        if isinstance(result, pd.DataFrame):
            return {
                "type": "dataframe",
                "columns": list(result.columns),
                "dtypes": {c: str(result[c].dtype) for c in result.columns},
                "data": result.to_dict(orient="records"),
            }
        if isinstance(result, pd.Series):
            return {
                "type": "series",
                "columns": [result.name or "value"],
                "data": result.to_dict(),
            }
        if isinstance(result, list):
            return {"type": "list", "columns": None, "data": result}
        if isinstance(result, dict):
            # Normalize dicts into a list-of-records structure for consistency
            try:
                # Case 1: dict of scalars → convert to key/value rows
                if all(not isinstance(v, (list, dict)) for v in result.values()):
                    data = [{"key": k, "value": v} for k, v in result.items()]
                    return {
                        "type": "dataframe",
                        "columns": ["key", "value"],
                        "data": data,
                    }
                # Case 2: dict of lists → convert to DataFrame directly
                df = pd.DataFrame(result)
                return {
                    "type": "dataframe",
                    "columns": list(df.columns),
                    "data": df.to_dict(orient="records"),
                }
            except Exception:
                # Fallback: wrap as single-row DataFrame
                return {
                    "type": "dataframe",
                    "columns": list(result.keys()),
                    "data": [result],
                }
        # scalar
        return {"type": "scalar", "columns": None, "data": result}

    @staticmethod
    def result_to_code_snippet(serialized: dict) -> str:
        """
        Return Python code (as a string) that reconstructs the prior result
        into a variable called `prior_result`. Used by memory-aware codegen
        and viz_node so the LLM-generated code has a live object to work with.
        """
        t = serialized.get("type")
        data_json = json.dumps(serialized.get("data"), default=str)
        if t == "dataframe":
            return (
                "import json, pandas as pd\n"
                f"_data = {data_json}\n"
                "prior_result = pd.DataFrame(_data)\n"
                "for col in prior_result.columns:\n"
                "\tprior_result[col] = pd.to_numeric(prior_result[col], errors='coerce')\n"
            )
        if t == "series":
            return (
                "import json, pandas as pd\n" f"prior_result = pd.Series({data_json})"
            )
        return f"import pandas as pd\nprior_result = {data_json}"

    @staticmethod
    def get_schema(serialized: dict) -> dict:
        """Return a strict schema object for downstream LLM use."""
        if serialized.get("type") != "dataframe":
            return {
                "type": serialized.get("type"),
                "columns": serialized.get("columns"),
            }

        return {
            "type": "dataframe",
            "columns": serialized.get("columns"),
            "dtypes": serialized.get("dtypes", {}),
        }

    @staticmethod
    def result_preview(serialized: dict, max_rows: int = 5) -> str:
        """
        Return a compact human-readable description of the serialized result
        for injection into LLM prompts. Includes type, exact column names,
        and up to `max_rows` sample rows so the LLM uses correct column names
        instead of guessing — the key fix for viz_node and follow-up codegen.
        """
        t = serialized.get("type")
        data = serialized.get("data")
        cols = serialized.get("columns")

        if t == "scalar":
            return f"Type: scalar\nValue: {data}"

        if t == "series":
            items = (
                list(data.items()) if isinstance(data, dict) else list(enumerate(data))
            )
            lines = [
                "Type: Series",
                f"Name/index column: {cols[0] if cols else 'unknown'}",
                f"Length: {len(items)}",
                "Sample entries (index → value):",
            ]
            for k, v in items[:max_rows]:
                lines.append(f"  {k!r}: {v!r}")
            if len(items) > max_rows:
                lines.append(f"  … ({len(items) - max_rows} more rows)")
            return "\n".join(lines)

        if t in ("dataframe", "list", "dict"):
            records = (
                data
                if isinstance(data, list)
                else ([data] if isinstance(data, dict) else [])
            )
            lines = [
                "Type: DataFrame",
                f"Shape: {len(records)} rows × {len(cols or [])} columns",
                f"Exact column names: {cols!r}",
                f"Column dtypes: {serialized.get('dtypes', {})}",
            ]
            if records:
                sample = records[:max_rows]
                try:
                    lines.append("Sample rows:")
                    lines.append(
                        pd.DataFrame(sample, columns=cols).to_string(index=False)
                    )
                except Exception:
                    for r in sample:
                        lines.append(f"  {r}")
            if len(records) > max_rows:
                lines.append(f"  … ({len(records) - max_rows} more rows)")
            return "\n".join(lines)

        return f"Type: {t}\nData: {str(data)[:300]}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_turn(self, question: str, final_answer: str, execution_result) -> None:
        self.turns.append(
            {
                "question": question,
                "final_answer": final_answer,
                "execution_result": self.serialize_result(execution_result),
            }
        )

    def get_last_result_serialized(self) -> dict | None:
        if not self.turns:
            return None
        return self.turns[-1]["execution_result"]

    def get_history_text(self, max_turns: int = 5) -> str:
        """Return a compact text summary of the last N turns for prompt injection."""
        recent = self.turns[-max_turns:]
        lines = []
        for i, t in enumerate(recent, 1):
            lines.append(f"Turn {i}:")
            lines.append(f"  Q: {t['question']}")
            lines.append(f"  A: {t['final_answer']}")
        return "\n".join(lines) if lines else "(no prior turns)"

    def has_prior_result(self) -> bool:
        return bool(self.turns)


# ===========================================================================
# Summarize node
# ===========================================================================


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


def summarize_node(state: dict) -> dict:
    """Load CSV summary; use cached value from memory if available."""
    memory: ConversationMemory = state["memory"]
    csv_path = state["csv_path"]
    if memory.csv_summary is None:
        memory.csv_summary = _build_csv_summary(csv_path)
    return {"csv_summary": memory.csv_summary}


# ===========================================================================
# Codegen node  (memory-aware)
# ===========================================================================

CODEGEN_SYSTEM_FRESH = """You are an expert Python / pandas data analyst.
Write a self-contained Python script that answers the user's question about a CSV file.

Rules:
1. Read the CSV using pandas from the exact path provided.
2. Store the final answer in a variable called `result`.
   `result` must be a DataFrame, Series, scalar, list, or dict.
3. Do NOT use print(), display(), or any I/O besides reading the CSV.
4. Only import pandas as pd and standard library modules.
5. Return ONLY raw Python — no markdown, no explanation.
6. Use the dataset summary for correct column names and null handling.
7. For comparison or ranking questions (e.g. "which X has the highest/lowest Y",
   "top N by Z", "compare X across Y"), store ALL groups and their aggregated
   values in `result` as a DataFrame — not just the single top/bottom entry.
   A downstream component will identify the winner from the full table.
"""

CODEGEN_USER_FRESH = """CSV path: {csv_path}

Dataset summary:
{csv_summary}

Conversation history:
{history}

Question: {question}

Write the Python code now."""


CODEGEN_SYSTEM_FOLLOWUP = """You are an expert Python / pandas data analyst.
The user is asking a follow-up question. Instead of re-reading the original CSV,
you must operate on `prior_result`, which is already defined for you.

Rules:
1. Do NOT read any CSV file.
2. Use `prior_result` as your input — it is already defined above your code.
3. Store the final answer in a variable called `result`.
4. Only import pandas as pd and standard library modules (no other imports).
5. Return ONLY raw Python — no markdown, no explanation.
6. Use the EXACT column names shown in the prior result preview below.
   Do not guess or rename columns.
7. `result` should be a DataFrame or Series whenever the answer involves
   multiple rows or categories — never collapse multi-row data into a single
   scalar or dict unless the question explicitly asks for one number.
"""

CODEGEN_USER_FOLLOWUP = """Prior result — structure and sample data:
{result_preview}

Conversation history:
{history}

Follow-up question: {question}

Write Python code that operates on `prior_result` and stores the answer in `result`."""


def codegen_node(state: dict) -> dict:
    memory: ConversationMemory = state["memory"]
    is_followup = state.get("is_followup", False)

    if is_followup and memory.has_prior_result():
        prior_serialized = memory.get_last_result_serialized()
        prior_code = ConversationMemory.result_to_code_snippet(prior_serialized)
        preview = ConversationMemory.result_preview(prior_serialized)
        prompt = CODEGEN_USER_FOLLOWUP.format(
            result_preview=preview,
            history=memory.get_history_text(),
            question=state["question"],
        )
        messages = [
            {"role": "system", "content": CODEGEN_SYSTEM_FOLLOWUP},
            {"role": "user", "content": prompt},
        ]
    else:
        prior_code = None
        prompt = CODEGEN_USER_FRESH.format(
            csv_path=state["csv_path"],
            csv_summary=state.get("csv_summary", "(no summary available)"),
            history=memory.get_history_text(),
            question=state["question"],
        )
        messages = [
            {"role": "system", "content": CODEGEN_SYSTEM_FRESH},
            {"role": "user", "content": prompt},
        ]

    code = call_llm(messages)
    code = re.sub(r"^```(?:python)?\n?", "", code)
    code = re.sub(r"\n?```$", "", code)

    # For follow-ups, prepend the prior_result reconstruction so `prior_result`
    # is actually defined when exec() runs the LLM-generated code.
    if prior_code is not None:
        code = prior_code + "\n\n" + code

    return {"generated_code": code}


# ===========================================================================
# Execute node
# ===========================================================================


def execute_node(state: dict) -> dict:
    env: dict = {}
    try:
        exec(state["generated_code"], env)
        execution_result = env.get("result", None)
        execution_error = None
    except Exception:
        execution_result = None
        execution_error = traceback.format_exc()
    return {"execution_result": execution_result, "execution_error": execution_error}


# ===========================================================================
# Evaluate node
# ===========================================================================

EVALUATE_SYSTEM = """You are a strict code-output evaluator.
Given a question and the execution result of some Python code, decide whether
the result correctly and meaningfully answers the question.

Reply with exactly one word: PASS or FAIL.

Return FAIL if:
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
    result_str = str(state.get("execution_result", ""))[:3000]
    error_str = str(state.get("execution_error", "None"))
    verdict = call_llm(
        [
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


# ===========================================================================
# Respond node
# ===========================================================================

RESPOND_SYSTEM = """You are a clear, concise data analyst.
Given a question, the raw execution result from a pandas analysis, and a
dataset summary, write a final human-readable answer.

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
    result_str = str(state.get("execution_result", "No result available."))[:3000]
    final_answer = call_llm(
        [
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


# ===========================================================================
# Visualize node
# ===========================================================================

VIZ_DECIDE_SYSTEM = """You are a data visualization expert.
Given a user question and the execution result of a data analysis,
decide whether a visualization would be helpful and if so, what kind.

Respond with ONLY a JSON object (no markdown) with this schema:
{
  "should_visualize": true | false,
  "chart_type": "bar" | "line" | "scatter" | "pie" | "histogram" | "box" | null,
  "reason": "<one sentence>"
}

Visualization is useful for: comparisons, distributions, trends, rankings, proportions.
Visualization is NOT useful for: single scalar values, raw text answers, yes/no results.
"""

VIZ_DECIDE_USER = """Question: {question}
Final answer: {final_answer}

Prior result structure:
{result_preview}

Should we visualize this?"""

VIZ_CODE_SYSTEM = """You are an expert at writing Plotly Python code.
Write code that creates a Plotly figure from `prior_result` (already defined).

Rules:
1. Use plotly.graph_objects or plotly.express — import as needed.
2. Store the figure in a variable called `fig`.
3. Make the figure look clean and professional with a proper title.
4. Do NOT call fig.show() or write any files.
5. Return ONLY raw Python code — no markdown, no explanation.
6. Use ONLY the exact column names listed in the prior result schema below.
   Do not guess, rename, or invent column names.
7. Use a dark theme: layout bgcolor='#111111', paper_bgcolor='#111111', font color='#e0e0e0'.
"""

VIZ_CODE_USER = """Prior result — exact structure you must use:
{result_preview}

Question: {question}
Chart type: {chart_type}

Write Plotly code that creates a `fig` from `prior_result`.
Use only the column names shown above — do not guess."""


def viz_node(state: dict) -> dict:
    """
    1. Serialize the execution result and build a schema preview.
    2. Pass the preview to the LLM to decide whether/how to visualize.
    3. Pass the same preview to the LLM to generate Plotly code with
       correct column names.
    4. Prepend the result reconstruction snippet, execute, capture fig JSON.
    """
    execution_result = state.get("execution_result")

    # Serialize once — reused for preview, reconstruction, and both LLM calls
    serialized = ConversationMemory.serialize_result(execution_result)
    preview = ConversationMemory.result_preview(serialized)
    schema = ConversationMemory.get_schema(serialized)

    # Step 1: decide whether to visualize
    decide_raw = call_llm(
        [
            {"role": "system", "content": VIZ_DECIDE_SYSTEM},
            {
                "role": "user",
                "content": VIZ_DECIDE_USER.format(
                    question=state["question"],
                    final_answer=state.get("final_answer", ""),
                    result_preview=preview
                    + "\n\nSCHEMA:\n"
                    + json.dumps(schema, indent=2),
                ),
            },
        ]
    )
    try:
        decide_raw = re.sub(r"^```(?:json)?\n?", "", decide_raw)
        decide_raw = re.sub(r"\n?```$", "", decide_raw)
        decision = json.loads(decide_raw)
    except Exception:
        return {"viz_json": None, "viz_decision": None}

    if not decision.get("should_visualize"):
        return {"viz_json": None, "viz_decision": decision}

    chart_type = decision.get("chart_type", "bar")

    # Step 2: generate Plotly code — preview tells LLM exact column names
    prior_code = ConversationMemory.result_to_code_snippet(serialized)
    viz_code_raw = call_llm(
        [
            {"role": "system", "content": VIZ_CODE_SYSTEM},
            {
                "role": "user",
                "content": VIZ_CODE_USER.format(
                    result_preview=preview
                    + "\n\nSCHEMA:\n"
                    + json.dumps(schema, indent=2),
                    question=state["question"],
                    chart_type=chart_type,
                ),
            },
        ]
    )
    viz_code_raw = re.sub(r"^```(?:python)?\n?", "", viz_code_raw)
    viz_code_raw = re.sub(r"\n?```$", "", viz_code_raw)

    # Step 3: prepend reconstruction so `prior_result` is defined, then exec
    env: dict = {}
    try:
        exec(prior_code + "\n\n" + viz_code_raw, env)  # noqa: S102
        fig = env.get("fig")
        viz_json = fig.to_json() if fig is not None else None
    except Exception:
        viz_json = None
        decision["error"] = traceback.format_exc()

    return {"viz_json": viz_json, "viz_decision": decision}
