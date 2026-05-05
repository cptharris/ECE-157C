from typing import Dict, Any
import pandas as pd


def apply_op(df: pd.DataFrame, op: str, params: State):
    if op is None:
        return df
    if op == "select_columns":
        cols = params.get("columns", [])
        return df[cols].copy()
    if op == "filter":
        query = params.get("query") or params.get("condition")
        if query:
            return df.query(query)
        return df
    if op == "groupby_agg":
        by = params.get("by", [])
        agg = params.get("agg", {})
        if by:
            return df.groupby(by).agg(agg).reset_index()
        return df
    if op == "describe":
        return df.describe()
    if op == "sort":
        by = params.get("by", [])
        ascending = params.get("ascending", True)
        return df.sort_values(by=by, ascending=ascending)
    if op == "head":
        n = params.get("n", 5)
        return df.head(n)
    if op == "tail":
        n = params.get("n", 5)
        return df.tail(n)
    # Default: no-op
    return df


def execute_node(state: State) -> State:
    try:
        df = pd.read_csv("dataset.csv")

        steps = state.get("steps", []) or []
        trace = []

        for i, step in enumerate(steps):
            op = None
            params = {}
            if isinstance(step, dict):
                op = step.get("op")
                params = step.get("params", {})
            else:
                op = getattr(step, "op", None)
                params = getattr(step, "params", {}) if hasattr(step, "params") else {}

            before = len(df)
            df = apply_op(df, op, params)
            after = len(df) if df is not None else 0
            trace.append(f"step_{i}: {op} input_rows={before} output_rows={after}")

        state["trace"] = trace
        state["execution_result"] = f"Executed {len(steps)} steps on {csv_path}"
        state["execution_error"] = None
        state["generated_code"] = ""
        state["final_answer"] = state.get("final_answer", "")
        state["evaluation"] = "SUCCESS"
    except Exception as e:
        state["execution_error"] = str(e)
        state["execution_result"] = None
        state["generated_code"] = ""
        state["final_answer"] = ""
        state["trace"] = []
        state["evaluation"] = "FAIL"
    return state
