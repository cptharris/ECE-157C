import pandas as pd
import traceback

from schemas import AgentState, TraceEntry
from operators import dispatch_op


def execute_node(state: AgentState) -> AgentState:
    try:
        csv_path = state["csv_path"]
        df = pd.read_csv(csv_path)

        trace_entries: List[TraceEntry] = []

        for i, step in enumerate(state["plan"].steps):
            try:
                before = df.shape
                df = dispatch_op(df, step)
                after = df.shape
            except Exception as e:
                trace_entries.append(TraceEntry(
                    step_index=i, op=op, input_shape=before, output_shape=after, error=str(e)
                    ))

        state["trace"] = trace_entries
        state["execution_result"] = df.to_dict(orient="records")
    except Exception as e:
        state["execution_result"] = str(e)
        state["trace"] = trace_entries

    return state
