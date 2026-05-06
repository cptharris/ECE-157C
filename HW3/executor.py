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
            error = ""
            before = df.shape

            try:
                df = dispatch_op(df, step)
            except Exception as e:
                error = str(e)
                traceback.print_exception(e)

            after = df.shape

            trace_entries.append(
                TraceEntry(
                    step_index=i,
                    op=step.op,
                    input_shape=before,
                    output_shape=after,
                    error=error,
                )
            )

        state["trace"] = trace_entries
        state["execution_result"] = df.to_dict(orient="records")
    except Exception as e:
        state["execution_result"] = str(e)
        state["trace"] = trace_entries
        traceback.print_exception(e)

    return state
