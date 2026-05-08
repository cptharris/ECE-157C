from typing import List
import pandas as pd
import traceback

from schemas import AgentState, TraceEntry
from operators import dispatch_op

from schemas import SnapshotStep, RestoreStep, JoinStep
from operators import _snapshot, _restore, _join


def execute_node(state: AgentState) -> AgentState:
    state["run_count"] += 1
    trace_entries: List[TraceEntry] = []
    snapshots: dict[str, pd.DataFrame] = {}

    try:
        df = pd.read_csv(state["csv_path"])

        for i, step in enumerate(state["plan"].steps):
            before = df.shape

            try:
                if isinstance(step, (SnapshotStep, RestoreStep, JoinStep)):
                    df = {
                        SnapshotStep: _snapshot,
                        RestoreStep: _restore,
                        JoinStep: _join,
                    }[type(step)](df, step, snapshots)
                else:
                    df = dispatch_op(df, step)

                trace_entries.append(
                    TraceEntry(
                        step_index=i,
                        op=step.op,
                        input_shape=before,
                        output_shape=df.shape,
                        error=None,
                    )
                )

            except Exception as e:
                traceback.print_exc()

                trace_entries.append(
                    TraceEntry(
                        step_index=i,
                        op=step.op,
                        input_shape=before,
                        output_shape=None,
                        error=str(e),
                    )
                )

        state["execution_result"] = df.to_dict(orient="records")

    except Exception as e:
        traceback.print_exc()
        state["execution_result"] = str(e)

    state["trace"] = trace_entries
    return state
