from langchain_core.runnables import RunnableConfig
from typing import List, Any
import pandas as pd
import traceback

from schemas import GraphState

from plan_schemas import TraceEntry, DatasetEntry
from plan_operators import dispatch_op

from plan_schemas import SnapshotStep, RestoreStep, JoinStep, LoadDatasetStep
from plan_operators import _snapshot, _restore, _join, _load_dataset


def plan_execute_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    execution_result = None
    trace_entries: List[TraceEntry] = []
    snapshots: dict[str, pd.DataFrame] = {}

    dataset_registry = {}
    for csv_path in state["csv_paths"]:
        dataset_registry[csv_path] = DatasetEntry(
            csv_path=csv_path, valid=False, dataframe=None
        )

    try:
        df = _load_dataset(dataset_registry, state["csv_paths"][0])

        for i, step in enumerate(state["plan_execute_result"]["plan"].steps):
            before = df.shape

            try:
                if isinstance(step, (SnapshotStep, RestoreStep, JoinStep)):
                    df = {
                        SnapshotStep: _snapshot,
                        RestoreStep: _restore,
                        JoinStep: _join,
                    }[type(step)](df, step, snapshots)
                elif isinstance(step, LoadDatasetStep):
                    df = _load_dataset(dataset_registry, step.dataset_name)
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
                print(
                    f"error at step {i}: {step.model_dump_json(indent=2, ensure_ascii=False)}"
                )
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

        execution_result = df.to_dict(orient="records")

    except Exception as e:
        traceback.print_exc()
        execution_result = str(e)

    updated: PlanExecuteResult = {
        **state["plan_execute_result"],
        "trace": trace_entries,
        "execution_result": execution_result,
    }

    return {
        "plan_execute_result": updated,
    }
