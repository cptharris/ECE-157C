from langchain_core.runnables import RunnableConfig
from typing import List, Any
import pandas as pd
import traceback
import json

from schemas import GraphState

from plan_schemas import TraceEntry, DatasetEntry
from plan_operators import dispatch_op

from plan_schemas import (
    SnapshotStep,
    RestoreStep,
    JoinStep,
    ConcatStep,
    LoadDatasetStep,
    DisplayStep,
)
from plan_operators import _snapshot, _restore, _join, _concat, _load_dataset


def plan_execute_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    execution_result = ""
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
                if isinstance(step, (SnapshotStep, RestoreStep, JoinStep, ConcatStep)):
                    df = {
                        SnapshotStep: _snapshot,
                        RestoreStep: _restore,
                        JoinStep: _join,
                        ConcatStep: _concat,
                    }[type(step)](df, step, snapshots)
                elif isinstance(step, LoadDatasetStep):
                    df = _load_dataset(dataset_registry, step.dataset_name)
                elif isinstance(step, DisplayStep):
                    execution_result += str(df.to_dict(step.orient))
                else:
                    df = dispatch_op(df, step)
                
                df = df.copy()

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
