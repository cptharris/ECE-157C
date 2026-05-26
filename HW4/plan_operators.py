import pandas as pd
import numpy as np
from typing import Any, Dict, List
from plan_schemas import *


def _select_columns(df: pd.DataFrame, step: SelectColumnsStep) -> pd.DataFrame:
    cols = step.columns
    return df[cols].copy()


def _rename_columns(df: pd.DataFrame, step: RenameColumnsStep) -> pd.DataFrame:
    return df.rename(columns=step.mapping)


def _filter_rows(df: pd.DataFrame, step: FilterRowsStep) -> pd.DataFrame:
    mask = None
    for cond in step.conditions:
        column = cond.column
        operator = cond.operator
        value = (
            df[cond.value].astype(object)
            if (not isinstance(cond.value, list) and cond.value in df.columns)
            else cond.value
        )
        conjunction = cond.conjunction
        ser = df[column].astype(object)
        if operator == "==":
            pred = ser == value
        elif operator == "!=":
            pred = ser != value
        elif operator == ">":
            pred = ser > value
        elif operator == ">=":
            pred = ser >= value
        elif operator == "<":
            pred = ser < value
        elif operator == "<=":
            pred = ser <= value
        elif operator == "in":
            pred = ser.isin(value if isinstance(value, list) else [value])
        elif operator == "not_in":
            pred = ~ser.isin(value if isinstance(value, list) else [value])
        elif operator == "contains":
            pred = ser.astype(str).str.contains(str(value))
        elif operator == "not_contains":
            pred = ~ser.astype(str).str.contains(str(value))
        elif operator == "startswith":
            pred = ser.astype(str).str.startswith(str(value))
        elif operator == "endswith":
            pred = ser.astype(str).str.endswith(str(value))
        elif operator == "is_null":
            pred = ser.isnull()
        elif operator == "is_not_null":
            pred = ~ser.isnull()
        else:
            pred = pd.Series([True] * len(df), index=df.index)

        if mask is None:
            mask = pred
        else:
            if conjunction.upper() == "AND":
                mask = mask & pred
            else:
                mask = mask | pred
    return df[mask].copy()


def _derive_columns(df: pd.DataFrame, step: DeriveColumnsStep) -> pd.DataFrame:
    for c in step.columns:
        new_col = c.new_column
        left = c.left_source_column
        right = c.right_source_column
        op = c.operation
        if op == "add":
            df[new_col] = df[left] + df[right]
        elif op == "subtract":
            df[new_col] = df[left] - df[right]
        elif op == "multiply":
            df[new_col] = df[left] * df[right]
        elif op == "divide":
            df[new_col] = df[left] / df[right]
    return df


def _conditional_columns(
    df: pd.DataFrame,
    step: ConditionalColumnsStep,
) -> pd.DataFrame:
    for c in step.columns:
        left = df[c.left_source_column]

        if isinstance(c.right_source, str) and c.right_source in df.columns:
            right = df[c.right_source]
        else:
            right = c.right_source

        if c.operator == "==":
            pred = left == right
        elif c.operator == "!=":
            pred = left != right
        elif c.operator == ">":
            pred = left > right
        elif c.operator == ">=":
            pred = left >= right
        elif c.operator == "<":
            pred = left < right
        elif c.operator == "<=":
            pred = left <= right
        else:
            raise ValueError(f"Unsupported conditional operator: {c.operator}")

        df[c.new_column] = np.where(pred, c.true_value, c.false_value)

    return df


_STAR_COL = "*"


def _group_aggregate(df: pd.DataFrame, step: GroupAggregateStep) -> pd.DataFrame:
    by = step.group_by
    aggs = step.aggregations

    has_star = any(agg.source_column == "*" for agg in aggs if agg.source_column)
    if has_star:
        df = df.assign(**{_STAR_COL: 1})

    agg_spec = {
        agg.new_column: (
            (_STAR_COL if agg.source_column == "*" else agg.source_column),
            agg.function,
        )
        for agg in aggs
        if agg.source_column and agg.function and agg.new_column
    }

    if not agg_spec:
        return df[by].drop_duplicates().reset_index(drop=True) if by else df.iloc[:0]

    if by:
        return df.groupby(by, as_index=False).agg(**agg_spec)
    else:
        return df.agg(
            **{col: (src, func) for col, (src, func) in agg_spec.items()}
        ).T.reset_index(drop=True)


def _sort_rows(df: pd.DataFrame, step: SortRowsStep) -> pd.DataFrame:
    sort_by = step.sort_by
    if not sort_by:
        return df
    keys = [sk.column for sk in sort_by]
    dirs = [sk.direction for sk in sort_by]
    ascending = [d == "asc" for d in dirs]
    return df.sort_values(by=keys, ascending=ascending)


def _limit_rows(df: pd.DataFrame, step: LimitRowsStep) -> pd.DataFrame:
    return df.iloc[step.offset : step.offset + step.n]


def _distinct_rows(df: pd.DataFrame, step: DistinctRowsStep) -> pd.DataFrame:
    cols = step.columns
    if cols:
        return df.drop_duplicates(subset=cols)
    return df.drop_duplicates()


def _pivot(df: pd.DataFrame, step: PivotStep) -> pd.DataFrame:
    if not step.index or not step.columns or not step.values:
        return df
    return df.pivot_table(
        index=step.index, columns=step.columns, values=step.values, aggfunc=step.aggfunc
    ).reset_index()


def _snapshot(
    df: pd.DataFrame,
    step: SnapshotStep,
    snapshots: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    snapshots[step.name] = df.copy()
    return df


def _restore(
    df: pd.DataFrame,
    step: RestoreStep,
    snapshots: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if step.name not in snapshots:
        raise KeyError(
            f"Snapshot '{step.name}' not found. Available: {list(snapshots)}"
        )
    return snapshots[step.name].copy()


def _join(
    df: pd.DataFrame,
    step: JoinStep,
    snapshots: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if step.right not in snapshots:
        raise KeyError(
            f"Snapshot '{step.right}' not found. Available: {list(snapshots)}"
        )
    right_df = snapshots[step.right]
    return df.merge(right_df, on=step.on, how=step.how, suffixes=step.suffixes)


def _load_dataset(
    dataset_registry: dict[str, DatasetEntry],
    dataset_name: str,
) -> pd.DataFrame:
    if dataset_name not in dataset_registry:
        raise KeyError(f"Dataset '{dataset_name}' not found.")
    
    if dataset_registry[dataset_name]["dataframe"] is not None:
        return dataset_registry[dataset_name]["dataframe"]
    
    dataset_registry[dataset_name]["dataframe"] = pd.read_csv("datasets/" + dataset_registry[dataset_name]["csv_path"])

    return dataset_registry[dataset_name]["dataframe"]


def dispatch_op(df: pd.DataFrame, step: Step) -> pd.DataFrame:
    if isinstance(step, SelectColumnsStep):
        return _select_columns(df, step)
    if isinstance(step, RenameColumnsStep):
        return _rename_columns(df, step)
    if isinstance(step, FilterRowsStep):
        return _filter_rows(df, step)
    if isinstance(step, DeriveColumnsStep):
        return _derive_columns(df, step)
    if isinstance(step, ConditionalColumnsStep):
        return _conditional_columns(df, step)
    if isinstance(step, GroupAggregateStep):
        return _group_aggregate(df, step)
    if isinstance(step, SortRowsStep):
        return _sort_rows(df, step)
    if isinstance(step, LimitRowsStep):
        return _limit_rows(df, step)
    if isinstance(step, DistinctRowsStep):
        return _distinct_rows(df, step)
    if isinstance(step, PivotStep):
        return _pivot(df, step)
    raise TypeError(f"Unsupported Step type: {type(step)}")
