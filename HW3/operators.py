import pandas as pd
from typing import Any, Dict, List
from schemas import *


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
        value = cond.value
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


def _group_aggregate(df: pd.DataFrame, step: GroupAggregateStep) -> pd.DataFrame:
    by = step.group_by
    aggs = step.aggregations
    if not by:
        return df
    result = df.copy()
    for agg in aggs:
        src = agg.source_column
        func = agg.function
        new_col = agg.new_column
        if src is None or func is None or new_col is None:
            continue
        tmp = df.groupby(by)[src].agg(func).reset_index().rename(columns={src: new_col})
        result = result.merge(tmp, on=by, how="left")
    return result


def _sort_rows(df: pd.DataFrame, step: SortRowsStep) -> pd.DataFrame:
    sort_by = step.sort_by
    if not sort_by:
        return df
    keys = [sk.column for sk in sort_by]
    dirs = [sk.direction for sk in sort_by]
    ascending = [d == "asc" for d in dirs]
    return df.sort_values(by=keys, ascending=ascending)


def _limit_rows(df: pd.DataFrame, step: LimitRowsStep) -> pd.DataFrame:
    return df.iloc[step.offset: step.offset + step.n]


def _distinct_rows(df: pd.DataFrame, step: DistinctRowsStep) -> pd.DataFrame:
    cols = step.columns
    if cols:
        return df.drop_duplicates(subset=cols)
    return df.drop_duplicates()


def _pivot(df: pd.DataFrame, step: PivotStep) -> pd.DataFrame:
    if not step.index or not step.columns or not step.values:
        return df
    return df.pivot_table(index=step.index, columns=step.columns, values=step.values, aggfunc=step.aggfunc).reset_index()


def dispatch_op(df: pd.DataFrame, step: Step) -> pd.DataFrame:
    if isinstance(step, SelectColumnsStep):
        return _select_columns(df, step)
    if isinstance(step, RenameColumnsStep):
        return _rename_columns(df, step)
    if isinstance(step, FilterRowsStep):
        return _filter_rows(df, step)
    if isinstance(step, DeriveColumnsStep):
        return _derive_columns(df, step)
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
