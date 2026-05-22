from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Reusable sub-schemas
# ---------------------------------------------------------------------------


class FilterCondition(BaseModel):
    """A single predicate. Conditions within a step are ANDed together by default."""

    column: str
    operator: Literal[
        "==",
        "!=",
        ">",
        ">=",
        "<",
        "<=",
        "in",
        "not_in",
        "contains",
        "not_contains",
        "startswith",
        "endswith",
        "is_null",
        "is_not_null",
    ]
    # None is valid for is_null / is_not_null; list for in / not_in
    value: Optional[Union[str, int, float, bool, list]] = None
    conjunction: Literal["AND", "OR"] = "AND"  # joins this condition with the next


class DerivedColumn(BaseModel):
    new_column: str  # output column name
    left_source_column: str
    right_source_column: str
    operation: Literal["add", "subtract", "multiply", "divide"]


class Aggregation(BaseModel):
    source_column: str  # source column; use "*" for count(*)
    function: Literal[
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "count",
        "nunique",
        "std",
        "var",
        "first",
        "last",
    ]
    new_column: str  # output column name


class SortKey(BaseModel):
    column: str
    direction: Literal["asc", "desc"] = "asc"


# ---------------------------------------------------------------------------
# Step op-codes  (discriminated on the `op` field)
# ---------------------------------------------------------------------------


class SelectColumnsStep(BaseModel):
    """Keep only the specified columns, in the given order."""

    op: Literal["select_columns"] = "select_columns"
    columns: list[str]


class RenameColumnsStep(BaseModel):
    """Rename columns. Keys are current names, values are new names."""

    op: Literal["rename_columns"] = "rename_columns"
    mapping: dict[str, str]


class FilterRowsStep(BaseModel):
    """Retain rows that satisfy all (or any, per conjunction) conditions."""

    op: Literal["filter_rows"] = "filter_rows"
    conditions: list[FilterCondition]


class DeriveColumnsStep(BaseModel):
    """Add or overwrite columns via expressions evaluated against the current frame."""

    op: Literal["derive_columns"] = "derive_columns"
    columns: list[DerivedColumn]


class GroupAggregateStep(BaseModel):
    """GROUP BY group_by columns, then apply each aggregation. Reduces to one row per unique group.
    If group_by is empty, the entire DataFrame is treated as one group, yielding a single row.
    """

    op: Literal["group_aggregate"] = "group_aggregate"
    group_by: list[str]
    aggregations: list[Aggregation]


class SortRowsStep(BaseModel):
    """Order rows. Keys are applied left-to-right (primary → secondary → …)."""

    op: Literal["sort_rows"] = "sort_rows"
    sort_by: list[SortKey]


class LimitRowsStep(BaseModel):
    """Return at most n rows, skipping the first offset rows."""

    op: Literal["limit_rows"] = "limit_rows"
    n: int = Field(gt=0)
    offset: int = Field(default=0, ge=0)


class DistinctRowsStep(BaseModel):
    """Drop duplicate rows. If columns is None, consider all columns."""

    op: Literal["distinct_rows"] = "distinct_rows"
    columns: Optional[list[str]] = None


class PivotStep(BaseModel):
    """Pivot: rows become columns. Equivalent to a spreadsheet pivot table."""

    op: Literal["pivot"] = "pivot"
    index: list[str]  # row identifiers
    columns: str  # column whose values become new column headers
    values: str  # column to aggregate
    aggfunc: Literal["sum", "mean", "count", "min", "max"] = "sum"


class SnapshotStep(BaseModel):
    """Save the current DataFrame into the snapshot store under a name. Current df is unchanged."""

    op: Literal["snapshot"] = "snapshot"
    name: str


class RestoreStep(BaseModel):
    """Replace the current DataFrame with a previously saved snapshot."""

    op: Literal["restore"] = "restore"
    name: str


class JoinStep(BaseModel):
    """Merge the current DataFrame (left) with a named snapshot (right)."""

    op: Literal["join"] = "join"
    right: str
    on: list[str]
    how: Literal["inner", "left", "right", "outer"] = "left"
    suffixes: tuple[str, str] = ("_x", "_y")


# ---------------------------------------------------------------------------
# Step union  (the unified classes defining deterministic execution steps)
# ---------------------------------------------------------------------------


# The discriminated union — Pydantic resolves the correct subclass from `op`
Step = Annotated[
    Union[
        SelectColumnsStep,
        RenameColumnsStep,
        FilterRowsStep,
        DeriveColumnsStep,
        GroupAggregateStep,
        SortRowsStep,
        LimitRowsStep,
        DistinctRowsStep,
        PivotStep,
        SnapshotStep,
        RestoreStep,
        JoinStep,
    ],
    Field(discriminator="op"),
]


# ---------------------------------------------------------------------------
# Trace entry  (one per executed step)
# ---------------------------------------------------------------------------


class TraceEntry(BaseModel):
    step_index: int
    op: str
    input_shape: tuple[int, int]  # (rows, cols) before the step
    output_shape: tuple[int, int]  # (rows, cols) after the step
    error: Optional[str] = None  # set if execution failed
