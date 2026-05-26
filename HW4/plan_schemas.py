from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union, TypedDict
from pydantic import BaseModel, Field
import pandas as pd


# ---------------------------------------------------------------------------
# Reusable sub-schemas
# ---------------------------------------------------------------------------


class FilterCondition(BaseModel):
    """A single predicate. Conditions within a step are ANDed together by default."""

    column: str = Field(
        description="Name of the column whose values are evaluated by this predicate."
    )
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
    ] = Field(
        description=(
            "Comparison operation applied to the column. "
            "Use equality/inequality for scalar comparisons, in/not_in for membership, "
            "contains/startswith/endswith for string matching, and is_null/is_not_null "
            "for missing-value checks."
        )
    )
    value: Optional[Union[str, int, float, bool, list]] = Field(
        default=None,
        description=(
            "Comparison value used by the operator (can also be a string column name). "
            "Use a list for in/not_in. Leave as None for is_null/is_not_null."
        ),
    )
    conjunction: Literal["AND", "OR"] = Field(
        default="AND",
        description=(
            "Logical connector joining this condition to the next condition in the list."
        ),
    )


class DerivedColumn(BaseModel):
    new_column: str = Field(
        description="Name of the derived output column to create or overwrite."
    )
    left_source_column: str = Field(
        description="Left-hand input column used in the operation."
    )
    right_source: Union[str, int, float, bool] = Field(
        description="Right-hand input column name or scalar literal used in the operation."
    )
    operation: Literal[
        "add",
        "subtract",
        "multiply",
        "divide",
        "==",
        "!=",
        ">",
        ">=",
        "<",
        "<=",
    ] = Field(
        description="Arithmetic or comparison operation applied between the left and right inputs."
    )
    true_value: Union[int, float, bool, str] = Field(
        default=1,
        description="Value assigned when a comparison operation evaluates to true."
    )
    false_value: Union[int, float, bool, str] = Field(
        default=0,
        description="Value assigned when a comparison operation evaluates to false."
    )


class Aggregation(BaseModel):
    source_column: str = Field(
        description=("Column to aggregate. Use '*' to represent count(*).")
    )
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
    ] = Field(description="Aggregation function applied to the source column.")
    new_column: str = Field(
        description="Name of the output column containing the aggregation result."
    )


class SortKey(BaseModel):
    column: str = Field(description="Column used as a sorting key.")
    direction: Literal["asc", "desc"] = Field(
        default="asc",
        description="Sort direction for the key: ascending or descending.",
    )


# ---------------------------------------------------------------------------
# Step op-codes  (discriminated on the `op` field)
# ---------------------------------------------------------------------------


class SelectColumnsStep(BaseModel):
    """Keep only the specified columns, in the given order."""

    op: Literal["select_columns"] = Field(
        default="select_columns",
        description="Operation identifier for selecting a subset of columns.",
    )
    columns: list[str] = Field(
        description="Ordered list of columns to retain in the DataFrame."
    )


class RenameColumnsStep(BaseModel):
    """Rename columns. Keys are current names, values are new names."""

    op: Literal["rename_columns"] = Field(
        default="rename_columns",
        description="Operation identifier for renaming columns.",
    )
    mapping: dict[str, str] = Field(
        description="Dictionary mapping existing column names to new column names."
    )


class FilterRowsStep(BaseModel):
    """Retain rows that satisfy all (or any, per conjunction) conditions."""

    op: Literal["filter_rows"] = Field(
        default="filter_rows", description="Operation identifier for row filtering."
    )
    conditions: list[FilterCondition] = Field(
        min_length=1,
        description="Ordered list of filter predicates evaluated against the DataFrame.",
    )


class DeriveColumnsStep(BaseModel):
    """Add or overwrite columns via expressions evaluated against the current frame."""

    op: Literal["derive_columns"] = Field(
        default="derive_columns",
        description="Operation identifier for derived-column creation.",
    )
    columns: list[DerivedColumn] = Field(
        min_length=1, description="Definitions of all derived columns to compute."
    )


class GroupAggregateStep(BaseModel):
    """GROUP BY group_by columns, then apply each aggregation. Reduces to one row per unique group.
    If group_by is empty, the entire DataFrame is treated as one group, yielding a single row.
    """

    op: Literal["group_aggregate"] = Field(
        default="group_aggregate",
        description="Operation identifier for grouped aggregation.",
    )
    group_by: list[str] = Field(
        description="Columns used to define grouping partitions before aggregation."
    )
    aggregations: list[Aggregation] = Field(
        min_length=1, description="Aggregation computations applied within each group."
    )


class SortRowsStep(BaseModel):
    """Order rows. Keys are applied left-to-right (primary → secondary → …)."""

    op: Literal["sort_rows"] = Field(
        default="sort_rows", description="Operation identifier for sorting rows."
    )
    sort_by: list[SortKey] = Field(
        min_length=1,
        description="Ordered list of sorting keys applied from highest to lowest priority.",
    )


class LimitRowsStep(BaseModel):
    """Return at most n rows, skipping the first offset rows."""

    op: Literal["limit_rows"] = Field(
        default="limit_rows", description="Operation identifier for row limiting."
    )
    n: int = Field(gt=0, description="Maximum number of rows to retain after slicing.")
    offset: int = Field(
        default=0,
        ge=0,
        description="Number of rows to skip before collecting the result window.",
    )


class DistinctRowsStep(BaseModel):
    """Drop duplicate rows. If columns is None, consider all columns."""

    op: Literal["distinct_rows"] = Field(
        default="distinct_rows",
        description="Operation identifier for duplicate-row removal.",
    )
    columns: Optional[list[str]] = Field(
        default=None,
        description=(
            "Subset of columns used to determine uniqueness. "
            "If omitted, the entire row is considered."
        ),
    )


class PivotStep(BaseModel):
    """Pivot: rows become columns. Equivalent to a spreadsheet pivot table."""

    op: Literal["pivot"] = Field(
        default="pivot", description="Operation identifier for pivot-table reshaping."
    )
    index: list[str] = Field(
        description="Columns that remain as row identifiers in the pivoted output."
    )
    columns: str = Field(
        description="Column whose unique values become new output column headers."
    )
    values: str = Field(
        description="Column whose values are aggregated into the pivot table."
    )
    aggfunc: Literal["sum", "mean", "count", "min", "max"] = Field(
        default="sum",
        description="Aggregation function applied during pivot-table construction.",
    )


class SnapshotStep(BaseModel):
    """Save the current DataFrame into the snapshot store under a name. Current df is unchanged."""

    op: Literal["snapshot"] = Field(
        default="snapshot",
        description="Operation identifier for saving an intermediate DataFrame snapshot.",
    )
    name: str = Field(
        description="Unique snapshot identifier used for later restore or join operations."
    )


class RestoreStep(BaseModel):
    """Replace the current DataFrame with a previously saved snapshot."""

    op: Literal["restore"] = Field(
        default="restore",
        description="Operation identifier for restoring a saved snapshot.",
    )
    name: str = Field(
        description="Name of the snapshot to load into the active DataFrame."
    )


class JoinStep(BaseModel):
    """Merge the current DataFrame (left) with a named snapshot (right)."""

    op: Literal["join"] = Field(
        default="join",
        description="Operation identifier for joining against a saved snapshot.",
    )
    right: str = Field(
        description="Name of the snapshot used as the right-hand side of the join."
    )
    on: list[str] = Field(
        min_length=1,
        description="Column names shared between both DataFrames and used as join keys.",
    )
    how: Literal["inner", "left", "right", "outer"] = Field(
        default="left", description="Join strategy controlling row retention semantics."
    )
    suffixes: list[str] = Field(
        default_factory=lambda: ["_x", "_y"],
        min_length=2,
        max_length=2,
        description="Suffixes appended to overlapping non-key columns after the join.",
    )


class LoadDatasetStep(BaseModel):
    """Load a dataset from the available dataset registry."""

    op: Literal["load_dataset"] = Field(
        default="load_dataset",
        description="Operation identifier for loading a dataset.",
    )
    dataset_name: str = Field(
        description="Name of dataset to load from the dataset registry."
    )


class DisplayStep(BaseModel):
    """Append the active dataframe to the execution result output."""

    op: Literal["display"] = Field(
        default="display",
        description="Operation identifier for displaying a dataframe."
    )
    orient: Literal["dict", "list", "series", "split", "records", "index"] = Field(
        default="records",
        description="The orientation direction to display with."
    )

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
        LoadDatasetStep,
        DisplayStep,
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


# ---------------------------------------------------------------------------
# Dataset entry
# ---------------------------------------------------------------------------


class DatasetEntry(TypedDict):
    csv_path: str
    dataframe: Optional[pd.DataFrame] = None
