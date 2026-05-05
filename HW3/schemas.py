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
        "==", "!=", ">", ">=", "<", "<=",
        "in", "not_in",
        "contains", "not_contains",
        "startswith", "endswith",
        "is_null", "is_not_null",
    ]
    # None is valid for is_null / is_not_null; list for in / not_in
    value: Optional[Union[str, int, float, bool, list]] = None
    conjunction: Literal["AND", "OR"] = "AND"  # joins this condition with the next


class DeriveColumn(BaseModel):
    new_column: str             # output column name
    left_source_column: str
    right_source_column: str
    operation: Literal[
        "add", "subtract",
        "multiply", "divide"
    ]


class Aggregation(BaseModel):
    source_column: str                      # source column; use "*" for count(*)
    function: Literal[
        "sum", "mean", "median", "min", "max",
        "count", "count_distinct",
        "std", "var", "first", "last",
    ]
    new_column: str                       # output column name


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
    """GROUP BY group_by columns, then apply each aggregation."""
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
    index: list[str]          # row identifiers
    columns: str              # column whose values become new column headers
    values: str               # column to aggregate
    aggfunc: Literal["sum", "mean", "count", "min", "max"] = "sum"


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
    ],
    Field(discriminator="op"),
]


# ---------------------------------------------------------------------------
# Trace entry  (one per executed step)
# ---------------------------------------------------------------------------

class TraceEntry(BaseModel):
    step_index: int
    op: str
    input_shape: tuple[int, int]   # (rows, cols) before the step
    output_shape: tuple[int, int]  # (rows, cols) after the step
    error: Optional[str] = None    # set if execution failed
    duration_ms: Optional[float] = None

    def __str__(self):
        return f"{str(self.step_index).ljust(5)} | {self.op.ljust(15)} | {str(self.input_shape[0]).ljust(10)} | {str(self.output_shape[0]).ljust(12)}"


# ---------------------------------------------------------------------------
# Plan  (what the planner node produces)
# ---------------------------------------------------------------------------

class Plan(BaseModel):
    reasoning: str        # chain-of-thought: why these steps, in this order
    steps: list[Step]


# ---------------------------------------------------------------------------
# LangGraph state  (TypedDict so LangGraph can handle reducers cleanly)
# ---------------------------------------------------------------------------
# Note: DataFrames are not JSON-serializable, so we do NOT store them in state.
# The executor replays all steps from the original DataFrame on each invocation,
# which keeps state serializable and checkpointing free. For large datasets
# consider caching the frame externally and storing only a reference key.

from typing import TypedDict

class AgentState(TypedDict):
    question: str
    csv_path: str
    dataset_description: str       # filled by describe_dataset_node
    plan: Optional[Plan]           # filled by planner_node
    trace: list[TraceEntry]        # appended to by executor_node
    final_answer: Optional[str]    # filled by responder_node
    retry_count: int           # tracks re-plan attempts
    max_retries: int           # set at invocation time, e.g. 2


# ---------------------------------------------------------------------------
# Format Node
# ---------------------------------------------------------------------------


def format_node(state: State) -> State:
    state["trace"] = ["step  | operation       | input rows | output rows "] \
        + ["-"*51] \
        + [e.__str__() for e in state["trace"]]

    return state
