"""
schemas.py
==========
Schema definitions for the data analytics AI agent.

Design principles
-----------------
- Pydantic BaseModels  →  structured LLM outputs (enforced with .model_validate())
- TypedDicts           →  plain data containers passed through the graph
- Literal types        →  every discriminator field is narrowly typed
- Optional[T]          →  fields absent on entry and filled progressively
- Annotated reducers   →  accumulator fields that must never overwrite

Architecture
------------
A single GraphState is used throughout the main graph and all subgraphs.
Subgraph nodes read and write the same state object; there are no separate
subgraph state classes. This keeps the mental model flat and eliminates
state-mapping boilerplate at subgraph boundaries.

Graph structure
---------------
  orchestrate_node
      │
      ├─ "analytics" ──► analytics subgraph
      │                    init_node → step_node (loop) → answer_node
      │                        │
      │                        ▼
      │                  plan_execute subgraph
      │                    plan_node → execute_node → answer_node
      │                        │
      │                        ▼
      │                  validation_node ──► (approved) ──► finalize_node
      │                        │
      │                        └──► (retry) ──► analytics subgraph
      │
      └─ "generic"  ──► generic subgraph
                          search_node → respond_node
                              │
                              ▼
                          finalize_node

State field lifecycle
---------------------

  Node / subgraph                  Fields written
  ──────────────────────────────── ─────────────────────────────────────────
  [input]                          question, csv_paths
  orchestrate_node                 agent_type
  analytics: init_node             namespace, current_step_index, is_complete
  analytics: step_node             steps (+append), plots (+append),
                                   namespace (updated), current_step_index,
                                   is_complete
  analytics: answer_node           analytics_result
  plan_execute: plan_node          plan_execute_result (plan field)
  plan_execute: execute_node       plan_execute_result (trace, execution_result fields)
  plan_execute: answer_node        plan_execute_result (final_answer field)
  validation_node                  validation_result
  retry edge                       retry_count (+1), validation_feedback,
                                   analytics_result (cleared for re-run),
                                   plan_execute_result (cleared for re-run)
  generic: search_node             generic_result (search_result field)
  generic: respond_node            generic_result (response field)
  finalize_node                    answer, final_plots

Constants
---------
  MAX_ANALYTICS_STEPS  — hard ceiling on analytics loop iterations
  MAX_RETRY_CYCLES     — hard ceiling on validation → retry cycles
  PLOTLY_NAMESPACE_KEY — key injected into the sandbox namespace;
                         analytics code writes completed figures here
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from plan_execute.schemas import Step, TraceEntry

# ---------------------------------------------------------------------------
# Runtime constants
# ---------------------------------------------------------------------------

MAX_ANALYTICS_STEPS: int = 12  # Hard ceiling on analytics loop iterations.
MAX_RETRY_CYCLES: int = 2  # Hard ceiling on validation → retry cycles.

AgentType = Literal["analytics", "generic"]

# Key injected into the sandbox namespace before the first analytics step.
# Analytics code captures Plotly figures by writing:
#   _plotly["Descriptive Title"] = fig.to_dict()
# The step_node harvests this dict into list[Plot] after each execution.
PLOTLY_NAMESPACE_KEY: str = "_plotly"


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class ExecutionResult(TypedDict):
    """
    Output of a single call to execute(code, namespace).

    stdout  — everything printed via print(); primary channel for captured results.
    stderr  — warnings / non-fatal tracebacks written to stderr.
    error   — exception message when execution raised; None on clean exit.
    """

    stdout: str
    stderr: str
    error: Optional[str]


class Plot(TypedDict):
    """
    A single Plotly figure captured from the analytics sandbox.

    Produced by the analytics step_node, which harvests PLOTLY_NAMESPACE_KEY
    from the execution namespace after each execute() call.  Analytics code
    is expected to write:

        import plotly.express as px
        fig = px.bar(df, x="category", y="count", title="Revenue by Category")
        _plotly["Revenue by Category"] = fig.to_dict()

    Do not call fig.show() — it blocks or errors in the sandbox.
    """

    title: str
    figure_json: dict[str, Any]


# ---------------------------------------------------------------------------
# Structured LLM outputs  (Pydantic — .model_validate() on every parse)
# ---------------------------------------------------------------------------


class OrchestrationDecision(BaseModel):
    """
    Output of the orchestrate_node.

    The LLM reads the user question and CSV paths and decides whether the
    question requires data analytics or a generic web-search answer.
    """

    agent_type: AgentType = Field(
        description=(
            "Route to 'analytics' for questions that require inspecting or "
            "computing over the provided CSV datasets. "
            "Route to 'generic' for factual, conceptual, or domain-knowledge "
            "questions that do not require the data."
        )
    )
    reasoning: str = Field(
        description="One- or two-sentence justification for the routing decision."
    )


class AnalyticsAction(BaseModel):
    """
    Output of the analytics subgraph's step_node LLM call.

    The LLM receives the question, the full prior step history, and any
    retry feedback, then produces reasoning and code for the next step.

    Sandbox conventions (enforced in the system prompt)
    ---------------------------------------------------
    - All imports must be at the top of the code block.
    - The namespace is persistent: variables and imports from earlier steps
      are available without redefinition.
    - Use print() for all scalar and tabular results you want captured.
    - Capture Plotly figures via:  _plotly["Title"] = fig.to_dict()
    - Do not call fig.show().
    """

    reasoning: str = Field(
        description=(
            "Explicit step-by-step reasoning about what to explore or compute "
            "next, informed by the question and all prior execution results. "
            "Stored in state and visible to every future step."
        )
    )
    code: str = Field(
        description=(
            "Python code to execute in the next sandbox step. "
            "Capture scalar/tabular outputs via print(). "
            "Capture Plotly figures via _plotly['Descriptive Title'] = fig.to_dict()."
        )
    )
    is_final_step: bool = Field(
        description=(
            "True when this code block completes the analysis and a final answer "
            "can be written immediately after execution. "
            "False when further exploration or computation will be needed."
        )
    )


class AnalyticsFinalAnswer(BaseModel):
    """
    Output of the analytics subgraph's answer_node.

    After the last step_node execution, the answer_node calls the LLM with the
    full step history (reasoning, code, and execution results for every iteration)
    and synthesises a conclusive natural-language answer.
    """

    final_answer: str = Field(
        description=(
            "Thorough, natural-language answer to the user's question grounded "
            "in the analysis results. Reference specific numbers, trends, or "
            "findings from stdout. Do not include raw code or DataFrames."
        )
    )


class PlanToExecute(BaseModel):
    """
    Output of the plan-execute subgraph's plan_node.

    The plan_node receives only the user question and CSV paths — it does NOT
    see the analytics agent's code or results.  It independently determines
    what computations are needed to answer the question, expressed entirely as
    typed Steps from planner_schemas.

    Execution is deterministic: the execute_node dispatches each Step in order
    using the same Step dispatcher as the original planner agent.  No free-form
    code is used.
    """

    reasoning: str = Field(
        description=(
            "Step-by-step reasoning about what needs to be computed to answer "
            "the question, and why these specific Steps are sufficient. "
            "Written before committing to the step sequence."
        )
    )
    steps: list[Step] = Field(
        min_length=1,
        description=(
            "Ordered sequence of deterministic Steps that compute an independent "
            "answer to the user's question. "
            "Each Step is dispatched by the execute_node; no free-form code is used. "
            "Use snapshot/restore to branch and rejoin the DataFrame as needed."
        ),
    )
    description: str = Field(
        description=(
            "Plain-language description of what the execution result is expected "
            "to represent, written before execution. "
            "The answer_node uses this to correctly interpret the trace output "
            "when synthesising final_answer."
        )
    )


class PlanExecuteFinalAnswer(BaseModel):
    """
    Output of the plan-execute subgraph's answer_node.

    After the execute_node runs all Steps and records the trace, the answer_node
    calls the LLM with the question, the plan's description, and the
    execution_result string to synthesise a concise natural-language answer.

    Intentionally parallel in structure to AnalyticsFinalAnswer so the
    validation_node can compare both final_answer fields directly.
    """

    final_answer: str = Field(
        description=(
            "Natural-language answer to the user's question derived solely from "
            "the deterministic Step execution results. "
            "Reference specific numbers or findings from execution_result. "
            "Do not include Step definitions or raw trace data."
        )
    )


class GenericResponse(BaseModel):
    """
    Output of the generic subgraph's respond_node.

    The respond_node calls the LLM with the user question and the raw DDG
    search text, then produces a markdown-formatted answer with citations.
    """

    response: str = Field(
        description=(
            "Markdown-formatted answer to the user's question. "
            "Cite sources inline as [Source Name](URL). "
            "Ground every claim in the search results; do not fabricate information."
        )
    )


class ValidationDecision(BaseModel):
    """
    Output of the validation_node.

    The validation_node receives both AnalyticsResult and PlanExecuteResult,
    compares their final_answer fields and supporting evidence, and issues
    one of five verdicts.

    Verdict semantics
    -----------------
    approved              — Analysis is correct and complete; proceed to output.
    needs_more_analysis   — Core results are consistent but something was missed
                            or under-explored; the analytics loop should continue.
    inconsistent_results  — Validator computed materially different values;
                            analytics results cannot be trusted.
    missing_analysis      — A required piece of analysis was not attempted.
    retry                 — Results are too unreliable; restart analytics from
                            scratch with the provided feedback.
    """

    verdict: Literal[
        "approved",
        "needs_more_analysis",
        "inconsistent_results",
        "missing_analysis",
        "retry",
    ] = Field(description="Outcome of the validation check.")
    reasoning: str = Field(
        description=(
            "Evidence-based reasoning comparing the analytics agent's findings "
            "against the plan-execute agent's independent computations. "
            "Cite specific numbers, column names, or chart observations from both."
        )
    )
    feedback: Optional[str] = Field(
        default=None,
        description=(
            "Actionable instructions for the analytics agent when verdict is not "
            "'approved'. Specify exactly what to fix, add, recompute, or redo. "
            "Must be None when verdict is 'approved'."
        ),
    )


# ---------------------------------------------------------------------------
# Result bundles  (TypedDict — plain data, no validation overhead)
# ---------------------------------------------------------------------------


class AnalyticsStep(TypedDict):
    """
    Complete record of one analytics loop iteration.

    Accumulated in GraphState.steps via an operator.add reducer so the full
    execution history is visible to the LLM at every subsequent step and to
    the answer_node when it synthesises the final answer.
    """

    step_index: int  # 0-based iteration counter.
    reasoning: str  # LLM reasoning prior to code generation.
    code: str  # Executed Python code.
    execution: ExecutionResult  # stdout / stderr / error from the sandbox.
    plots_captured: list[str]  # Titles of plots written to _plotly this step.


class AnalyticsResult(TypedDict):
    """
    Consolidated output of the analytics subgraph, written by answer_node.
    Passed to both the plan-execute subgraph (as context for the question)
    and the validation_node (for comparison against PlanExecuteResult).
    """

    final_answer: str  # Synthesised natural-language answer.
    plots: list[Plot]  # All plots captured across all iterations.
    steps: list[AnalyticsStep]  # Full iteration history.


class PlanExecuteResult(TypedDict):
    """
    Complete output of the plan-execute subgraph.

    Produced across three nodes:
      plan_node     — LLM generates PlanToExecute (question + CSV paths only;
                      analytics code and results are not visible).
      execute_node  — Step dispatcher walks the plan; populates trace and
                      execution_result.
      answer_node   — LLM synthesises final_answer from the question, the
                      plan's description, and execution_result.

    Passed wholesale to the validation_node for comparison against AnalyticsResult.
    """

    plan: PlanToExecute  # The generated Step-based plan.
    trace: list[TraceEntry]  # Per-step execution record from the Step dispatcher.
    execution_result: str  # Concatenated string output captured across all Steps.
    final_answer: str  # LLM-synthesised answer based on execution_result.


class ValidationResult(TypedDict):
    """
    Output of the validation_node.

    The validation_node reads analytics_result and plan_execute_result from
    state, compares them, and writes this bundle back to state.
    """

    decision: ValidationDecision  # Verdict + reasoning + optional feedback.


class SearchResult(TypedDict):
    """
    Output of the generic subgraph's search_node.
    Raw text returned by a single call to call_ddg(query).
    """

    query: str
    raw_text: str


class GenericResult(TypedDict):
    """
    Complete output of the generic subgraph.

    Assembled across two nodes:
      search_node  — calls call_ddg and writes search_result.
      respond_node — calls the LLM with the search text and writes response.
    """

    search_result: SearchResult  # Raw DDG output used to compose the response.
    response: str  # Markdown-formatted answer with inline citations.


# ---------------------------------------------------------------------------
# Single graph state
# ---------------------------------------------------------------------------


class GraphState(TypedDict):
    """
    Unified state shared across the main graph and all subgraphs.

    Using one flat state eliminates subgraph-boundary mapping boilerplate and
    keeps every field visible for debugging at any point in the execution trace.

    Reducer fields
    --------------
    steps  — Annotated[list[AnalyticsStep], operator.add]
             Each analytics loop iteration appends its AnalyticsStep; the list
             is never overwritten.  The full history is forwarded to the LLM on
             every step and to the answer_node for final synthesis.

    plots  — Annotated[list[Plot], operator.add]
             All Plotly figures accumulate across iterations.  The step_node
             harvests the _plotly sandbox dict after each execute() call and
             appends new Plot entries here.

    Namespace note
    --------------
    `namespace` holds the live Python globals dict shared across execute() calls.
    It is intentionally excluded from LangGraph checkpoints because it may contain
    non-serialisable objects (DataFrames, open file handles, etc.).
    On a cold resume the analytics init_node reinitialises it to {"_plotly": {}}.

    Field lifecycle  (see module docstring for the full node-by-node table)
    ───────────────────────────────────────────────────────────────────────
    [input]             : question, csv_paths
    orchestrate_node    : agent_type
    analytics init      : namespace, current_step_index, is_complete
    analytics step loop : steps (+append), plots (+append), namespace,
                          current_step_index, is_complete
    analytics answer    : analytics_result
    plan_execute        : plan_execute_result (across three nodes)
    validation_node     : validation_result
    retry edge          : retry_count (+1), validation_feedback,
                          analytics_result (cleared), plan_execute_result (cleared)
    generic subgraph    : generic_result (across two nodes)
    finalize_node       : answer, final_plots
    """

    # ── Inputs ───────────────────────────────────────────────────────────────
    question: str
    csv_paths: list[str]

    # ── Orchestration ─────────────────────────────────────────────────────────
    agent_type: Optional[AgentType]

    # ── Analytics loop ────────────────────────────────────────────────────────
    # Append-only execution memory; reducers prevent overwrite across iterations.
    steps: Annotated[list[AnalyticsStep], operator.add]
    plots: Annotated[list[Plot], operator.add]

    # Live sandbox environment; pre-seeded with {"_plotly": {}} by init_node.
    namespace: dict[str, Any]

    current_step_index: int  # Incremented by step_node each iteration.
    is_complete: bool  # Flipped to True by step_node when is_final_step.

    # ── Analytics output ──────────────────────────────────────────────────────
    analytics_result: Optional[AnalyticsResult]

    # ── Plan-execute output ───────────────────────────────────────────────────
    # Filled progressively: plan_node sets plan, execute_node sets trace and
    # execution_result, answer_node sets final_answer.
    plan_execute_result: Optional[PlanExecuteResult]

    # ── Validation output ─────────────────────────────────────────────────────
    validation_result: Optional[ValidationResult]

    # ── Retry control ─────────────────────────────────────────────────────────
    retry_count: int  # Checked against MAX_RETRY_CYCLES before each retry.
    validation_feedback: Optional[str]  # Injected by the retry edge; None on first run.

    # ── Generic output ────────────────────────────────────────────────────────
    generic_result: Optional[GenericResult]

    # ── Final output ──────────────────────────────────────────────────────────
    answer: Optional[str]
    final_plots: Optional[
        list[Plot]
    ]  # Distinct name avoids collision with the accumulator.


# ---------------------------------------------------------------------------
# Input / output schemas  (thin wrappers for graph.invoke() / graph.stream())
# ---------------------------------------------------------------------------


class GraphInput(TypedDict):
    """
    Supplied to graph.invoke() or graph.stream().

    Example:
        graph.invoke(GraphInput(
            question="Which product category had the highest average revenue?",
            csv_paths=["datasets/sales_2024.csv", "datasets/products.csv"],
        ))
    """

    question: str
    csv_paths: list[str]


class GraphOutput(TypedDict):
    """
    Returned by the compiled graph after the finalize_node runs.
    final_plots is an empty list for generic-domain answers.
    """

    answer: str
    final_plots: list[Plot]
