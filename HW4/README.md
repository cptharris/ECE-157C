# ECE 272C HW4: Analytics Agent

Overview
--------
HW4 implements a LangGraph-based analytics agent that can route questions to an analytics
pipeline (that reads and analyzes CSV datasets) or to a generic web-search path. The
architecture combines a deterministic plan-execute subgraph with a separate analytics
loop and a validation step, enabling both data-driven reasoning and robust cross-checks.

Key ideas:
- Route questions via the orchestrator to either analytics (with datasets) or generic
  (non-dataset) handling.
- Analytics path loads datasets, runs iterative analysis steps, and streams plots and
  results as the analysis progresses.
- Plan-execute path runs a deterministic plan against the data to produce a final output
  and then a validator compares analytics results with the plan results.

What you’ll find in this repository
- `analytics_agent.py` — analytics node implementations that drive the data-focused exploration (init, step, and final answer generation).
- `orchestrator.py` — routes the question to the right path.
- `dataset_node.py` — builds dataset schemas by profiling datasets in the `datasets/` directory and asking the LLM to describe relevant schema information.
- `graph.py` — the LangGraph state graph wiring that connects orchestrator, analytics, plan-execute, validation, and the finalization steps.
- `plan_planner.py`, `plan_schemas.py`, `plan_executor.py`, `plan_operators.py`,
  `plan_respond.py`, `validator.py` — the subgraph-based planner, execution, and validation components that enable the deterministic plan-and-execute flow.
- `server.py` — a FastAPI backend that serves the frontend and streams events for the UI via Server-Sent Events (SSE).
- `frontend.html` — a lightweight UI that lists available CSV datasets and lets you ask questions which are streamed in real time.
- `schemas.py` — all structural schemas used by the graph (GraphState, inputs, outputs, and typed dictionaries for analytics steps and plan executions).
- `datasets/` — directory for sample CSV files used in analytics experiments.

Prerequisites
- Python 3.10+ (recommended 3.11)
- Install dependencies (examples):
  - `pip install langgraph langchain langsmith pandas plotly pydantic`

Project layout notes
- The primary API surface is provided by `server.py`. Use:
  - `python server.py` to run the backend directly in development mode.
  - Or: `uvicorn server:app --reload --port 8000` from the directory to enable hot reloading.
- The frontend (`frontend.html`) consumes `/api/files` and `/api/stream` endpoints and is loaded by visiting http://localhost:8000/ when the server runs.
- Datasets must live in the `datasets/` directory. The UI queries `/api/files` to list available CSV files.

How it works (high level)
- Orchestrator (`orchestrator.py`) decides whether a question should be handled by analytics (data-driven) or by a generic search path.
- Dataset fan-out (`dataset_node.py`) profiles the chosen datasets and feeds information to the analytics subgraph.
- Analytics path (`analytics_agent.py`) runs an iterative loop of steps (init, step, answer) that produce plots and analytical output.
- Plan-execute subgraph (`plan_planner.py`, `plan_executor.py`, `plan_schemas.py`, `plan_operators.py`, `plan_respond.py`) computes a deterministic plan, executes it against the same data, and creates a final answer.
- Validation (`validator.py`) compares analytics results against the plan-execute results.
- Finalization (`finalize.py`) produces the final answer and any final plots for the user.

State and data models
- The core runtime state is implemented in `schemas.py` as GraphState. It contains:
  - inputs: `question`, `csv_paths`
  - `orchestration_decision`, `dataset_thoughts`
  - analysis state: `namespace`, `steps`, `plots`, `current_step_index`, `is_complete`, `analytics_result`
  - plan execution state: `plan_execute_result` (`plan`, `trace`, `execution_result`, `final_answer`)
  - `validation_result`, `retry_count`, `retry` `feedback`, `generic_result`, `answer`, `final_plots`

Usage example
- Place one or more CSV files into the `datasets/` directory. Then start the server, visit http://127.0.0.1:8000/ and select CSVs from the left panel, and type a question in the input box.
- The UI streams events (`node_start`, `node_end`, `token`, `step`, `step_result`, `plot`, `final_answer`) via SSE. You can view intermediate steps, plots, and final answers as they are produced.

Possible extension points
- You can customize datasets or add new operations by adjusting `plan_schemas.py` and `plan_operators.py` and then re-running the graph to test new capabilities.

Commands and quick checks
- List available CSV files:
  `curl -s http://127.0.0.1:8000/api/files`
- Start the server:
  `python server.py`
