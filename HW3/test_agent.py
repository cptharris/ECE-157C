"""
test_agent.py
-------------
Provides the required run_agent() function and runs all 18 questions
(9 fixed housing.csv + 9 custom dataset), saving results to results.csv.

Usage:
    python test_agent.py
"""

import csv
import os
import json
from agent import agent
from schemas import AgentState, TraceEntry, Plan, Step


# ---------------------------------------------------------------------------
# Required function (signature fixed by assignment)
# ---------------------------------------------------------------------------


def run_agent(question: str, csv_path: str) -> dict:
    return agent.invoke(
        {
            "question": question,
            "csv_path": csv_path,
            "dataset_description": None,
            "plan": Plan(
                reasoning="",
                steps=[
                    {
                        "op": "filter_rows",
                        "conditions": [
                            {
                                "column": "A",
                                "operator": "==",
                                "value": 1
                            }
                        ]
                    }
                ]
            ),
            "trace": [],
            "final_answer": None,
            "retry_count": 0,
            "max_retries": 2,
        }
    )


# ---------------------------------------------------------------------------
# Question lists
# ---------------------------------------------------------------------------


CUSTOM_QUESTIONS = [
    "What's goin' on?",
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all(output_csv: str = "results.csv") -> None:
    """Run all questions and write results to results.csv."""
    rows = []

    all_tasks = [("dataset.csv", q) for q in CUSTOM_QUESTIONS]

    for csv_path, question in all_tasks:
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"\n\n{'='*45} BEGIN TASK {'='*45}\n\n")
        print(f"Dataset : {dataset_name}")
        print(f"Question: {question}")

        result = run_agent(question, csv_path)

        print(json.dumps(result, indent=4))

        rows.append(
            {
                "dataset_name": dataset_name,
                "question": question,
                "plan": result["plan"],
                "trace": result["trace"],
                "final_answer": result["final_answer"],
            }
        )

    # Write results.csv
    fieldnames = ["dataset_name", "question", "plan", "trace", "final_answer"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {output_csv}")


if __name__ == "__main__":
    run_all()
