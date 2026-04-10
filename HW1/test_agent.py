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
from agent import app


# ---------------------------------------------------------------------------
# Required function (signature fixed by assignment)
# ---------------------------------------------------------------------------


def run_agent(question: str, csv_path: str) -> dict:
    """
    Run the CSV question-answering agent.

    Parameters
    ----------
    question : str
        Natural language question about the dataset.
    csv_path : str
        Path to the CSV file to analyse.

    Returns
    -------
    dict with keys:
        generated_code   : str
        execution_result : object
        execution_error  : str | None
        evaluation       : "PASS" or "FAIL"
        final_answer     : str
    """
    initial_state = {
        "question": question,
        "csv_path": csv_path,
        "retry_count": 0,
    }
    final_state = app.invoke(initial_state)

    return {
        "generated_code": final_state.get("generated_code", ""),
        "execution_result": final_state.get("execution_result", None),
        "execution_error": final_state.get("execution_error", None),
        "evaluation": final_state.get("evaluation", "FAIL"),
        "final_answer": final_state.get("final_answer", ""),
    }


# ---------------------------------------------------------------------------
# Question lists
# ---------------------------------------------------------------------------

HOUSING_CSV = "housing.csv"

HOUSING_QUESTIONS = [
    # Simple
    "What is the average median house value across the dataset?",
    "Which ocean proximity category has the highest average median house value?",
    "What are the minimum, maximum, and median values of median house value?",
    # Intermediate
    "How does median income vary across different ocean proximity categories?",
    "Which geographic areas (based on latitude and longitude ranges) have the highest average house prices?",
    "How does population density (defined as population per household) relate to median house value?",
    # Advanced
    "Identify the top 5 most expensive geographic areas and explain the key factors contributing to their high prices.",
    "Find coastal areas where house prices are relatively low despite proximity to the ocean. What factors might explain this?",
    "Identify areas with similar median income levels but significantly different median house values. What factors might explain these differences?",
]

CUSTOM_CSV = "custom_dataset.csv"

# TODO: replace these placeholders with your real custom questions once you
#       have chosen your dataset.  Keep the same 3/3/3 simple/inter/advanced
#       split required by the assignment.
CUSTOM_QUESTIONS = [
    # Simple
    "TODO: simple question 1",
    "TODO: simple question 2",
    "TODO: simple question 3",
    # Intermediate
    "TODO: intermediate question 1",
    "TODO: intermediate question 2",
    "TODO: intermediate question 3",
    # Advanced
    "TODO: advanced question 1",
    "TODO: advanced question 2",
    "TODO: advanced question 3",
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all(output_csv: str = "results.csv") -> None:
    """Run all questions and write results to results.csv."""
    rows = []

    all_tasks = (
        [(HOUSING_CSV, q) for q in HOUSING_QUESTIONS]
        # + [(CUSTOM_CSV, q) for q in CUSTOM_QUESTIONS]
    )

    for csv_path, question in all_tasks:
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"\n\n{'='*45} BEGIN TASK {'='*45}\n\n")
        print(f"Dataset : {dataset_name}")
        print(f"Question: {question}")

        result = run_agent(question, csv_path)

        print(f"\n\n{'='*23} Code {'='*23}\n\n")
        print(result["generated_code"])

        print(f"\n\n{'='*17} Execution Result {'='*17}\n\n")
        print(result["execution_result"])

        print(f"\n\n{'='*20} Evaluation {'='*20}\n\n")
        print(f"Eval    : {result['evaluation']}")
        print(f"Answer  : {result['final_answer']}")

        rows.append(
            {
                "dataset_name": dataset_name,
                "question": question,
                "generated_code": result["generated_code"],
                "final_answer": result["final_answer"],
            }
        )

    # Write results.csv
    fieldnames = ["dataset_name", "question", "generated_code", "final_answer"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {output_csv}")


if __name__ == "__main__":
    run_all()
