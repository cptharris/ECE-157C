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

CUSTOM_QUESTIONS = [
    # Simple
    "What is the average popularity score across all tracks in the dataset?",
    "Which track genre has the highest average danceability?",
    "What percentage of tracks in the dataset are marked as explicit?",
    # Intermediate
    "How does average energy level differ across the top 10 most common genres?",
    "Is there a relationship between tempo and danceability? Bin tempo into slow (<90 BPM), medium (90–130 BPM), and fast (>130 BPM) and compare average danceability across bins.",
    "Which genres have the highest average valence (musical positivity), and how does that compare to their average popularity?",
    # Advanced
    "Identify the top 10 most popular tracks and analyze what audio features (energy, danceability, valence, tempo) they share. Do popular tracks cluster around specific feature ranges?",
    "Find genres where high acousticness (>0.7) coexists with high popularity (>60). What characteristics do these genres share, and what might explain the combination?",
    "Find genre pairs where average tempo differs by less than 5 BPM and average energy differs by less than 0.05, but average popularity differs by more than 15 points. For each pair, identify which audio features (danceability, acousticness, speechiness, instrumentalness, valence) show the largest differences.",
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all(output_csv: str = "results.csv") -> None:
    """Run all questions and write results to results.csv."""
    rows = []

    all_tasks = (
        [(HOUSING_CSV, q) for q in HOUSING_QUESTIONS] +
        [(CUSTOM_CSV, q) for q in CUSTOM_QUESTIONS]
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
