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
            "run_count": 0,
            "max_runs": 2,
        }
    )


# ---------------------------------------------------------------------------
# Question lists
# ---------------------------------------------------------------------------

DATASET_NAME = "dataset.csv"

CUSTOM_QUESTIONS = [
    # ============
    #    BASIC
    # ============
    "What is the average nightly price for entire home/apt listings in Manhattan?",
    "Rank boroughs by which has the most listings that are available year-round.",
    "How many listings in Brooklyn are priced under $100 per night?",
    "What are the top 5 hosts by total number of listings across all boroughs?",
    # "What is the average number of reviews for listings that require a minimum stay of exactly 1 night?",
    # "What is the most expensive room type on average in Queens?",
    # "Which neighbourhoods in Staten Island have fewer than 10 listings?",
    "Describe the availability for listings that have at least 10 reviews.",
    # ============
    # INTERMEDIATE
    # ============
    "Which borough has the highest average estimated annual revenue potential, defined as price times availability_365?",
    "Among listings with at least 10 reviews, describe the average minimum total cost (price times minimum nights) by room type.",
    "What percentage of listings in each borough are entire home/apt?",
    "For listings requiring a minimum stay of at least 7 nights, what is the average price and average availability_365 by room type?",
    # "What is the average price-per-minimum-night by room type, for listings priced under $500?",
    "For listings with above-average monthly reviews, what is the average price by borough?",
    # "Which neighbourhood has the highest ratio of average price to average number of reviews, among those with at least 20 listings?",
    # "Among hosts with more than 3 listings, what is the average price of their listings by borough?",
    # ============
    #   ADVANCED
    # ============
    "Among entire home/apt listings in Manhattan available more than 200 days per year and priced under $300, what are the top 3 neighbourhoods by average number of reviews?",
    'Which neighbourhood has the highest concentration of "high-value" listings (where price > average price and availability > average availability and number of reviews > average number of reviews), among neighbourhoods with at least 30 total listings?',
    # "For each borough, what is the top neighbourhood by average estimated annual revenue potential (price times availability_365), among listings with at least 50 reviews?",
    # "Among private room listings that are available at least 100 days per year and have a reviews_per_month above 0.5, which 5 neighbourhoods have the lowest average price?",
    "What is the average price gap between shared room and private room listings within each borough, and which borough has the largest gap?",
    "Among listings with at least 20 reviews and a minimum stay of 1—3 nights, which listing in which borough has the lowest total cost (price times minimum nights)? Disregard listings with a price of $0.",
    "Which hosts operating in all 5 boroughs have the highest average listing price, among hosts with at least 20 total listings?",
    # "For entire home/apt listings priced between $75 and $250, which borough shows the strongest correlation proxy between availability and review count — i.e., highest average (availability_365 times reviews_per_month) — and what is that value?",
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all(output_csv: str = "results.csv") -> None:
    """Run all questions and write results to results.csv."""
    rows = []

    all_tasks = [(DATASET_NAME, q) for q in CUSTOM_QUESTIONS]
    fieldnames = ["question", "final_answer", "plan", "trace"]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for csv_path, question in all_tasks:
            dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
            print(f"\n\n{'='*45} BEGIN TASK {'='*45}\n")
            print(f"{'='*5}  Dataset  : {dataset_name}")
            print(f"{'='*5}  Question : {question}\n\n")

            result = run_agent(question, csv_path)

            print(f"\n\n{'='*45} END TASK {'='*45}\n")

            writer.writerow(
                {
                    "question": question,
                    "final_answer": result["final_answer"],
                    "plan": json.dumps(result["plan"], indent=2, ensure_ascii=False),
                    "trace": json.dumps(result["trace"], indent=2, ensure_ascii=False),
                }
            )

            print(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\nTASK COMPLETE! {len(CUSTOM_QUESTIONS)} questions answered.")


if __name__ == "__main__":
    run_all()
