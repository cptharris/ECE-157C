import csv
import json
import hashlib
import os

INPUT_FILE = "results.csv"
OUTPUT_DIR = "report-data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_hash(question: str) -> str:
    return hashlib.md5(question.encode()).hexdigest()


def write_markdown(hash_id: str, final_answer: str):
    path = os.path.join(OUTPUT_DIR, f"{hash_id}.md")
    with open(path, "w") as f:
        f.write(final_answer)


def write_plan_json(hash_id: str, plan_raw: str):
    path = os.path.join(OUTPUT_DIR, f"{hash_id}.json")
    plan = json.loads(plan_raw)
    reformattedSteps = []
    for step in plan["steps"]:
        reformattedSteps.append(json.dumps(step, indent=None).replace("\"", ""))
    
    plan["steps"] = reformattedSteps

    with open(path, "w") as f:
        json.dump(plan, f, indent=2)


def write_trace_tex(hash_id: str, trace_raw: str):
    path = os.path.join(OUTPUT_DIR, f"{hash_id}.tex")
    trace = json.loads(trace_raw)

    def fmt(val):
        return str(val).replace("_", r"\_")

    rows = []
    for entry in trace:
        step = entry["step_index"]
        op = "\\texttt{" + fmt(entry["op"]) + "}"
        in_shape = "$\\times$".join(str(x) for x in entry["input_shape"])
        out_shape = "$\\times$".join(str(x) for x in entry["output_shape"])
        rows.append(f"  {step} & {op} & {in_shape} & {out_shape} \\\\")

    body = "\n".join(rows)
    tex = (
        "\\begin{center}\n"
        "\\footnotesize"
        "\\begin{tabular}{|c|l|r|r|}\n"
        "\\hline\n"
        "Step & Operation & Input (R$\\times$C) & Output (R$\\times$C) \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{center}\n"
    )
    with open(path, "w") as f:
        f.write(tex)


index = {}

with open(INPUT_FILE, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        question = row["question"]
        hash_id = make_hash(question)
        index[question] = hash_id

        write_markdown(hash_id, row["final_answer"].replace("−", "-"))
        write_plan_json(hash_id, row["plan"])
        write_trace_tex(hash_id, row["trace"])

        print(f"[{hash_id}] {question[:60]}...")

with open(os.path.join(OUTPUT_DIR, "index.json"), "w") as f:
    json.dump(index, f, indent=2)

print(f"\nDone. {len(index)} row(s) processed. Output in '{OUTPUT_DIR}/'.")
