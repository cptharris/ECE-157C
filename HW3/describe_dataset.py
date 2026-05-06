import pandas as pd
import json
from schemas import AgentState


def describe_dataset_node(state: AgentState) -> AgentState:
    if state.get("dataset_description") is not None:
        return state

    df = pd.read_csv("dataset.csv")

    columns = []

    for col, dtype in df.dtypes.items():
        columns.append(f"{col!r} ({dtype})")

    rows, cols = df.shape

    state["dataset_description"] = {
        "shape": f"{rows} rows by {cols} columns",  
        "columns": ", ".join(columns),
        "sample": df.head(2).to_dict(orient="records"),
    }

    # state["dataset_description"] = json.dumps(
    #     {
    #         "row_count": len(df),
    #         "columns": df.dtypes.astype(str).to_dict(),
    #         "sample": df.head(2).to_dict(orient="records"),
    #     },
    #     indent=None,
    # )[:3000]

    return state
