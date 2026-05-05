import pandas as pd
import json
from schemas import AgentState


def describe_dataset_node(state: AgentState) -> AgentState:
    if state.get("dataset_description") is not None:
        return state

    df = pd.read_csv("dataset.csv")

    state["dataset_description"] = {
        "row_count": len(df),
        "columns": df.dtypes.astype(str).to_dict(),
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
