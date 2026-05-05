from typing import Any, Dict
import pandas as pd
import json


def describe_dataset_node(state: State) -> State:
    if state.get("dataset_description") is not None:
        return state

    df = pd.read_csv("dataset.csv")

    state["dataset_description"] = json.dumps(
        {
            "row_count": len(df),
            "columns": df.dtypes.astype(str).to_dict(),
            "sample": df.head(2).to_dict(orient="records"),
        },
        indent=None,
    )[:3000]

    return state
