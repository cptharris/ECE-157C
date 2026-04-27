from .call_llm import call_llm
from typing import Dict, Any
import json


def summarize_node(state: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    path = f"datasets/{state["dataset_name"]}"
    df = pd.read_csv(path)

    state["dataset_context"] = {
        "column_types": df.dtypes.astype(str).to_dict(),
        "row_count": len(df),
        # "sample_rows": df.head(3).to_dict(orient="records")
    }

    return state
