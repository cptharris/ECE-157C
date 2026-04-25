from .call_llm import call_llm
from .schemas import Artifact
import json


def summarize_node(state: Artifact) -> Artifact:
    import pandas as pd

    path = f"datasets/{state.dataset_name}"
    df = pd.read_csv(path)

    state.context = {
        "columns": list(df.columns),
        "column_types": df.dtypes.astype(str).to_dict(),
        "row_count": len(df),
        "sample_rows": df.head(5).to_dict(orient="records")
    }

    return state
