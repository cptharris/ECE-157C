from typing import Any, Dict
import pandas as pd
from utils import describe_dataframe_like


def describe_dataset_node(state: Dict[str, Any]) -> Dict[str, Any]:
    if state["dataset_desc"] is not None:
        return state

    path = f"datasets/{state['metadata']['dataset_name']}"
    df = pd.read_csv(path)

    state["dataset_desc"] = describe_dataframe_like(df)

    return state
