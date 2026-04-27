import pandas as pd
from typing import Dict, Any


def execute_node(state: Dict[str, Any]) -> Dict[str, Any]:
    path = f"datasets/{state['dataset_name']}"
    df = pd.read_csv(path)

    local_vars = {"df": df, "pd": pd}

    try:
        exec(state["code"], {}, local_vars)
        result = local_vars.get("result", None)

        state["execution"] = {
            "result": result,
            "error": None
        }

    except Exception as e:
        state["execution"] = {
            "result": None,
            "error": str(e)
        }

    return state
