import pandas as pd
from .schemas import Artifact


def execute_node(state: Artifact) -> Artifact:
    path = f"datasets/{state.dataset_name}"
    df = pd.read_csv(path)

    local_vars = {"df": df, "pd": pd}

    try:
        exec(state.code["snippet"], {}, local_vars)
        result = local_vars.get("result", None)

        state.execution = {
            "stdout": "",
            "result": result,
            "error": None
        }

    except Exception as e:
        state.execution = {
            "stdout": "",
            "result": None,
            "error": str(e)
        }

    return state
