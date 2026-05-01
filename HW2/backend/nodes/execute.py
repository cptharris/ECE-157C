import pandas as pd
import traceback
from typing import Dict, Any
from utils import make_json_safe, describe_dataframe_like


def execute_node(state: Dict[str, Any]) -> Dict[str, Any]:
    df = pd.read_csv(f"datasets/{state['metadata']['dataset_name']}")

    local_vars = {"df": df, "pd": pd}

    try:
        exec(state["execution"]["data_code"], local_vars)

        data = make_json_safe(local_vars.get("data", None))

        state["execution"]["data"] = data
        state["execution"]["error"] = None
        state["execution"]["data_desc"] = describe_dataframe_like(
            pd.DataFrame(data) if isinstance(data, list) else data
        )
    except Exception as e:
        state["execution"]["data"] = None
        state["execution"]["error"] = str(e)
        state["execution"]["data_desc"] = ""
        traceback.print_exception(e)

    return state
