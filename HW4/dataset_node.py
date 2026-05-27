from langchain_core.runnables import RunnableConfig
from typing import Any

from schemas import GraphState, DatasetThoughts
from utilities import call_llm

SYSTEM = f"""\
Return ONLY this JSON-formatted response:
{{
  "reasoning": "{DatasetThoughts.model_json_schema()["properties"]["reasoning"]["description"]}",
  "schemas": "{DatasetThoughts.model_json_schema()["properties"]["schemas"]["description"]}"
}}
"""


def dataset_node(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    import pandas as pd
    import json

    dataset_schema = {}
    for csv_path in state["csv_paths"]:
        df = pd.read_csv("datasets/" + csv_path)
        dataset_schema[csv_path] = {
            "shape": df.shape,
            "columns": df.dtypes.apply(lambda x: x.name).to_dict(),
        }

    thoughts = call_llm(
        system_prompt=SYSTEM,
        user_prompt=f"""
Question:
{state["question"]}

Unfiltered Dataset Specifications:
{json.dumps(dataset_schema, indent=None)}
    """,
        who="dataset",
    )

    thoughts = DatasetThoughts.model_validate_json(thoughts)

    return {"dataset_thoughts": thoughts}
