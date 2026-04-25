from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class Artifact(BaseModel):
    run_id: str
    step: str

    input_question: str
    dataset_name: str

    context: Dict[str, Any] = {}

    code: Dict[str, Any] = {}

    execution: Dict[str, Any] = {}

    analysis: Dict[str, Any] = {}

    visualization: Dict[str, Any] = {
        "should_visualize": False,
        "reason": "",
        "plotly_code": ""
    }

    final_answer: Optional[str] = None
