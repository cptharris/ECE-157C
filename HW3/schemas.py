from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Step:
    op: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    op: str
    input_rows: int
    output_rows: int


@dataclass
class State:
    question: str
    steps: List[Step]
    trace: List[Trace]
    final_answer: Optional[str] = None
    dataset_description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def state_to_dict(state: State) -> Dict[str, Any]:
    return state.to_dict()
