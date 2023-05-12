
from dataclasses import dataclass
from typing import List

from count_model.evaluate.evaluation import Evaluation


@dataclass
class ScoreSummary:
    dynamic: List[Evaluation]
    dynamic_ndcg: float
    # static:List[Evaluation]
    # static_ndcg:float