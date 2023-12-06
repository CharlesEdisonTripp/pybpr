from dataclasses import dataclass


@dataclass
class Evaluation:
    name: str
    score: float
    positives: float
    negatives: float