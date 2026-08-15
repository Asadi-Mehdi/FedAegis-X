from dataclasses import dataclass


@dataclass
class CostMatrix:

    false_positive: float

    false_negative: float

    true_positive: float = 0.0

    true_negative: float = 0.0
