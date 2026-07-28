from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class RoundHistory:

    round: int

    accuracy: float

    precision: float

    recall: float

    f1: float


class History:

    def __init__(self):

        self.records = []

    def add(self, metrics, round_number):

        self.records.append(

            RoundHistory(

                round=round_number,

                accuracy=metrics["accuracy"],

                precision=metrics["precision"],

                recall=metrics["recall"],

                f1=metrics["f1"]

            )

        )

    def save_json(self, filename):

        Path(filename).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(filename, "w") as f:

            json.dump(

                [asdict(r) for r in self.records],

                f,

                indent=4

            )
