import json
from pathlib import Path


class EvaluationReport:

    def __init__(self):

        self.results = {}

    def add(
            self,
            key,
            value):

        self.results[key] = value

    def save(
            self,
            filename):

        Path(filename).parent.mkdir(

            parents=True,

            exist_ok=True

        )

        with open(

                filename,

                "w"

        ) as f:

            json.dump(

                self.results,

                f,

                indent=4

            )
