from pathlib import Path
import pandas as pd


class ResultWriter:

    def __init__(self, output_dir):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(

            parents=True,

            exist_ok=True

        )

    def save_metrics(self, history):

        rows = []

        for item in history.records:

            rows.append(

                {

                    "round": item.round,

                    "accuracy": item.accuracy,

                    "precision": item.precision,

                    "recall": item.recall,

                    "f1": item.f1

                }

            )

        pd.DataFrame(rows).to_csv(

            self.output_dir / "metrics.csv",

            index=False

        )
