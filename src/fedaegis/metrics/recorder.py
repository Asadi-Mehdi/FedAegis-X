from pathlib import Path
import pandas as pd


class MetricsRecorder:

    def __init__(
            self,
            output_dir):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.records = []

    def add(
            self,
            round_number,
            metrics):

        row = {

            "round": round_number,

            **metrics

        }

        self.records.append(row)

    def save(self):

        df = pd.DataFrame(self.records)

        df.to_csv(

            self.output_dir / "metrics.csv",

            index=False

        )
