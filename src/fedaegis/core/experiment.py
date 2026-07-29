from pathlib import Path
import json


class Experiment:

    def __init__(

            self,

            cfg):

        self.cfg = cfg

    def save(self):

        Path("outputs").mkdir(

            exist_ok=True

        )

        with open(

                "outputs/experiment.json",

                "w"

        ) as f:

            json.dump(

                self.cfg,

                f,

                indent=4

            )
