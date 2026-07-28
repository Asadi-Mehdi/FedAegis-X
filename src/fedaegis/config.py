from pathlib import Path
import yaml


class Config:

    def __init__(self, path="configs/default.yaml"):

        self.path = Path(path)

        with open(self.path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def __getitem__(self, item):
        return self.cfg[item]
