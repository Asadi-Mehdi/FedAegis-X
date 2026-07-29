from .iris_loader import IrisLoader
from .csv_loader import CSVLoader


class DatasetFactory:

    @staticmethod
    def create(cfg):

        name = cfg["dataset"]["name"]

        if name == "iris":

            return IrisLoader(

                test_size=cfg["dataset"]["test_size"],

                random_state=cfg["dataset"]["random_state"]

            )

        return CSVLoader(

            csv_path=cfg["dataset"]["path"],

            label_column=cfg["dataset"]["label"],

            test_size=cfg["dataset"]["test_size"],

            random_state=cfg["dataset"]["random_state"]

        )
