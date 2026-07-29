import pandas as pd
from sklearn.model_selection import train_test_split

from .base_loader import BaseDatasetLoader


class CSVLoader(BaseDatasetLoader):

    def __init__(
            self,
            csv_path,
            label_column,
            test_size=0.2,
            random_state=42):

        self.csv_path = csv_path
        self.label_column = label_column
        self.test_size = test_size
        self.random_state = random_state

    def load(self):

        df = pd.read_csv(self.csv_path)

        X = df.drop(
            columns=[self.label_column]
        ).values

        y = df[self.label_column].values

        return train_test_split(

            X,

            y,

            test_size=self.test_size,

            random_state=self.random_state,

            stratify=y

        )
