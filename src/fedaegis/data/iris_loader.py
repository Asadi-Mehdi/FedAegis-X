from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class IrisLoader:

    def __init__(self,
                 test_size=0.2,
                 random_state=42):

        self.test_size = test_size
        self.random_state = random_state

    def load(self):

        dataset = load_iris()

        X_train, X_test, y_train, y_test = train_test_split(
            dataset.data,
            dataset.target,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=dataset.target,
        )

        return X_train, X_test, y_train, y_test
