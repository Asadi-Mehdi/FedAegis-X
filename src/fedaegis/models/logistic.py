import numpy as np

from sklearn.linear_model import LogisticRegression

from .base import BaseModel


class LogisticModel(BaseModel):

    def __init__(self):

        self.model = LogisticRegression(
            max_iter=500,
            random_state=42
        )

    def fit(self, X, y):

        self.model.fit(X, y)

    def predict(self, X):

        return self.model.predict(X)

    def get_parameters(self):

        return {
            "coef": self.model.coef_.copy(),
            "intercept": self.model.intercept_.copy(),
        }

    def set_parameters(self, params):

        self.model.coef_ = np.array(params["coef"])

        self.model.intercept_ = np.array(params["intercept"])

        self.model.classes_ = np.arange(
            self.model.coef_.shape[0]
        )
