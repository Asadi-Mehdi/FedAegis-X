from sklearn.linear_model import SGDClassifier

from .base import BaseModel


class SGDModel(BaseModel):

    def __init__(self):

        self.model = SGDClassifier(

            loss="log_loss",

            random_state=42

        )

        self.initialized = False

    def fit(
            self,
            X,
            y):

        if not self.initialized:

            self.model.partial_fit(

                X,

                y,

                classes=list(sorted(set(y)))

            )

            self.initialized = True

        else:

            self.model.partial_fit(

                X,

                y

            )

    def predict(
            self,
            X):

        return self.model.predict(X)

    def get_parameters(self):

        return {

            "coef": self.model.coef_.copy(),

            "intercept": self.model.intercept_.copy()

        }

    def set_parameters(
            self,
            params):

        self.model.coef_ = params["coef"]

        self.model.intercept_ = params["intercept"]

        self.initialized = True
