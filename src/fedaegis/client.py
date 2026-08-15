from sklearn.metrics import confusion_matrix

from fedaegis.core.client_report import ClientReport
from fedaegis.trust.reliability import (
    ReliabilityCalculator
)


class Client:

    def __init__(
        self,
        client_id,
        model,
        X,
        y
    ):

        self.client_id = client_id
        self.model = model
        self.X = X
        self.y = y

    def train(self):

        self.model.fit(
            self.X,
            self.y
        )

        predictions = self.model.predict(
            self.X
        )

        fnr = self._calculate_fnr(
            self.y,
            predictions
        )

        class_balance = (
            ReliabilityCalculator
            .class_balance_score(self.y)
        )

        reliability = (
            ReliabilityCalculator.calculate(
                fnr=fnr,
                samples=len(self.X),
                class_balance=class_balance
            )
        )

        return ClientReport(

            client_id=self.client_id,

            samples=len(self.X),

            fnr=fnr,

            class_balance=class_balance,

            reliability=reliability,

            parameters=self.model.get_parameters()
        )

    @staticmethod
    def _calculate_fnr(
        y_true,
        y_pred
    ):

        matrix = confusion_matrix(
            y_true,
            y_pred
        )

        if matrix.shape == (2, 2):

            tn, fp, fn, tp = matrix.ravel()

            denominator = tp + fn

            if denominator == 0:
                return 0.0

            return float(
                fn / denominator
            )

        # Multiclass FNR:
        # macro-average one-vs-rest FNR
        fnr_values = []

        labels = range(
            matrix.shape[0]
        )

        for label in labels:

            tp = matrix[label, label]

            fn = (
                matrix[label, :].sum()
                - tp
            )

            denominator = tp + fn

            if denominator > 0:

                fnr_values.append(
                    fn / denominator
                )

        if not fnr_values:
            return 0.0

        return float(
            sum(fnr_values)
            / len(fnr_values)
        )

    def update_global_model(
        self,
        global_parameters
    ):

        self.model.set_parameters(
            global_parameters
        )

    def predict(self, X):

        return self.model.predict(X)
