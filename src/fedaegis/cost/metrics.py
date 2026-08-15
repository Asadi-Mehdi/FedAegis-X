import numpy as np
from sklearn.metrics import confusion_matrix


class CostMetrics:

    @staticmethod
    def total_cost(
            y_true,
            y_pred,
            matrix):

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred
        ).ravel()

        return (

            fp * matrix.false_positive +

            fn * matrix.false_negative +

            tp * matrix.true_positive +

            tn * matrix.true_negative

        )

    @staticmethod
    def false_negative_rate(
            y_true,
            y_pred):

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred
        ).ravel()

        if tp + fn == 0:

            return 0.0

        return fn / (tp + fn)
