import numpy as np


class ReliabilityCalculator:

    @staticmethod
    def class_balance_score(y):

        values, counts = np.unique(
            y,
            return_counts=True
        )

        if len(values) <= 1:
            return 1.0

        proportions = (
            counts / counts.sum()
        )

        entropy = -np.sum(
            proportions * np.log(
                proportions + 1e-12
            )
        )

        max_entropy = np.log(
            len(values)
        )

        if max_entropy == 0:
            return 1.0

        return float(
            entropy / max_entropy
        )

    @staticmethod
    def calculate(
        fnr,
        samples,
        class_balance,
        alpha=0.45,
        beta=0.35,
        gamma=0.20
    ):

        fnr_score = max(
            0.0,
            1.0 - float(fnr)
        )

        sample_score = (
            np.log1p(samples)
            / np.log1p(100000)
        )

        sample_score = min(
            1.0,
            sample_score
        )

        score = (
            alpha * fnr_score
            + beta * sample_score
            + gamma * class_balance
        )

        return float(
            max(score, 1e-12)
        )
