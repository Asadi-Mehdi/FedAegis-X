import numpy as np


class ReliabilityCalculator:

    @staticmethod
    def calculate(
            fnr,
            samples,
            class_balance):

        fnr_score = 1.0 - fnr

        sample_score = np.log1p(samples)

        score = (

            0.45 * fnr_score +

            0.35 * sample_score +

            0.20 * class_balance

        )

        return float(score)
