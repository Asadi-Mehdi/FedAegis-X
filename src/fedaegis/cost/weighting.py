import numpy as np


class CostWeightCalculator:

    @staticmethod
    def sample_weights(
            y,
            cost_matrix):

        weights = np.ones(len(y), dtype=float)

        positive = np.max(y)

        weights[y == positive] = cost_matrix.false_negative

        weights[y != positive] = cost_matrix.false_positive

        return weights
