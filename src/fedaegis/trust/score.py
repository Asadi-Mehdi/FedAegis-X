import numpy as np


class TrustNormalizer:

    @staticmethod
    def normalize(scores):

        scores = np.array(scores, dtype=float)

        scores = scores / scores.sum()

        return scores
