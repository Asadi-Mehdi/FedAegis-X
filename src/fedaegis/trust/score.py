import numpy as np


class TrustNormalizer:

    @staticmethod
    def normalize(scores):

        values = np.asarray(
            scores,
            dtype=float
        )

        if values.size == 0:
            raise ValueError(
                "No reliability scores."
            )

        values = np.maximum(
            values,
            1e-12
        )

        total = values.sum()

        if total <= 0:
            return np.ones(
                len(values)
            ) / len(values)

        return values / total
