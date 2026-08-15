import copy
import numpy as np

from .base import BaseAggregator
from fedaegis.trust.score import TrustNormalizer


class DynamicCostAwareAggregator(
    BaseAggregator
):

    def aggregate(
        self,
        reports
    ):

        if not reports:
            raise ValueError(
                "No client reports received."
            )

        scores = [
            report.reliability
            for report in reports
        ]

        weights = TrustNormalizer.normalize(
            scores
        )

        model = copy.deepcopy(
            reports[0].parameters
        )

        for key in model:

            model[key] = np.sum(
                [
                    weights[index]
                    * reports[index].parameters[key]
                    for index in range(
                        len(reports)
                    )
                ],
                axis=0
            )

        return model
