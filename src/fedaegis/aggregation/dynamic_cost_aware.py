import copy
import numpy as np

from .base import BaseAggregator

from fedaegis.trust.score import TrustNormalizer


class DynamicCostAwareAggregator(BaseAggregator):

    def aggregate(
            self,
            reports):

        weights = TrustNormalizer.normalize(

            [

                r.reliability

                for r in reports

            ]

        )

        model = copy.deepcopy(

            reports[0].parameters

        )

        for key in model.keys():

            model[key] = np.sum(

                [

                    weights[i] *

                    reports[i].parameters[key]

                    for i in range(

                        len(reports)

                    )

                ],

                axis=0

            )

        return model
