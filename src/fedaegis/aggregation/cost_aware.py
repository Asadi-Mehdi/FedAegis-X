import copy
import numpy as np

from .base import BaseAggregator


class CostAwareAggregator(BaseAggregator):

    def aggregate(
            self,
            client_updates):

        scores = np.array(

            [

                client["cost_score"]

                for client in client_updates

            ],

            dtype=float

        )

        scores = scores / scores.sum()

        model = copy.deepcopy(

            client_updates[0]["params"]

        )

        for key in model:

            model[key] = sum(

                scores[i] *

                client_updates[i]["params"][key]

                for i in range(

                    len(client_updates)

                )

            )

        return model
