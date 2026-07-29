import copy
import numpy as np

from .base import BaseAggregator


class WeightedFedAvg(BaseAggregator):

    def aggregate(
            self,
            client_updates):

        total = sum(

            c["samples"]

            for c in client_updates

        )

        model = copy.deepcopy(

            client_updates[0]["params"]

        )

        for key in model:

            model[key] = sum(

                client["params"][key] *

                client["samples"] / total

                for client in client_updates

            )

        return model
