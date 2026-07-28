import copy
import numpy as np

from .base import BaseAggregator


class FedAvgAggregator(BaseAggregator):

    def aggregate(self, client_parameters):

        global_model = copy.deepcopy(client_parameters[0])

        for key in global_model:

            tensors = [
                c[key]
                for c in client_parameters
            ]

            global_model[key] = np.mean(
                tensors,
                axis=0
            )

        return global_model
