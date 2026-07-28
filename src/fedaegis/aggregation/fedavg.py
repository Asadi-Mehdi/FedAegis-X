import copy
import numpy as np

from .base import BaseAggregator


class FedAvgAggregator(BaseAggregator):

    def aggregate(

            self,

            client_updates):

        global_model = copy.deepcopy(

            client_updates[0]

        )

        for key in global_model.keys():

            global_model[key] = np.mean(

                [

                    update[key]

                    for update in client_updates

                ],

                axis=0

            )

        return global_model
