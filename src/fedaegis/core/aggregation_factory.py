from fedaegis.aggregation.fedavg import FedAvgAggregator
from fedaegis.aggregation.weighted import WeightedFedAvg


class AggregationFactory:

    @staticmethod
    def create(name):

        if name == "fedavg":

            return FedAvgAggregator()

        if name == "weighted_fedavg":

            return WeightedFedAvg()

        raise ValueError(name)
