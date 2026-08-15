from fedaegis.aggregation.fedavg import FedAvgAggregator
from fedaegis.aggregation.weighted import WeightedFedAvg
from fedaegis.aggregation.dynamic_cost_aware import (
    DynamicCostAwareAggregator,
)


class AggregationFactory:

    @staticmethod
    def create(name):

        if name == "fedavg":
            return FedAvgAggregator()

        if name == "weighted_fedavg":
            return WeightedFedAvg()

        if name == "dynamic_cost":
            return DynamicCostAwareAggregator()

        raise ValueError(
            f"Unknown aggregation method: {name}"
        )
