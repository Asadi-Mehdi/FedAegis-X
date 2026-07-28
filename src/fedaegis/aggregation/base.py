from abc import ABC, abstractmethod


class BaseAggregator(ABC):

    @abstractmethod
    def aggregate(self, client_parameters):
        pass
