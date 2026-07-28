class Server:

    def __init__(
            self,
            aggregator):

        self.aggregator = aggregator

        self.global_parameters = None

    def aggregate(
            self,
            updates):

        self.global_parameters = self.aggregator.aggregate(
            updates
        )

        return self.global_parameters

    def broadcast(self):

        return self.global_parameters
