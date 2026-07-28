class Server:

    def __init__(
            self,
            aggregator):

        self.aggregator = aggregator

    def aggregate(
            self,
            client_updates):

        return self.aggregator.aggregate(
            client_updates
        )
