class Client:

    def __init__(
            self,
            client_id,
            model,
            X,
            y):

        self.client_id = client_id

        self.model = model

        self.X = X

        self.y = y

    def train(self):

        self.model.fit(
            self.X,
            self.y
        )

        return self.model.get_parameters()

    def predict(self, X):

        return self.model.predict(X)
