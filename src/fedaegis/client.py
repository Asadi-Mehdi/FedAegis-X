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

return {

    "params":

        self.model.get_parameters(),

    "samples":

        len(self.X)

}
    def update_global_model(
            self,
            global_parameters):

        self.model.set_parameters(
            global_parameters
        )

    def predict(
            self,
            X):

        return self.model.predict(X)
