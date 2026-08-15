from fedaegis.core.client_report import ClientReport
from fedaegis.trust.reliability import ReliabilityCalculator


class Client:

    def __init__(
        self,
        client_id,
        model,
        X,
        y
    ):
        self.client_id = client_id
        self.model = model
        self.X = X
        self.y = y

    def train(self):

        self.model.fit(
            self.X,
            self.y
        )

        parameters = self.model.get_parameters()

        samples = len(self.X)

        # Commit #0010 baseline reliability.
        # Real local FNR/class-balance evaluation will replace
        # these values in the next implementation stage.
        fnr = 0.0
        class_balance = 1.0

        reliability = ReliabilityCalculator.calculate(
            fnr=fnr,
            samples=samples,
            class_balance=class_balance
        )

        return ClientReport(
            client_id=self.client_id,
            samples=samples,
            fnr=fnr,
            class_balance=class_balance,
            reliability=reliability,
            parameters=parameters
        )

    def update_global_model(
        self,
        global_parameters
    ):

        self.model.set_parameters(
            global_parameters
        )

    def predict(self, X):

        return self.model.predict(X)class Client:

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

        len(self.X),

    "cost_score":

        float(len(self.X))

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
from fedaegis.core.client_report import ClientReport
from fedaegis.trust.reliability import ReliabilityCalculator

class_balance = 1.0

fnr = 0.0

reliability = ReliabilityCalculator.calculate(

    fnr,

    len(self.X),

    class_balance

)

return ClientReport(

    client_id=self.client_id,

    samples=len(self.X),

    fnr=fnr,

    class_balance=class_balance,

    reliability=reliability,

    parameters=self.model.get_parameters()

)
