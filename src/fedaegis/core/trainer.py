from fedaegis.metrics.classification import evaluate


class FederatedTrainer:

    def __init__(

            self,

            clients,

            server,

            history,

            rounds):

        self.clients = clients

        self.server = server

        self.history = history

        self.rounds = rounds

    def fit(

            self,

            X_test,

            y_test):

        global_parameters = None

        global_model = None

        for rnd in range(

                1,

                self.rounds + 1):

            updates = []

            for client in self.clients:

                if global_parameters is not None:

                    client.model.set_parameters(

                        global_parameters

                    )

                updates.append(

                    client.train()

                )

            global_parameters = self.server.aggregate(

                updates

            )

            global_model = self.clients[0].model

            global_model.set_parameters(

                global_parameters

            )

            prediction = global_model.predict(

                X_test

            )

            metrics = evaluate(

                y_test,

                prediction

            )

            self.history.add(

                metrics,

                rnd

            )

            print(

                f"Round {rnd} -> "

                f"Accuracy={metrics['accuracy']:.4f}"

            )

        return global_model
