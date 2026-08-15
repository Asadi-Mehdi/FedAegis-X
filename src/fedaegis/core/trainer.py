from fedaegis.metrics.classification import evaluate


class FederatedTrainer:

    def __init__(
        self,
        clients,
        server,
        history,
        rounds
    ):

        self.clients = clients
        self.server = server
        self.history = history
        self.rounds = rounds

    def fit(
        self,
        X_test,
        y_test
    ):

        global_parameters = None
        global_model = None

        for round_number in range(
            1,
            self.rounds + 1
        ):

            reports = []

            for client in self.clients:

                if global_parameters is not None:

                    client.update_global_model(
                        global_parameters
                    )

                report = client.train()

                reports.append(
                    report
                )

            global_parameters = (
                self.server.aggregate(
                    reports
                )
            )

            self.server.global_parameters = (
                global_parameters
            )

            self.clients[0].update_global_model(
                global_parameters
            )

            global_model = (
                self.clients[0].model
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
                round_number
            )

            print(
                f"Round {round_number} -> "
                f"Accuracy="
                f"{metrics['accuracy']:.4f}"
            )

            for report in reports:

                print(
                    f"  Client "
                    f"{report.client_id} | "
                    f"FNR="
                    f"{report.fnr:.4f} | "
                    f"Balance="
                    f"{report.class_balance:.4f} | "
                    f"Reliability="
                    f"{report.reliability:.4f}"
                )

        return global_model
