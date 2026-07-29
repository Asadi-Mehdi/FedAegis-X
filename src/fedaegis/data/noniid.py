import numpy as np


class DirichletPartitioner:

    def __init__(
            self,
            alpha=0.5,
            random_state=42):

        self.alpha = alpha
        self.random_state = random_state

    def split(
            self,
            X,
            y,
            clients):

        np.random.seed(self.random_state)

        classes = np.unique(y)

        client_indices = [[] for _ in range(clients)]

        for cls in classes:

            idx = np.where(y == cls)[0]

            np.random.shuffle(idx)

            proportions = np.random.dirichlet(

                [self.alpha] * clients

            )

            cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]

            parts = np.split(idx, cuts)

            for cid, part in enumerate(parts):

                client_indices[cid].extend(part.tolist())

        partitions = []

        for idx in client_indices:

            idx = np.array(idx)

            partitions.append(

                (

                    X[idx],

                    y[idx]

                )

            )

        return partitions
