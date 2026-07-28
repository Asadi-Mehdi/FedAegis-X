import numpy as np


def iid_partition(
        X,
        y,
        clients):

    idx = np.random.permutation(
        len(X)
    )

    X = X[idx]

    y = y[idx]

    x_parts = np.array_split(
        X,
        clients
    )

    y_parts = np.array_split(
        y,
        clients
    )

    return list(
        zip(
            x_parts,
            y_parts
        )
    )
