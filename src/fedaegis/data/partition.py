import numpy as np


def iid_partition(X, y, clients):

    x_parts = np.array_split(X, clients)

    y_parts = np.array_split(y, clients)

    return list(zip(x_parts, y_parts))
