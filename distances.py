import numpy as np


def euclidean(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)


def cosine(a, b):
    a = np.array(a)
    b = np.array(b)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom < 1e-9:
        return 1.0

    similarity = np.dot(a, b) / denom

    return 1.0 - similarity


def manhattan(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.sum(np.abs(a - b))


def get_distance(metric):
    if metric == "cosine":
        return cosine

    if metric == "manhattan":
        return manhattan

    return euclidean