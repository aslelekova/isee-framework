import numpy as np

from .data import DIMENSIONS, DIM_SIGN

SIGNS = np.array([DIM_SIGN[d] for d in DIMENSIONS], float)


def benefit_oriented(S):
    S = np.asarray(S, float)
    return np.where(SIGNS > 0, S, 1.0 - S)


def additive(S, w):
    S = np.asarray(S, float)
    return S @ (np.asarray(w, float) * SIGNS)


def geometric(S, w, floor=0.1):
    B = floor + (1.0 - floor) * benefit_oriented(S)
    return np.exp(np.log(B) @ np.asarray(w, float))


def ranks(scores):
    return (-np.asarray(scores, float)).argsort().argsort() + 1
