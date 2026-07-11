import numpy as np


def equal(k=5):
    return np.full(k, 1.0 / k)


def entropy(B):
    B = np.asarray(B, float) + 1e-9
    P = B / B.sum(axis=0, keepdims=True)
    e = -(P * np.log(P)).sum(axis=0) / np.log(B.shape[0])
    d = 1.0 - e
    return d / d.sum()


def pca(B):
    B = np.asarray(B, float)
    Z = (B - B.mean(axis=0)) / (B.std(axis=0) + 1e-9)
    cov = np.cov(Z.T)
    _, vec = np.linalg.eigh(cov)
    load = np.abs(vec[:, -1])
    return load / load.sum()


def critic(B):
    B = np.asarray(B, float)
    sd = B.std(axis=0)
    corr = np.corrcoef(B.T)
    conflict = (1.0 - corr).sum(axis=0)
    c = sd * conflict
    return c / c.sum()
