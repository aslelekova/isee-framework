import numpy as np
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


def select_k(B, k_range=range(2, 7), seed=42):
    out = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=50, random_state=seed).fit(B)
        out[k] = silhouette_score(B, km.labels_)
    return out


def kmeans(B, k, seed=42):
    km = KMeans(n_clusters=k, n_init=50, random_state=seed).fit(B)
    return km.labels_, km.cluster_centers_


def ward(B, k):
    Z = linkage(B, method="ward")
    labels = fcluster(Z, t=k, criterion="maxclust") - 1
    coph, _ = cophenet(Z, pdist(B))
    return labels, Z, coph


def agreement(labels_a, labels_b):
    return adjusted_rand_score(labels_a, labels_b)


def pca_project(B):
    Z = (B - B.mean(axis=0)) / (B.std(axis=0) + 1e-9)
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    var = s**2 / (s**2).sum()
    return Z @ Vt[:2].T, var[:2], Vt[:2]
