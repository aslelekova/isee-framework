import numpy as np
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             silhouette_score)


def select_k(B, k_range=range(2, 11), seed=42):
    out = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=50, random_state=seed).fit(B)
        out[k] = {
            "silhouette": silhouette_score(B, km.labels_),
            "calinski_harabasz": calinski_harabasz_score(B, km.labels_),
        }
    return out


def bootstrap_stability(B, k, n_boot=200, seed=42):
    rng = np.random.default_rng(seed)
    ref = KMeans(n_clusters=k, n_init=50, random_state=seed).fit(B)
    aris = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(B), size=len(B))
        if len(np.unique(idx)) <= k:
            continue
        km = KMeans(n_clusters=k, n_init=10,
                    random_state=int(rng.integers(1e9))).fit(B[idx])
        boot_labels = cdist(B, km.cluster_centers_).argmin(axis=1)
        aris.append(adjusted_rand_score(ref.labels_, boot_labels))
    return float(np.mean(aris)), float(np.std(aris))


def bootstrap_stability_grouped(B, countries, k, n_boot=200, seed=42):
    rng = np.random.default_rng(seed)
    ref = KMeans(n_clusters=k, n_init=50, random_state=seed).fit(B)
    uniq = sorted(set(countries))
    members = {c: [j for j, cc in enumerate(countries) if cc == c]
               for c in uniq}
    aris = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.array([j for c in picked for j in members[c]])
        if len(np.unique(idx)) <= k:
            continue
        km = KMeans(n_clusters=k, n_init=10,
                    random_state=int(rng.integers(1e9))).fit(B[idx])
        boot_labels = cdist(B, km.cluster_centers_).argmin(axis=1)
        aris.append(adjusted_rand_score(ref.labels_, boot_labels))
    return float(np.mean(aris)), float(np.std(aris))


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
