"""SOM helpers shared by the v1, v2 and v3 classifiers."""

from __future__ import annotations

import numpy as np
from minisom import MiniSom


def _fit_som(Xc, seed, *, map_size, sigma, learning_rate, neighborhood_function,
             n_iterations):
    som = MiniSom(
        map_size,
        map_size,
        Xc.shape[1],
        sigma=sigma,
        learning_rate=learning_rate,
        neighborhood_function=neighborhood_function,
        random_seed=seed,
    )
    som.random_weights_init(Xc)
    som.train_random(Xc, n_iterations)
    return som


CHUNK_ELEMS = 2 ** 22  # max elements of a temporary (rows, neurons, features) array


def _chunks(n_rows, per_row):
    step = max(1, CHUNK_ELEMS // max(1, per_row))
    for i in range(0, n_rows, step):
        yield slice(i, min(i + step, n_rows))


def sq_dists(X, W) -> np.ndarray:
    """(n, m) squared Euclidean distances, computed in memory-bounded row chunks
    (same formula as full broadcasting, so results are bit-identical)."""
    out = np.empty((len(X), len(W)))
    for sl in _chunks(len(X), W.size):
        out[sl] = ((X[sl, None, :] - W[None, :, :]) ** 2).sum(-1)
    return out


AUTO_TRIM_K = 3.0
AUTO_TRIM_NEIGHBOURS = 5


def check_trim(trim):
    if isinstance(trim, str):
        if trim != "auto":
            raise ValueError("trim must be a float in [0, 1) or 'auto'")
    elif not 0.0 <= trim < 1.0:
        raise ValueError("trim must be in [0, 1)")


def auto_keep(Xc):
    """Indices of a class's objects that are not robust outliers ("auto" trimming).

    Outlier score: mean distance to the 5 nearest objects of the same class (computed
    on the data, not on a SOM: gross outliers capture neurons of their own, which makes
    their BMU distance small and hides them). An object is dropped if its log score
    exceeds median + 3 * 1.4826 * MAD; this tolerates up to 50% contamination and drops
    very little from clean data. Returns None if nothing is dropped.
    """
    n = len(Xc)
    k = min(AUTO_TRIM_NEIGHBOURS, n - 1)
    if k < 1:
        return None
    knn = np.empty(n)
    for sl in _chunks(n, Xc.size):
        d = np.sqrt(((Xc[sl, None, :] - Xc[None, :, :]) ** 2).sum(-1))
        d[np.arange(sl.stop - sl.start), np.arange(sl.start, sl.stop)] = np.inf
        knn[sl] = np.sort(d, axis=1)[:, :k].mean(1)
    score = np.log(knn + 1e-12)
    med = np.median(score)
    mad = 1.4826 * np.median(np.abs(score - med))
    if mad == 0:
        return None
    keep = np.flatnonzero(score <= med + AUTO_TRIM_K * mad)
    return keep if len(keep) < n else None


def trim_keep(d, trim):
    """Indices kept by fixed-fraction trimming, given BMU distances `d` (None = keep all).

    Drops the `trim` share with the largest distance (kept in order of distance).
    """
    n_keep = int(np.ceil((1.0 - trim) * len(d)))
    if trim > 0 and n_keep < len(d):
        return np.argsort(d, kind="stable")[:n_keep]
    return None


def train_class_soms(X, y, classes, *, map_size, sigma, learning_rate,
                     neighborhood_function, n_iterations, rng, trim=0.0):
    """Train one square SOM per class on that class's samples only.

    Weights are initialised with random samples of the class (as in the paper).
    `rng` is a numpy Generator; one seed is drawn from it per class.

    trim : fraction in [0, 1) or "auto".
        float: after a first fit, the `trim` share of the class's samples with the largest
        BMU distance is dropped and the map is re-initialised and retrained on the rest.
        "auto": robust outliers (see `auto_keep`) are removed before the (single) fit.
    """
    check_trim(trim)
    kw = dict(map_size=map_size, sigma=sigma, learning_rate=learning_rate,
              neighborhood_function=neighborhood_function, n_iterations=n_iterations)
    soms = []
    for c in classes:
        Xc = X[y == c]
        seed = int(rng.integers(2**31 - 1))
        if trim == "auto":
            keep = auto_keep(Xc)
            soms.append(_fit_som(Xc if keep is None else Xc[keep], seed, **kw))
            continue
        som = _fit_som(Xc, seed, **kw)
        if trim:
            keep = trim_keep(bmu_distances([som], Xc)[:, 0], trim)
            if keep is not None:
                som = _fit_som(Xc[keep], seed, **kw)
        soms.append(som)
    return soms


def bmu_distances(soms, X) -> np.ndarray:
    """Matrix (n_samples, n_maps) of Euclidean distances to the BMU on each map."""
    out = np.empty((X.shape[0], len(soms)))
    for k, som in enumerate(soms):
        W = som.get_weights().reshape(-1, X.shape[1])
        out[:, k] = np.sqrt(sq_dists(X, W)).min(1)
    return out
