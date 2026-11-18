"""SOM helpers shared by the v1 (paper) and v2 classifiers."""

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


def train_class_soms(X, y, classes, *, map_size, sigma, learning_rate,
                     neighborhood_function, n_iterations, rng, trim=0.0):
    """Train one square SOM per class on that class's samples only.

    Weights are initialised with random samples of the class (as in the paper).
    `rng` is a numpy Generator; one seed is drawn from it per class.

    trim : fraction in [0, 1). If > 0, after a first fit the `trim` share of the class's
        samples with the largest BMU distance is dropped and the map is re-initialised
        and retrained on the rest (robustness to gross outliers in the training data).
    """
    if not 0.0 <= trim < 1.0:
        raise ValueError("trim must be in [0, 1)")
    kw = dict(map_size=map_size, sigma=sigma, learning_rate=learning_rate,
              neighborhood_function=neighborhood_function, n_iterations=n_iterations)
    soms = []
    for c in classes:
        Xc = X[y == c]
        seed = int(rng.integers(2**31 - 1))
        som = _fit_som(Xc, seed, **kw)
        n_keep = int(np.ceil((1.0 - trim) * len(Xc)))
        if trim > 0 and n_keep < len(Xc):
            d = bmu_distances([som], Xc)[:, 0]
            keep = np.argsort(d, kind="stable")[:n_keep]
            som = _fit_som(Xc[keep], seed, **kw)
        soms.append(som)
    return soms


def bmu_distances(soms, X) -> np.ndarray:
    """Matrix (n_samples, n_maps) of Euclidean distances to the BMU on each map."""
    out = np.empty((X.shape[0], len(soms)))
    for k, som in enumerate(soms):
        W = som.get_weights().reshape(-1, X.shape[1])
        out[:, k] = np.sqrt(((X[:, None, :] - W[None, :, :]) ** 2).sum(-1)).min(1)
    return out
