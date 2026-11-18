"""SOM helpers shared by the v1 (paper) and v2 classifiers."""

from __future__ import annotations

import numpy as np
from minisom import MiniSom


def train_class_soms(X, y, classes, *, map_size, sigma, learning_rate,
                     neighborhood_function, n_iterations, rng):
    """Train one square SOM per class on that class's samples only.

    Weights are initialised with random samples of the class (as in the paper).
    `rng` is a numpy Generator; one seed is drawn from it per class.
    """
    soms = []
    for c in classes:
        Xc = X[y == c]
        som = MiniSom(
            map_size,
            map_size,
            X.shape[1],
            sigma=sigma,
            learning_rate=learning_rate,
            neighborhood_function=neighborhood_function,
            random_seed=int(rng.integers(2**31 - 1)),
        )
        som.random_weights_init(Xc)
        som.train_random(Xc, n_iterations)
        soms.append(som)
    return soms


def bmu_distances(soms, X) -> np.ndarray:
    """Matrix (n_samples, n_maps) of Euclidean distances to the BMU on each map."""
    out = np.empty((X.shape[0], len(soms)))
    for k, som in enumerate(soms):
        W = som.get_weights().reshape(-1, X.shape[1])
        out[:, k] = np.sqrt(((X[:, None, :] - W[None, :, :]) ** 2).sum(-1)).min(1)
    return out
