"""Benchmark datasets: real (scikit-learn) and synthetic non-Gaussian ones.

Every loader takes a seed and returns (X, y). Synthetic generators draw a fresh sample
per seed from a *fixed* generative model (structure seeds are constant), so repeated
splits measure sampling variability of one problem, not different problems.
Synthetic sets have 2 informative dimensions plus `n_noise` pure-noise dimensions.
"""

from __future__ import annotations

import numpy as np
from sklearn import datasets as skd

N_PER_CLASS = 100
N_NOISE = 2


def _add_noise_dims(X, rng, n_noise=N_NOISE):
    return np.hstack([X, rng.normal(size=(len(X), n_noise))])


def multimodal(seed, n=N_PER_CLASS):
    """3 classes, each a mixture of 3 well-separated blobs at interleaved positions."""
    structure = np.random.default_rng(12345)
    grid = np.array([(i, j) for i in range(3) for j in range(3)], float) * 3.0
    labels = structure.permutation(np.repeat(np.arange(3), 3))
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(3):
        centres = grid[labels == c]
        comp = rng.integers(0, 3, n)
        X.append(centres[comp] + rng.normal(scale=0.6, size=(n, 2)))
        y.append(np.full(n, c))
    return _add_noise_dims(np.vstack(X), rng), np.concatenate(y)


def rings(seed, n=N_PER_CLASS):
    """3 concentric rings (radii 1, 2, 3) with radial noise."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c, r in enumerate((1.0, 2.0, 3.0)):
        t = rng.uniform(0, 2 * np.pi, n)
        rad = r + rng.normal(scale=0.2, size=n)
        X.append(np.column_stack([rad * np.cos(t), rad * np.sin(t)]))
        y.append(np.full(n, c))
    return _add_noise_dims(np.vstack(X), rng), np.concatenate(y)


def moons(seed, n=N_PER_CLASS):
    rng = np.random.default_rng(seed)
    X, y = skd.make_moons(2 * n, noise=0.25, random_state=int(rng.integers(2**31 - 1)))
    return _add_noise_dims(X, rng), y


def xor(seed, n=N_PER_CLASS):
    """2 classes; class = sign(x1) xor sign(x2), Gaussian blobs at (+-1.5, +-1.5)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c, quads in enumerate([[(1, 1), (-1, -1)], [(1, -1), (-1, 1)]]):
        q = np.array(quads, float)[rng.integers(0, 2, n)] * 1.5
        X.append(q + rng.normal(scale=0.7, size=(n, 2)))
        y.append(np.full(n, c))
    return _add_noise_dims(np.vstack(X), rng), np.concatenate(y)


def heavy_tailed(seed, n=N_PER_CLASS, df=1.5):
    """3 classes at triangle vertices with multivariate Student-t (df=1.5) noise in 4-D."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0, 0, 0], [3, 0, 0, 0], [1.5, 2.6, 0, 0]], float)
    X, y = [], []
    for c in range(3):
        z = rng.normal(size=(n, 4))
        w = np.sqrt(df / rng.chisquare(df, size=(n, 1)))
        X.append(centres[c] + z * w)
        y.append(np.full(n, c))
    return np.vstack(X), np.concatenate(y)


def _real(loader):
    def load(seed):
        return loader(return_X_y=True)
    return load


REAL = {
    "iris": _real(skd.load_iris),
    "wine": _real(skd.load_wine),
    "breast_cancer": _real(skd.load_breast_cancer),
    "digits": _real(skd.load_digits),
}
SYNTHETIC = {
    "multimodal": multimodal,
    "rings": rings,
    "moons": moons,
    "xor": xor,
    "heavy_tailed": heavy_tailed,
}
ALL = {**REAL, **SYNTHETIC}
