"""Larger benchmark datasets from OpenML, cached under <repo>/data/openml.

Each loader returns (X, y) with float features and integer labels 0..K-1.
Large sets are subsampled (stratified, fixed seed) to keep the benchmark tractable;
MNIST is reduced to 50 PCA components (fitted on its own subsample: the PCA is part of
the data definition here, all methods see the same features).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

DATA_HOME = Path(__file__).resolve().parent.parent / "data" / "openml"
MAX_N = 20_000

# name -> OpenML data_id
OPENML_IDS = {
    "segment": 36,
    "satimage": 182,
    "optdigits": 28,
    "pendigits": 32,
    "letter": 6,
    "covertype": 1596,
    "mnist_pca50": 554,
}


def _fetch(data_id):
    kw = dict(data_id=data_id, as_frame=False, data_home=str(DATA_HOME))
    try:
        b = fetch_openml(parser="auto", **kw)
    except TypeError:  # pandas ARFF parser chokes on numeric categorical targets
        b = fetch_openml(parser="liac-arff", **kw)
    X = np.asarray(b.data, dtype=float)
    _, y = np.unique(np.asarray(b.target).astype(str), return_inverse=True)
    return X, y


def _subsample(X, y, n, seed=0):
    if len(X) <= n:
        return X, y
    X, _, y, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return X, y


def _drop_constant(X):
    return X[:, X.std(0) > 0]


def load(name):
    """Processed (X, y); cached as a plain .npz (no pickles) after the first call, so
    parallel workers never touch the OpenML cache concurrently."""
    cache = DATA_HOME / f"{name}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            return z["X"], z["y"]
    X, y = _fetch(OPENML_IDS[name])
    X, y = _subsample(X, y, MAX_N)
    if name == "mnist_pca50":
        X = PCA(50, random_state=0).fit_transform(X / 255.0)
    X = _drop_constant(X)
    DATA_HOME.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, y=y)
    return X, y


NAMES = list(OPENML_IDS)
