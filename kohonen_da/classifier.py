"""Faithful reimplementation of the method from

    Komarov I.V., Kuravsky L.S. "Discriminant Analysis Based on Kohonen Statistics".
    Modelling and Data Analysis, 2023, 13(4), 176-182. doi:10.17759/mda.2023130411

Algorithm (Section 3 of the paper):

1. For every class c train a separate SOM on the training samples of class c only.
   Weights are initialised with random samples from the data; distances are Euclidean.
2. For calibration samples of class c compute the distance to the best-matching unit
   (BMU) on map c.
3. Build a histogram of these distances with n = 1 + floor(log2 N) bins (Sturges) and
   normalise frequencies so that they sum to 1 -> empirical P(d | c).
4. For a new object x compute d_c(x) on every map, look up the bin frequency in the
   histogram of class c, normalise across classes and predict the argmax.

Points the paper leaves unspecified are exposed as parameters (SOM hyper-parameters,
behaviour when every class gets probability 0).
"""

from __future__ import annotations

import math

import numpy as np
from minisom import MiniSom
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array


def sturges_bins(n_obs: int) -> int:
    """Number of histogram bins by Sturges' rule: 1 + floor(log2 N)."""
    if n_obs < 1:
        raise ValueError("need at least one observation")
    return 1 + int(math.floor(math.log2(n_obs)))


class KohonenDA(ClassifierMixin, BaseEstimator):
    """Per-class SOM discriminant analysis (paper-faithful variant).

    Parameters
    ----------
    map_size : int
        Side of the square map; the paper uses 3 (3x3 = 9 neurons per class).
    sigma, learning_rate, neighborhood_function, n_iterations :
        SOM training hyper-parameters (not reported in the paper; MiniSom defaults
        for sigma/learning_rate/neighbourhood).
    zero_fallback : {"nearest", "none"}
        What to do when the histogram probability is 0 for every class (distance
        outside every histogram's range). The paper does not cover this case.
        "nearest" -> class with the smallest BMU distance; "none" -> predict -1.
    random_state : int or None
    """

    def __init__(
        self,
        map_size: int = 3,
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        neighborhood_function: str = "gaussian",
        n_iterations: int = 1000,
        zero_fallback: str = "nearest",
        random_state: int | None = None,
    ):
        self.map_size = map_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.neighborhood_function = neighborhood_function
        self.n_iterations = n_iterations
        self.zero_fallback = zero_fallback
        self.random_state = random_state

    # ------------------------------------------------------------------ fitting
    def fit(self, X, y, X_calib=None, y_calib=None):
        """Train one SOM per class on (X, y) and build distance histograms.

        If X_calib/y_calib are given, histograms are built from them (the paper uses
        "the test sample of each a-priori class"); otherwise from the training data.
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)

        self.soms_ = []
        for c in self.classes_:
            Xc = X[y == c]
            som = MiniSom(
                self.map_size,
                self.map_size,
                X.shape[1],
                sigma=self.sigma,
                learning_rate=self.learning_rate,
                neighborhood_function=self.neighborhood_function,
                random_seed=int(rng.integers(2**31 - 1)),
            )
            som.random_weights_init(Xc)  # init with random samples (as in the paper)
            som.train_random(Xc, self.n_iterations)
            self.soms_.append(som)

        if X_calib is None:
            X_calib, y_calib = X, y
        else:
            X_calib, y_calib = check_X_y(X_calib, y_calib)
        self.calibrate(X_calib, y_calib)
        return self

    def calibrate(self, X, y):
        """Build the normalised histogram of BMU distances for each class."""
        D = self.bmu_distances(X)
        self.hist_freqs_, self.hist_edges_ = [], []
        for k, c in enumerate(self.classes_):
            d = D[y == c, k]
            n_bins = sturges_bins(len(d))
            counts, edges = np.histogram(d, bins=n_bins)
            self.hist_freqs_.append(counts / counts.sum())
            self.hist_edges_.append(edges)
        return self

    # --------------------------------------------------------------- inference
    def bmu_distances(self, X) -> np.ndarray:
        """Matrix (n_samples, n_classes) of Euclidean distances to the BMU on each map."""
        check_is_fitted(self, "soms_")
        X = check_array(X)
        out = np.empty((X.shape[0], len(self.soms_)))
        for k, som in enumerate(self.soms_):
            W = som.get_weights().reshape(-1, X.shape[1])
            out[:, k] = np.sqrt(((X[:, None, :] - W[None, :, :]) ** 2).sum(-1)).min(1)
        return out

    def histogram_scores(self, X) -> np.ndarray:
        """Unnormalised scores: frequency of the bin of d_c(x) in histogram c (0 outside)."""
        D = self.bmu_distances(X)
        S = np.zeros_like(D)
        for k in range(len(self.classes_)):
            freqs, edges = self.hist_freqs_[k], self.hist_edges_[k]
            d = D[:, k]
            idx = np.searchsorted(edges, d, side="right") - 1
            idx[d == edges[-1]] = len(freqs) - 1  # last bin is right-closed (numpy convention)
            inside = (idx >= 0) & (idx < len(freqs))
            S[inside, k] = freqs[idx[inside]]
        return S

    def predict_proba(self, X) -> np.ndarray:
        """Scores normalised across classes. Rows where every score is 0 are NaN."""
        S = self.histogram_scores(X)
        tot = S.sum(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(tot > 0, S / tot, np.nan)

    def predict(self, X) -> np.ndarray:
        P = self.predict_proba(X)
        undecided = np.isnan(P).any(1)
        pred = np.empty(len(P), dtype=self.classes_.dtype)
        pred[~undecided] = self.classes_[np.argmax(P[~undecided], 1)]
        if undecided.any():
            if self.zero_fallback == "nearest":
                D = self.bmu_distances(np.asarray(X)[undecided])
                pred[undecided] = self.classes_[np.argmin(D, 1)]
            elif self.zero_fallback == "none":
                pred = pred.astype(object)
                pred[undecided] = -1
            else:
                raise ValueError(f"unknown zero_fallback={self.zero_fallback!r}")
        return pred

    def n_undecided(self, X) -> int:
        """How many objects get probability 0 for every class (method is undefined there)."""
        return int(np.isnan(self.predict_proba(X)).any(1).sum())
