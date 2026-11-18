"""KohonenDA v3 (experimental): neuron-level statistics on top of per-class SOMs.

Two ideas tested against v2:

1. Per-neuron scale.  Each class map becomes a Gaussian mixture centred on its
   neurons: weight ~ number of training objects the neuron wins (Laplace-smoothed),
   variance estimated from those objects and shrunk towards the class-pooled variance
   (`shrinkage` pseudo-objects). covariance="spherical" (one variance per neuron) or
   "diag" (one per neuron and feature; down-weights uninformative features locally).
   -> log p(x | c) for every class. Used for
      * scoring="density": Bayes rule argmax_c log p(x|c) + log prior_c,
      * class-wise conformal p-values with nonconformity -log p(x|c),
      * pooled novelty p-value with nonconformity -log sum_c prior_c p(x|c).

2. More than the nearest neuron.  scoring="joint_sorted": logistic regression on the
   sorted log distances to all neurons of every map (permutation-invariant, so features
   from fold maps and the final maps are comparable). scoring="joint_density": logistic
   regression on the K class log-likelihoods relative to the best class (clipped at -50;
   raw log-likelihoods of far objects reach -1e4 and wreck the feature scaling).

All calibration / stacking data are out-of-fold (cross-fitted), as in v2.
Both covariance types are fitted, so `covariance` and `scoring` can be switched after
fitting without refitting the maps.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .som import bmu_distances, train_class_soms

_EPS = 1e-12
COVARIANCES = ("spherical", "diag")


def _sq_dists(X, W):
    return ((X[:, None, :] - W[None, :, :]) ** 2).sum(-1)


class NeuronMixture:
    """Gaussian mixture per class built from the neurons of that class's SOM."""

    def __init__(self, soms, X, y_idx, *, covariance, shrinkage, reg_var, trim=0.0):
        self.covariance = covariance
        n_dim = X.shape[1]
        self.W, self.log_w, self.var = [], [], []
        for k, som in enumerate(soms):
            W = som.get_weights().reshape(-1, n_dim)
            Xc = X[y_idx == k]
            d2 = _sq_dists(Xc, W)
            bmu = d2.argmin(1)
            if trim > 0:
                # same trimming as for the SOM fit: worst-fitting objects don't enter
                # the neuron statistics either (they would inflate the variances)
                n_keep = int(np.ceil((1.0 - trim) * len(Xc)))
                keep = np.argsort(d2[np.arange(len(Xc)), bmu], kind="stable")[:n_keep]
                Xc, bmu = Xc[keep], bmu[keep]
            R = Xc - W[bmu]
            J = len(W)
            n_j = np.bincount(bmu, minlength=J).astype(float)
            if covariance == "spherical":
                pooled = (R ** 2).sum() / (len(Xc) * n_dim)
                ss = np.bincount(bmu, weights=(R ** 2).sum(1), minlength=J) / n_dim
                var = (ss + shrinkage * pooled) / (n_j + shrinkage) + reg_var
                var = np.repeat(var[:, None], n_dim, 1)
            elif covariance == "diag":
                pooled = (R ** 2).mean(0)
                ss = np.stack([np.bincount(bmu, weights=R[:, d] ** 2, minlength=J)
                               for d in range(n_dim)], 1)
                var = (ss + shrinkage * pooled) / (n_j[:, None] + shrinkage) + reg_var
            else:
                raise ValueError(f"unknown covariance={covariance!r}")
            self.W.append(W)
            self.var.append(var)
            self.log_w.append(np.log((n_j + 1) / (n_j.sum() + J)))

    def log_density(self, X) -> np.ndarray:
        """(n_samples, n_classes) matrix of log p(x | c)."""
        out = np.empty((len(X), len(self.W)))
        for k, (W, var, lw) in enumerate(zip(self.W, self.var, self.log_w)):
            diff2 = (X[:, None, :] - W[None, :, :]) ** 2
            comp = lw - 0.5 * (np.log(2 * np.pi * var).sum(1)
                               + (diff2 / var[None]).sum(-1))
            out[:, k] = logsumexp(comp, axis=1)
        return out


def sorted_log_dists(soms, X) -> np.ndarray:
    """(n_samples, n_classes * n_neurons): per map, log distances to all neurons, sorted."""
    feats = []
    for som in soms:
        W = som.get_weights().reshape(-1, X.shape[1])
        feats.append(np.sort(np.log(np.sqrt(_sq_dists(X, W)) + _EPS), axis=1))
    return np.hstack(feats)


def _relative(L, floor=-50.0):
    """Log likelihood relative to the best class, clipped: stable stacking features."""
    return np.clip(L - L.max(1, keepdims=True), floor, 0.0)


def _conformal(cal_sorted, scores):
    n_ge = len(cal_sorted) - np.searchsorted(cal_sorted, scores, side="left")
    return (1 + n_ge) / (len(cal_sorted) + 1)


class KohonenDA3(ClassifierMixin, BaseEstimator):
    """Per-class SOMs with neuron-level mixtures and richer stacking (experimental).

    Parameters (SOM and cross-fitting ones as in KohonenDA2)
    ----------
    scoring : {"density", "joint_density", "joint_sorted", "joint_min"}
        "joint_min" = v2's joint scoring (log distance to the nearest neuron per map).
    covariance : {"spherical", "diag"}
        Neuron variance model used by "density", "joint_density", p-values and novelty.
    shrinkage : float
        Pseudo-object count pulling neuron variances towards the class-pooled variance.
    reg_var : float
        Added to every variance (features are assumed standardised).
    reject_alpha, reject_label : reject objects whose pooled novelty p-value < alpha.
    """

    def __init__(self, map_size=3, sigma=1.0, learning_rate=0.5,
                 neighborhood_function="gaussian", n_iterations=1000, trim=0.0,
                 scoring="density", covariance="diag", shrinkage=5.0, reg_var=1e-2,
                 cv=5, C=1.0, reject_alpha=None, reject_label=-1, random_state=None):
        self.map_size = map_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.neighborhood_function = neighborhood_function
        self.n_iterations = n_iterations
        self.trim = trim
        self.scoring = scoring
        self.covariance = covariance
        self.shrinkage = shrinkage
        self.reg_var = reg_var
        self.cv = cv
        self.C = C
        self.reject_alpha = reject_alpha
        self.reject_label = reject_label
        self.random_state = random_state

    def _train_soms(self, X, y_idx, rng):
        return train_class_soms(
            X, y_idx, range(len(self.classes_)), map_size=self.map_size, sigma=self.sigma,
            learning_rate=self.learning_rate,
            neighborhood_function=self.neighborhood_function,
            n_iterations=self.n_iterations, rng=rng, trim=self.trim)

    def _mixtures(self, soms, X, y_idx):
        return {cov: NeuronMixture(soms, X, y_idx, covariance=cov, shrinkage=self.shrinkage,
                                   reg_var=self.reg_var, trim=self.trim)
                for cov in COVARIANCES}

    def _lr(self):
        return make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=5000))

    # ------------------------------------------------------------------ fitting
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        if len(self.classes_) < 2:
            raise ValueError("need at least two classes")
        rng = np.random.default_rng(self.random_state)
        y_idx = np.searchsorted(self.classes_, y)
        K = len(self.classes_)
        J = self.map_size ** 2

        oof_logd = {cov: np.empty((len(X), K)) for cov in COVARIANCES}
        oof_sorted = np.empty((len(X), K * J))
        skf = StratifiedKFold(self.cv, shuffle=True, random_state=int(rng.integers(2**31 - 1)))
        for tr, te in skf.split(X, y_idx):
            soms = self._train_soms(X[tr], y_idx[tr], rng)
            for cov, nm in self._mixtures(soms, X[tr], y_idx[tr]).items():
                oof_logd[cov][te] = nm.log_density(X[te])
            oof_sorted[te] = sorted_log_dists(soms, X[te])

        self.soms_ = self._train_soms(X, y_idx, rng)
        self.mixtures_ = self._mixtures(self.soms_, X, y_idx)
        self.log_prior_ = np.log(np.bincount(y_idx, minlength=K) / len(y_idx))

        self.class_calib_, self.novelty_calib_, self.joint_density_ = {}, {}, {}
        for cov in COVARIANCES:
            L = oof_logd[cov]
            self.class_calib_[cov] = [np.sort(-L[y_idx == k, k]) for k in range(K)]
            self.novelty_calib_[cov] = np.sort(-logsumexp(L + self.log_prior_, axis=1))
            self.joint_density_[cov] = self._lr().fit(_relative(L), y_idx)
        self.joint_sorted_ = self._lr().fit(oof_sorted, y_idx)
        # v2-style joint on log min distances (the first sorted feature of each map)
        self.joint_min_ = self._lr().fit(oof_sorted[:, ::J], y_idx)
        self.oof_log_density_ = oof_logd
        return self

    # --------------------------------------------------------------- inference
    def _check(self, X):
        check_is_fitted(self, "soms_")
        return check_array(X)

    def log_density(self, X) -> np.ndarray:
        return self.mixtures_[self.covariance].log_density(self._check(X))

    def bmu_distances(self, X) -> np.ndarray:
        return bmu_distances(self.soms_, self._check(X))

    def pvalues(self, X) -> np.ndarray:
        """Class-wise conformal p-values with nonconformity -log p(x | c)."""
        L = self.log_density(X)
        cal = self.class_calib_[self.covariance]
        return np.column_stack([_conformal(cal[k], -L[:, k]) for k in range(L.shape[1])])

    def novelty_pvalues(self, X) -> np.ndarray:
        """Pooled conformal p-value that x comes from any known class."""
        L = self.log_density(X)
        return _conformal(self.novelty_calib_[self.covariance],
                          -logsumexp(L + self.log_prior_, axis=1))

    def predict_set(self, X, alpha=0.05) -> np.ndarray:
        return self.pvalues(X) >= alpha

    def predict_proba(self, X) -> np.ndarray:
        X = self._check(X)
        if self.scoring == "density":
            L = self.mixtures_[self.covariance].log_density(X) + self.log_prior_
            return np.exp(L - logsumexp(L, axis=1, keepdims=True))
        if self.scoring == "joint_density":
            return self.joint_density_[self.covariance].predict_proba(
                _relative(self.mixtures_[self.covariance].log_density(X)))
        S = sorted_log_dists(self.soms_, X)
        if self.scoring == "joint_sorted":
            return self.joint_sorted_.predict_proba(S)
        if self.scoring == "joint_min":
            return self.joint_min_.predict_proba(S[:, ::self.map_size ** 2])
        raise ValueError(f"unknown scoring={self.scoring!r}")

    def predict(self, X) -> np.ndarray:
        pred = self.classes_[self.predict_proba(X).argmax(1)]
        if self.reject_alpha is not None:
            rejected = self.novelty_pvalues(X) < self.reject_alpha
            if rejected.any():
                try:
                    pred = pred.astype(np.result_type(pred, np.asarray(self.reject_label)))
                except TypeError:
                    pred = pred.astype(object)
                pred[rejected] = self.reject_label
        return pred
