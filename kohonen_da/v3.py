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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .som import _chunks, auto_keep, bmu_distances, sq_dists, train_class_soms, trim_keep

_EPS = 1e-12
COVARIANCES = ("spherical", "diag")
AUTO_TOP, AUTO_TOP_MAX_FEATURES = 5, 100


def _sq_dists(X, W):
    return sq_dists(X, W)


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
            if trim:
                # same trimming as for the SOM fit: worst-fitting objects don't enter
                # the neuron statistics either (they would inflate the variances)
                keep = (auto_keep(Xc) if trim == "auto"
                        else trim_keep(np.sqrt(d2[np.arange(len(Xc)), bmu]), trim))
                if keep is not None:
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
            logdet = np.log(2 * np.pi * var).sum(1)
            for sl in _chunks(len(X), W.size):
                diff2 = (X[sl, None, :] - W[None, :, :]) ** 2
                comp = lw - 0.5 * (logdet + (diff2 / var[None]).sum(-1))
                out[sl, k] = logsumexp(comp, axis=1)
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
    sorted_top : int, "auto" or None
        "joint_sorted" uses only the `sorted_top` nearest neurons of every map
        (None = all K*J sorted distances; with large maps that stacker dominates the
        fit time and overfits). "auto": all if K*J <= 100, else the 5 nearest.
    reject_alpha, reject_label : reject objects whose pooled novelty p-value < alpha.
    """

    def __init__(self, map_size=3, sigma=1.0, learning_rate=0.5,
                 neighborhood_function="gaussian", n_iterations=1000, trim=0.0,
                 scoring="density", covariance="diag", shrinkage=5.0, reg_var=1e-2,
                 cv=5, C=1.0, sorted_top=None, reject_alpha=None, reject_label=-1,
                 random_state=None):
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
        self.sorted_top = sorted_top
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

        self.class_calib_, self.novelty_calib_ = {}, {}
        for cov in COVARIANCES:
            L = oof_logd[cov]
            self.class_calib_[cov] = [np.sort(-L[y_idx == k, k]) for k in range(K)]
            self.novelty_calib_[cov] = np.sort(-logsumexp(L + self.log_prior_, axis=1))
        # stacking models are fitted lazily on first use (same data -> same result)
        self.stackers_ = {}
        self.oof_log_density_ = oof_logd
        self.oof_sorted_ = oof_sorted
        self.y_idx_ = y_idx
        return self

    def _stack_features(self, name, L=None, S=None):
        J = self.map_size ** 2
        if name.startswith("joint_density"):
            return _relative(L)
        if name == "joint_sorted":
            m = self.sorted_top
            if m == "auto":
                # all neurons while the stacker is small; else the 5 nearest per map
                # (the full stacker is slow and overfits on large maps)
                m = None if len(self.classes_) * J <= AUTO_TOP_MAX_FEATURES else AUTO_TOP
            if m is None or m >= J:
                return S
            return S.reshape(len(S), -1, J)[:, :, :m].reshape(len(S), -1)
        if name == "joint_min":  # v2-style: log distance to the nearest neuron per map
            return S[:, ::J]
        raise ValueError(name)

    def _stacker(self, name):
        """Logistic regression for a stacked scoring, fitted on out-of-fold features."""
        key = (name, self.sorted_top) if name == "joint_sorted" else name
        if key not in self.stackers_:
            if name.startswith("joint_density"):
                F = _relative(self.oof_log_density_[name.split(":")[1]])
            else:
                F = self._stack_features(name, S=self.oof_sorted_)
            self.stackers_[key] = self._lr().fit(F, self.y_idx_)
        return self.stackers_[key]

    def oof_accuracy(self, scoring, covariance=None) -> float:
        """Out-of-fold accuracy of a scoring on the training data (for model selection).

        "density" is evaluated directly on the out-of-fold log-likelihoods; stacked
        scorings by an inner cross_val_predict on their out-of-fold features.
        """
        check_is_fitted(self, "oof_sorted_")
        cov = covariance or self.covariance
        y = self.y_idx_
        L = self.oof_log_density_[cov]
        if scoring == "density":
            pred = (L + self.log_prior_).argmax(1)
        else:
            F = (_relative(L) if scoring == "joint_density"
                 else self._stack_features(scoring, S=self.oof_sorted_))
            n_splits = int(min(5, np.bincount(y).min()))
            pred = cross_val_predict(self._lr(), F, y, cv=n_splits)
        return float((pred == y).mean())

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
            return self._stacker(f"joint_density:{self.covariance}").predict_proba(
                _relative(self.mixtures_[self.covariance].log_density(X)))
        if self.scoring in ("joint_sorted", "joint_min"):
            S = sorted_log_dists(self.soms_, X)
            return self._stacker(self.scoring).predict_proba(
                self._stack_features(self.scoring, S=S))
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


class KohonenDA3Auto(ClassifierMixin, BaseEstimator):
    """KohonenDA3 with map size and scoring chosen by out-of-fold accuracy.

    For every size in `map_sizes` a KohonenDA3 is fitted (its cross-fitting already
    yields out-of-fold scores) and scorings are evaluated by `KohonenDA3.oof_accuracy`;
    the best (size, scoring) is kept. Ties go to the smaller map, then to the earlier
    scoring in `scorings`. No test data is used.

    selection : {"full", "fast"}
        "full": every scoring at every size (the stacked scorings need a nested CV each).
        "fast": "density" (free: already out-of-fold) at every size; the other scorings
        only at the `n_stack_sizes` sizes with the best density accuracy.
    Remaining parameters are passed to KohonenDA3.
    """

    def __init__(self, map_sizes=(3, 4, 5), scorings=("density", "joint_density", "joint_sorted"),
                 covariance="diag", trim=0.0, shrinkage=5.0, reg_var=1e-2, cv=5, C=1.0,
                 sorted_top="auto", selection="full", n_stack_sizes=1,
                 sigma=1.0, learning_rate=0.5, neighborhood_function="gaussian",
                 n_iterations=1000, reject_alpha=None, reject_label=-1, random_state=None):
        self.map_sizes = map_sizes
        self.scorings = scorings
        self.covariance = covariance
        self.trim = trim
        self.shrinkage = shrinkage
        self.reg_var = reg_var
        self.cv = cv
        self.C = C
        self.sorted_top = sorted_top
        self.selection = selection
        self.n_stack_sizes = n_stack_sizes
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.neighborhood_function = neighborhood_function
        self.n_iterations = n_iterations
        self.reject_alpha = reject_alpha
        self.reject_label = reject_label
        self.random_state = random_state

    def _model(self, size):
        return KohonenDA3(
            map_size=size, covariance=self.covariance, trim=self.trim,
            shrinkage=self.shrinkage, reg_var=self.reg_var, cv=self.cv, C=self.C,
            sorted_top=self.sorted_top, sigma=self.sigma, learning_rate=self.learning_rate,
            neighborhood_function=self.neighborhood_function,
            n_iterations=self.n_iterations, reject_alpha=self.reject_alpha,
            reject_label=self.reject_label, random_state=self.random_state)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if self.selection not in ("full", "fast"):
            raise ValueError(f"unknown selection={self.selection!r}")
        best, best_key, self.selection_ = None, None, {}

        def consider(i, m, j, scoring):
            nonlocal best, best_key
            acc = m.oof_accuracy(scoring)
            self.selection_[(m.map_size, scoring)] = acc
            key = (acc, -i, -j)
            if best_key is None or key > best_key:
                best_key, best = key, (m, scoring)

        cheap = [j for j, sc in enumerate(self.scorings) if sc == "density"]
        stacked = [j for j, sc in enumerate(self.scorings) if sc != "density"]
        if self.selection == "full":
            for i, size in enumerate(self.map_sizes):
                m = self._model(size).fit(X, y)
                for j, scoring in enumerate(self.scorings):
                    consider(i, m, j, scoring)
        else:
            models = [self._model(size).fit(X, y) for size in self.map_sizes]
            dens = [models[i].oof_accuracy("density") for i in range(len(models))]
            for i, m in enumerate(models):
                for j in cheap:
                    consider(i, m, j, "density")
            order = sorted(range(len(models)), key=lambda i: (-dens[i], i))
            for i in order[: max(1, self.n_stack_sizes)]:
                for j in stacked:
                    consider(i, models[i], j, self.scorings[j])
            del models
        self.best_ = best[0].set_params(scoring=best[1])
        self.map_size_, self.scoring_ = self.best_.map_size, best[1]
        self.classes_ = self.best_.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def _b(self):
        check_is_fitted(self, "best_")
        return self.best_

    def predict(self, X):
        return self._b().predict(X)

    def predict_proba(self, X):
        return self._b().predict_proba(X)

    def pvalues(self, X):
        return self._b().pvalues(X)

    def novelty_pvalues(self, X):
        return self._b().novelty_pvalues(X)

    def predict_set(self, X, alpha=0.05):
        return self._b().predict_set(X, alpha)

    def log_density(self, X):
        return self._b().log_density(X)
