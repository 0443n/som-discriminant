"""Additional baselines for benchmarking."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class GMMClassifier(ClassifierMixin, BaseEstimator):
    """Per-class Gaussian mixture + Bayes rule.

    The parametric counterpart of "one SOM per class": each class density is a
    mixture of `n_components` Gaussians (prototypes with covariances), and the
    class priors are the training frequencies.
    """

    def __init__(self, n_components: int = 3, covariance_type: str = "full",
                 reg_covar: float = 1e-3, random_state: int | None = None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.gmms_, priors = [], []
        for c in self.classes_:
            Xc = X[y == c]
            k = min(self.n_components, len(Xc))
            self.gmms_.append(GaussianMixture(
                k, covariance_type=self.covariance_type, reg_covar=self.reg_covar,
                random_state=self.random_state).fit(Xc))
            priors.append(len(Xc) / len(X))
        self.log_priors_ = np.log(priors)
        return self

    def _joint_log_likelihood(self, X):
        check_is_fitted(self, "gmms_")
        X = check_array(X)
        return np.column_stack([g.score_samples(X) for g in self.gmms_]) + self.log_priors_

    def predict_proba(self, X):
        L = self._joint_log_likelihood(X)
        L -= L.max(1, keepdims=True)
        P = np.exp(L)
        return P / P.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self._joint_log_likelihood(X).argmax(1)]
