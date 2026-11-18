"""KohonenDA v2: conformal scoring, cross-fitted calibration and a reject option.

Fixes for the weaknesses of the 2023 method found in the reproduction
(see README, "What this shows"):

* Bin lookup -> conformal p-value.  For class c with out-of-sample calibration
  distances d_1..d_n (distance of class-c objects to the BMU on map c):

      p_c(x) = (1 + #{i : d_i >= d_c(x)}) / (n + 1)

  It is never 0, is monotone (closer -> larger), needs no bins, and is on a common
  scale for all classes: if x really belongs to c, p_c(x) is (super-)uniform, so
  P(p_c(x) < alpha) <= alpha. This is class-conditional ("Mondrian") conformal
  prediction with the BMU distance as the nonconformity score.

* Leaky / wasteful calibration -> cross-fitting.  Calibration distances are
  out-of-fold: for each of `cv` folds, per-class SOMs are trained on the remaining
  folds and the held-out objects are mapped. The final maps are refit on all data.

* No way to say "none of the above" -> reject option. Default ("pooled"): a conformal
  p-value of the distance to the nearest prototype over all maps; reject if < alpha.
  Alternative ("classwise"): reject if max_c p_c(x) < alpha (empty prediction set).
  The pooled score separates unseen classes better (see README, open-set benchmark).

Two scorings are available for `predict_proba` / `predict` (both are fitted, so
`scoring` can be switched after fitting):

  "pvalue": p-values normalised across classes (a relative typicality, not a calibrated
            posterior); argmax ties are broken by the scale-free distance d_c / median_c.
  "joint" : multinomial logistic regression on the vector of log BMU distances to *all*
            maps, fitted on out-of-fold distances -> P(c | d_1, ..., d_K).

Prediction sets use the class-wise p-values; the reject option uses `reject_mode`.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .som import bmu_distances, train_class_soms

_EPS = 1e-12


class KohonenDA2(ClassifierMixin, BaseEstimator):
    """Per-class SOM discriminant analysis with conformal calibration.

    Parameters
    ----------
    map_size : int
        Side of the square map per class (3 -> 9 neurons, as in the paper).
    sigma, learning_rate, neighborhood_function, n_iterations :
        SOM hyper-parameters (MiniSom defaults, as in the v1 reproduction).
    scoring : {"pvalue", "joint"}
        How class probabilities are computed (see module docstring).
    cv : int
        Number of folds for cross-fitted calibration distances.
    reject_alpha : float or None
        If set, `predict` returns `reject_label` for objects judged novel at level alpha.
    reject_mode : {"pooled", "classwise"}
        "pooled": reject if the novelty p-value (min distance over all maps, calibrated
        on pooled out-of-fold distances) is < alpha. Ranks unseen objects better.
        "classwise": reject if max_c p_c(x) < alpha (empty prediction set).
    reject_label : object
        Label used for rejected objects.
    C : float
        Inverse regularisation strength of the logistic regression ("joint").
    trim : float in [0, 1)
        Share of each class's training objects with the largest BMU distance that is
        dropped before a second SOM fit (outlier-resistant prototypes). Calibration
        still uses all out-of-fold distances, so coverage refers to the data as given.
    random_state : int or None
    """

    def __init__(
        self,
        map_size: int = 3,
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        neighborhood_function: str = "gaussian",
        n_iterations: int = 1000,
        scoring: str = "pvalue",
        cv: int = 5,
        reject_alpha: float | None = None,
        reject_label=-1,
        reject_mode: str = "pooled",
        C: float = 1.0,
        trim: float = 0.0,
        random_state: int | None = None,
    ):
        self.map_size = map_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.neighborhood_function = neighborhood_function
        self.n_iterations = n_iterations
        self.scoring = scoring
        self.cv = cv
        self.reject_alpha = reject_alpha
        self.reject_label = reject_label
        self.reject_mode = reject_mode
        self.C = C
        self.trim = trim
        self.random_state = random_state

    def _som_kwargs(self, rng):
        return dict(
            map_size=self.map_size,
            sigma=self.sigma,
            learning_rate=self.learning_rate,
            neighborhood_function=self.neighborhood_function,
            n_iterations=self.n_iterations,
            rng=rng,
            trim=self.trim,
        )

    # ------------------------------------------------------------------ fitting
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        if len(self.classes_) < 2:
            raise ValueError("need at least two classes")
        rng = np.random.default_rng(self.random_state)
        y_idx = np.searchsorted(self.classes_, y)

        # out-of-fold BMU distances to every class map
        D_oof = np.empty((len(X), len(self.classes_)))
        skf = StratifiedKFold(self.cv, shuffle=True, random_state=int(rng.integers(2**31 - 1)))
        for tr, te in skf.split(X, y_idx):
            soms = train_class_soms(X[tr], y_idx[tr], range(len(self.classes_)),
                                    **self._som_kwargs(rng))
            D_oof[te] = bmu_distances(soms, X[te])
        self.oof_distances_ = D_oof

        # final maps on all data
        self.soms_ = train_class_soms(X, y_idx, range(len(self.classes_)),
                                      **self._som_kwargs(rng))

        # conformal calibration: own-class out-of-fold distances, sorted
        self.calib_dists_ = [np.sort(D_oof[y_idx == k, k]) for k in range(len(self.classes_))]
        self.calib_median_ = np.array([np.median(c) for c in self.calib_dists_])
        # pooled novelty calibration: min distance over all maps, out-of-fold
        self.novelty_calib_ = np.sort(D_oof.min(1))

        # joint model on log distances to all maps
        self.joint_ = LogisticRegression(C=self.C, max_iter=5000).fit(
            np.log(D_oof + _EPS), y_idx
        )
        return self

    # --------------------------------------------------------------- inference
    def bmu_distances(self, X) -> np.ndarray:
        check_is_fitted(self, "soms_")
        X = check_array(X)
        return bmu_distances(self.soms_, X)

    def _pvalues_from_d(self, D) -> np.ndarray:
        P = np.empty_like(D)
        for k, cal in enumerate(self.calib_dists_):
            n_ge = len(cal) - np.searchsorted(cal, D[:, k], side="left")
            P[:, k] = (1 + n_ge) / (len(cal) + 1)
        return P

    def pvalues(self, X) -> np.ndarray:
        """Conformal p-values (n_samples, n_classes): typicality of x for each class."""
        return self._pvalues_from_d(self.bmu_distances(X))

    def novelty_pvalues(self, X) -> np.ndarray:
        """Conformal p-value that x comes from *any* known class (n_samples,).

        Score: min_c d_c(x), the distance to the nearest prototype over all maps;
        calibrated on pooled out-of-fold scores. For objects of the known classes
        P(p < alpha) <= alpha (marginally over classes).
        """
        s = self.bmu_distances(X).min(1)
        cal = self.novelty_calib_
        n_ge = len(cal) - np.searchsorted(cal, s, side="left")
        return (1 + n_ge) / (len(cal) + 1)

    def predict_set(self, X, alpha: float = 0.05) -> np.ndarray:
        """Boolean (n_samples, n_classes): classes not rejected at level alpha.

        If x belongs to class c, c is in the set with probability >= 1 - alpha.
        An empty row means the object is atypical for every class.
        """
        return self.pvalues(X) >= alpha

    def predict_proba(self, X) -> np.ndarray:
        D = self.bmu_distances(X)
        if self.scoring == "pvalue":
            P = self._pvalues_from_d(D)
            return P / P.sum(1, keepdims=True)
        if self.scoring == "joint":
            return self.joint_.predict_proba(np.log(D + _EPS))
        raise ValueError(f"unknown scoring={self.scoring!r}")

    def predict(self, X) -> np.ndarray:
        D = self.bmu_distances(X)
        P = self._pvalues_from_d(D)
        if self.scoring == "pvalue":
            # p-values are multiples of 1/(n+1); a perturbation below that resolution
            # only breaks ties, using the scale-free distance d_c / median_c
            z = D / self.calib_median_
            res = 1.0 / (max(len(c) for c in self.calib_dists_) + 1)
            key = P - 0.5 * res * z / (1.0 + z)
            idx = key.argmax(1)
        elif self.scoring == "joint":
            idx = self.joint_.predict_proba(np.log(D + _EPS)).argmax(1)
        else:
            raise ValueError(f"unknown scoring={self.scoring!r}")
        pred = self.classes_[idx]
        if self.reject_alpha is not None:
            if self.reject_mode == "pooled":
                rejected = self.novelty_pvalues(X) < self.reject_alpha
            elif self.reject_mode == "classwise":
                rejected = P.max(1) < self.reject_alpha
            else:
                raise ValueError(f"unknown reject_mode={self.reject_mode!r}")
            if rejected.any():
                try:
                    pred = pred.astype(np.result_type(pred, np.asarray(self.reject_label)))
                except TypeError:
                    pred = pred.astype(object)
                pred[rejected] = self.reject_label
        return pred
