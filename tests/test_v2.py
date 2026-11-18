import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

from kohonen_da import KohonenDA2


@pytest.fixture(scope="module")
def iris():
    return load_iris(return_X_y=True)


@pytest.fixture(scope="module")
def model(iris):
    X, y = iris
    return KohonenDA2(random_state=0).fit(X, y)


def test_pvalues_never_zero_and_bounded(model, iris):
    X, _ = iris
    far = np.full((1, X.shape[1]), 100.0)
    proto = model.soms_[0].get_weights().reshape(-1, X.shape[1])[:1]
    P = model.pvalues(np.vstack([X, far, proto]))
    assert (P > 0).all() and (P <= 1).all()
    # the 2023 method gives 0 to an exact prototype; v2 gives the maximum p-value
    assert P[-1, 0] == 1.0
    # far away -> minimal p-value 1/(n+1) for every class
    n = np.array([len(c) for c in model.calib_dists_])
    assert np.allclose(P[-2], 1 / (n + 1))


def test_pvalue_monotone_in_distance(model):
    cal = model.calib_dists_[1]
    d = np.linspace(0, cal.max() * 1.5, 200)
    D = np.zeros((len(d), len(model.classes_)))
    D[:, 1] = d
    p = model._pvalues_from_d(D)[:, 1]
    assert (np.diff(p) <= 0).all()


def test_oof_distances_cover_all_samples(model, iris):
    X, y = iris
    assert model.oof_distances_.shape == (len(X), 3)
    assert np.isfinite(model.oof_distances_).all()
    assert [len(c) for c in model.calib_dists_] == [50, 50, 50]


@pytest.mark.parametrize("scoring", ["pvalue", "joint"])
def test_proba_and_accuracy(iris, scoring):
    X, y = iris
    m = KohonenDA2(scoring=scoring, random_state=0)
    acc = cross_val_score(m, X, y, cv=5).mean()
    assert acc > 0.9
    P = m.fit(X, y).predict_proba(X)
    assert np.allclose(P.sum(1), 1.0) and np.isfinite(P).all()


def test_scoring_switch_without_refit(model, iris):
    X, _ = iris
    m = clone(model).fit(*iris)
    a = m.predict_proba(X)
    m.set_params(scoring="joint")
    b = m.predict_proba(X)
    assert not np.allclose(a, b)


def test_reject_unknown_class(iris):
    X, y = iris
    known = y != 0  # train without setosa
    m = KohonenDA2(reject_alpha=0.05, random_state=0).fit(X[known], y[known])
    pred = m.predict(X[~known])
    assert (pred == -1).mean() > 0.9  # setosa is far from both maps
    assert m.predict_set(X[~known], 0.05).sum(1).max() == 0


def test_class_conditional_coverage(iris):
    # out-of-sample: P(p_y < alpha) should be about <= alpha (loose check, small data)
    X, y = iris
    rng = np.random.default_rng(1)
    miss = []
    for seed in range(10):
        idx = rng.permutation(len(X))
        tr, te = idx[:100], idx[100:]
        m = KohonenDA2(random_state=seed).fit(X[tr], y[tr])
        P = m.pvalues(X[te])
        miss.append((P[np.arange(len(te)), y[te]] < 0.1).mean())
    assert np.mean(miss) < 0.2


def test_string_labels_with_reject(iris):
    X, y = iris
    names = np.array(["setosa", "versicolor", "virginica"])[y]
    m = KohonenDA2(reject_alpha=0.05, reject_label="unknown", random_state=0).fit(X, names)
    pred = m.predict(np.vstack([X[:5], np.full((1, 4), 100.0)]))
    assert pred[-1] == "unknown" and set(pred[:5]) <= set(names)


def test_trim_zero_is_default_behaviour(iris):
    X, y = iris
    a = KohonenDA2(random_state=3).fit(X, y)
    b = KohonenDA2(random_state=3, trim=0.0).fit(X, y)
    assert np.array_equal(a.bmu_distances(X), b.bmu_distances(X))


def test_trim_resists_gross_outliers(iris):
    X, y = iris
    rng = np.random.default_rng(0)
    Xc = X.copy()
    idx = rng.choice(len(X), 30, replace=False)
    Xc[idx] += rng.normal(size=(30, 4)) * 5 * X.std(0)
    clean = np.setdiff1d(np.arange(len(X)), idx)
    errs = {}
    for trim in (0.0, 0.25):
        m = KohonenDA2(random_state=0, trim=trim).fit(Xc, y)
        # prototypes should stay close to the clean data
        errs[trim] = m.bmu_distances(X[clean])[np.arange(len(clean)), y[clean]].mean()
    assert errs[0.25] < errs[0.0]


def test_trim_validation(iris):
    with pytest.raises(ValueError):
        KohonenDA2(trim=1.0).fit(*iris)


def test_novelty_pvalue_monotone_in_min_distance(model, iris):
    X, _ = iris
    rng = np.random.default_rng(0)
    Z = np.vstack([X, X + rng.normal(scale=2.0, size=X.shape)])
    s = model.bmu_distances(Z).min(1)
    p = model.novelty_pvalues(Z)
    o = np.argsort(s)
    assert (np.diff(p[o]) <= 1e-12).all() and (p > 0).all()


@pytest.mark.parametrize("mode", ["pooled", "classwise"])
def test_reject_modes(iris, mode):
    X, y = iris
    known = y != 0
    m = KohonenDA2(reject_alpha=0.05, reject_mode=mode, random_state=0).fit(X[known], y[known])
    assert (m.predict(X[~known]) == -1).mean() > 0.9
