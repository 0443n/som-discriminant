import numpy as np
import pytest
from minisom import MiniSom
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from kohonen_da.datasets import ALL
from kohonen_da.v3 import KohonenDA3, NeuronMixture, sorted_log_dists

SCORINGS = ["density", "joint_density", "joint_sorted", "joint_min"]


@pytest.fixture(scope="module")
def iris():
    X, y = load_iris(return_X_y=True)
    return StandardScaler().fit_transform(X), y


@pytest.fixture(scope="module")
def model(iris):
    return KohonenDA3(random_state=0).fit(*iris)


def _single_neuron_som(w):
    som = MiniSom(1, 1, len(w), random_seed=0)
    som._weights[0, 0] = w
    return som


def test_mixture_matches_gaussian_for_one_neuron():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2)) * [1.0, 3.0]
    som = _single_neuron_som(np.zeros(2))
    nm = NeuronMixture([som], X, np.zeros(500, int), covariance="diag",
                       shrinkage=0.0, reg_var=0.0)
    var = (X ** 2).mean(0)
    x = np.array([[0.5, -1.0]])
    expected = -0.5 * (np.log(2 * np.pi * var).sum() + (x[0] ** 2 / var).sum())
    assert np.isclose(nm.log_density(x)[0, 0], expected)


def test_shrinkage_pulls_to_pooled_variance():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    som = _single_neuron_som(np.zeros(3))
    y = np.zeros(50, int)
    a = NeuronMixture([som], X, y, covariance="spherical", shrinkage=0.0, reg_var=0.0)
    b = NeuronMixture([som], X, y, covariance="spherical", shrinkage=1e6, reg_var=0.0)
    # single neuron: neuron variance == pooled variance, shrinkage is a no-op
    assert np.allclose(a.var[0], b.var[0])


def test_sorted_features_are_sorted(model, iris):
    X, _ = iris
    S = sorted_log_dists(model.soms_, X[:20])
    J = model.map_size ** 2
    assert S.shape == (20, 3 * J)
    for k in range(3):
        assert (np.diff(S[:, k * J:(k + 1) * J], axis=1) >= 0).all()


@pytest.mark.parametrize("scoring", SCORINGS)
@pytest.mark.parametrize("cov", ["spherical", "diag"])
def test_all_scorings_valid(model, iris, scoring, cov):
    X, y = iris
    model.set_params(scoring=scoring, covariance=cov)
    P = model.predict_proba(X)
    assert np.isfinite(P).all() and np.allclose(P.sum(1), 1)
    assert (model.predict(X) == y).mean() > 0.85


def test_density_scoring_solves_multimodal():
    # the case where nearest-prototype v2 lags far behind a per-class GMM
    X, y = ALL["multimodal"](0)
    X = StandardScaler().fit_transform(X)
    idx = np.random.default_rng(0).permutation(len(X))
    tr, te = idx[:200], idx[200:]
    m = KohonenDA3(scoring="density", covariance="diag", random_state=0).fit(X[tr], y[tr])
    assert (m.predict(X[te]) == y[te]).mean() > 0.9


def test_pvalues_and_novelty(model, iris):
    X, _ = iris
    model.set_params(covariance="diag")
    far = np.full((1, X.shape[1]), 50.0)
    P = model.pvalues(np.vstack([X, far]))
    assert (P > 0).all() and (P <= 1).all()
    n = np.array([len(c) for c in model.class_calib_["diag"]])
    assert np.allclose(P[-1], 1 / (n + 1))
    pn = model.novelty_pvalues(np.vstack([X, far]))
    assert pn[-1] == 1 / (len(model.novelty_calib_["diag"]) + 1)


def test_reject_unknown_class(iris):
    X, y = iris
    known = y != 0
    m = KohonenDA3(reject_alpha=0.05, random_state=0).fit(X[known], y[known])
    assert (m.predict(X[~known]) == -1).mean() > 0.9
    assert (m.predict(X[known]) == -1).mean() < 0.15


def test_trim_keeps_variances_clean(iris):
    X, y = iris
    rng = np.random.default_rng(0)
    Xc = X.copy()
    idx = rng.choice(len(X), 30, replace=False)
    Xc[idx] += rng.normal(size=(30, X.shape[1])) * 5
    v = {}
    for trim in (0.0, 0.25):
        m = KohonenDA3(trim=trim, random_state=0).fit(Xc, y)
        v[trim] = np.median([np.median(a) for a in m.mixtures_["diag"].var])
    clean = KohonenDA3(random_state=0).fit(X, y)
    v_clean = np.median([np.median(a) for a in clean.mixtures_["diag"].var])
    assert abs(np.log(v[0.25] / v_clean)) < abs(np.log(v[0.0] / v_clean))


def test_auto_trim_drops_gross_outliers_only():
    from kohonen_da.som import auto_keep
    rng = np.random.default_rng(0)
    clean = rng.normal(size=(100, 4))
    keep = auto_keep(clean)
    assert keep is None or len(keep) >= 90
    dirty = np.vstack([clean, rng.normal(size=(20, 4)) * 10 + 30])
    keep = auto_keep(dirty)
    assert keep is not None and set(range(100, 120)).isdisjoint(keep)


def test_auto_selection(iris):
    from kohonen_da.v3 import KohonenDA3Auto
    X, y = iris
    m = KohonenDA3Auto(map_sizes=(2, 3), random_state=0).fit(X, y)
    assert len(m.selection_) == 2 * 3
    assert m.selection_[(m.map_size_, m.scoring_)] == max(m.selection_.values())
    P = m.predict_proba(X)
    assert np.allclose(P.sum(1), 1) and (m.predict(X) == y).mean() > 0.9
    assert m.novelty_pvalues(X).shape == (len(X),)


def test_fast_selection_matches_full_when_all_sizes_stacked(iris):
    from kohonen_da.v3 import KohonenDA3Auto
    X, y = iris
    full = KohonenDA3Auto(map_sizes=(2, 3), random_state=0).fit(X, y)
    fast = KohonenDA3Auto(map_sizes=(2, 3), selection="fast", n_stack_sizes=2,
                          random_state=0).fit(X, y)
    assert (full.map_size_, full.scoring_) == (fast.map_size_, fast.scoring_)
    assert full.selection_ == fast.selection_
    fast1 = KohonenDA3Auto(map_sizes=(2, 3), selection="fast", random_state=0).fit(X, y)
    assert len(fast1.selection_) == 2 + 2  # density at both sizes + 2 stacked at one


def test_sorted_top_truncates_features(iris):
    X, y = iris
    m = KohonenDA3(map_size=3, scoring="joint_sorted", sorted_top=2, random_state=0).fit(X, y)
    F = m._stack_features("joint_sorted", S=m.oof_sorted_)
    assert F.shape[1] == 3 * 2
    assert np.allclose(m.predict_proba(X).sum(1), 1)


def test_sorted_top_auto_rule(iris):
    X, y = iris
    small = KohonenDA3(map_size=3, sorted_top="auto", random_state=0).fit(X, y)  # 3*9 <= 100
    assert small._stack_features("joint_sorted", S=small.oof_sorted_).shape[1] == 27
    big = KohonenDA3(map_size=6, sorted_top="auto", random_state=0).fit(X, y)    # 3*36 > 100
    assert big._stack_features("joint_sorted", S=big.oof_sorted_).shape[1] == 3 * 5
