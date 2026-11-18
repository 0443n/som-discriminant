import numpy as np
import pytest
from sklearn.datasets import load_iris

from kohonen_da import KohonenDA, sturges_bins


@pytest.mark.parametrize("n,expected", [(1, 1), (2, 2), (15, 4), (16, 5), (34, 6), (48, 6)])
def test_sturges(n, expected):
    assert sturges_bins(n) == expected


@pytest.fixture(scope="module")
def iris_model():
    X, y = load_iris(return_X_y=True)
    return KohonenDA(random_state=0).fit(X, y), X, y


def test_histograms_normalised(iris_model):
    m, X, y = iris_model
    for k, f in enumerate(m.hist_freqs_):
        assert np.isclose(f.sum(), 1.0)
        assert len(f) == sturges_bins((y == k).sum())


def test_bmu_distance_matches_minisom(iris_model):
    m, X, _ = iris_model
    D = m.bmu_distances(X[:10])
    for k, som in enumerate(m.soms_):
        ref = np.linalg.norm(X[:10] - som.quantization(X[:10]), axis=1)
        assert np.allclose(D[:, k], ref)


def test_scores_zero_outside_histogram(iris_model):
    m, X, _ = iris_model
    # exactly on a prototype of map 0 -> distance 0, below setosa's first bin edge
    w = m.soms_[0].get_weights().reshape(-1, X.shape[1])[0]
    assert m.hist_edges_[0][0] > 0
    assert m.histogram_scores(w[None])[0, 0] == 0.0
    # far away -> zero everywhere -> undefined probability
    far = np.full((1, X.shape[1]), 100.0)
    assert np.isnan(m.predict_proba(far)).all()
    assert m.n_undecided(far) == 1


def test_proba_rows_sum_to_one(iris_model):
    m, X, _ = iris_model
    P = m.predict_proba(X)
    ok = ~np.isnan(P).any(1)
    assert np.allclose(P[ok].sum(1), 1.0)


def test_reasonable_accuracy(iris_model):
    m, X, y = iris_model
    assert (m.predict(X) == y).mean() > 0.85
