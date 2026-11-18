import numpy as np
import pytest
from sklearn.model_selection import cross_val_score

from kohonen_da.baselines import GMMClassifier
from kohonen_da.datasets import ALL, SYNTHETIC


@pytest.mark.parametrize("name", list(ALL))
def test_dataset_shapes(name):
    X, y = ALL[name](0)
    assert X.ndim == 2 and len(X) == len(y)
    assert np.isfinite(X).all()
    assert len(np.unique(y)) >= 2


@pytest.mark.parametrize("name", list(SYNTHETIC))
def test_synthetic_reproducible_and_resampled(name):
    a, _ = SYNTHETIC[name](1)
    b, _ = SYNTHETIC[name](1)
    c, _ = SYNTHETIC[name](2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_gmm_classifier():
    X, y = ALL["multimodal"](0)
    m = GMMClassifier(3, random_state=0)
    assert cross_val_score(m, X, y, cv=3).mean() > 0.85
    P = m.fit(X, y).predict_proba(X)
    assert np.allclose(P.sum(1), 1)
