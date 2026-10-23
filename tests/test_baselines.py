"""Check the scratch model's mathematics independently of the flight dataset."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fdp.baselines import LogisticGD, sigmoid


def sample():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(600, 4))
    y = rng.binomial(1, sigmoid(X @ np.array([0.8, -1.2, 0.3, 0.5]) - 0.4))
    return X, y


def test_gradient_matches_finite_differences_with_unpenalised_intercept():
    X, y = sample()
    model = LogisticGD(l2=0.2)
    A = model._add_bias(X)
    model.w = np.array([0.7, 0.3, -0.2, 0.4, 0.1])
    analytic = model._grad(A, y)
    numeric = np.zeros_like(model.w)
    for j in range(len(model.w)):
        original = model.w[j]
        model.w[j] = original + 1e-6
        plus = model.loss(A, y)
        model.w[j] = original - 1e-6
        minus = model.loss(A, y)
        model.w[j] = original
        numeric[j] = (plus - minus) / 2e-6
    np.testing.assert_allclose(analytic, numeric, atol=1e-8)
    assert analytic[0] == pytest.approx(np.mean(sigmoid(A @ model.w) - y))


def test_batch_solution_matches_sklearn():
    X, y = sample()
    model = LogisticGD(lr=0.5, epochs=3000, tol=1e-12).fit(X, y)
    reference = LogisticRegression(C=1e10, max_iter=3000, tol=1e-10).fit(X, y)
    np.testing.assert_allclose(model.predict_proba(X), reference.predict_proba(X)[:, 1], atol=1e-4)
    assert np.all(np.diff(model.history) <= 1e-12)


@pytest.mark.parametrize("batch_size", [None, 32, 1])
def test_training_modes_learn_and_are_reproducible(batch_size):
    X, y = sample()
    params = dict(lr=0.05, epochs=40, batch_size=batch_size, seed=7)
    first = LogisticGD(**params).fit(X, y)
    second = LogisticGD(**params).fit(X, y)
    assert first.history[-1] < first.history[0] - 0.05
    np.testing.assert_array_equal(first.w, second.w)


def test_extreme_logits_have_stable_probabilities_and_loss():
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        np.testing.assert_allclose(sigmoid([-1000, 0, 1000]), [0, 0.5, 1])
        model = LogisticGD()
        model.w = np.array([0.0, 1.0])
        assert model.loss(np.array([[1.0, -1000], [1.0, 1000]]), np.array([1, 0])) == 1000
