"""Phase 7: logistic regression from scratch for the baseline experiments.

Training loop: initialise -> forward pass -> loss -> gradient step -> convergence.
The matching %%writefile cell in notebook 04 is the source of truth.
"""
import numpy as np


def sigmoid(z):
    """Numerically stable sigmoid, including for large positive/negative logits."""
    z = np.asarray(z, dtype=float)
    exp_neg_abs = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + exp_neg_abs), exp_neg_abs / (1 + exp_neg_abs))


class LogisticGD:
    """Binary logistic regression with batch, mini-batch or stochastic GD.

    batch_size=None uses all rows per step; 1 uses SGD; k uses mini-batches.
    The objective is mean Bernoulli negative log-likelihood plus
    (l2/2)*||w||^2. The intercept is not penalised. Inputs must be numeric,
    imputed and preferably scaled using statistics from training rows only.
    predict_proba returns a one-dimensional array of positive-class scores.
    """

    def __init__(self, lr=0.5, epochs=200, batch_size=None, l2=0.0, tol=1e-6, seed=42):
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.l2, self.tol, self.seed = l2, tol, seed

    @staticmethod
    def _add_bias(X):
        return np.column_stack([np.ones(len(X)), X])

    def loss(self, A, y):
        """Mean penalised log loss; A already includes the leading bias column."""
        logits = A @ self.w
        # log(1 + exp(z)) - y*z avoids log(0) and exponential overflow.
        ll = np.mean(np.logaddexp(0, logits) - y * logits)
        return float(ll + 0.5 * self.l2 * np.sum(self.w[1:] ** 2))

    def _grad(self, A, y):
        grad = A.T @ (sigmoid(A @ self.w) - y) / len(y)
        grad[1:] += self.l2 * self.w[1:]
        return grad

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
            raise ValueError("X must be a nonempty 2D array and y a matching 1D array.")
        if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("X must be finite and y must contain only 0 or 1.")
        if not np.isfinite([self.lr, self.l2, self.tol]).all() or self.lr <= 0 or self.l2 < 0 or self.tol < 0:
            raise ValueError("lr must be positive; l2 and tol must be nonnegative and finite.")
        if not isinstance(self.epochs, (int, np.integer)) or self.epochs < 1:
            raise ValueError("epochs must be a positive integer.")
        if self.batch_size is not None and (
            not isinstance(self.batch_size, (int, np.integer)) or self.batch_size < 1
        ):
            raise ValueError("batch_size must be None or a positive integer.")

        rng = np.random.default_rng(self.seed)
        A = self._add_bias(X)
        self.n_features_in_ = X.shape[1]
        self.w = np.zeros(A.shape[1])                               # 1. initialise
        self.history = [self.loss(A, y)]
        for _ in range(self.epochs):
            if self.batch_size is None:
                self.w -= self.lr * self._grad(A, y)               # 2-4. forward, gradient, step
            else:
                order = rng.permutation(len(y))
                for i in range(0, len(y), self.batch_size):
                    b = order[i:i + self.batch_size]
                    self.w -= self.lr * self._grad(A[b], y[b])
            self.history.append(self.loss(A, y))
            if abs(self.history[-2] - self.history[-1]) < self.tol: # 5. convergence check
                break
        self.n_iter_ = len(self.history) - 1
        return self

    def predict_proba(self, X):
        if not hasattr(self, "w"):
            raise ValueError("Call fit before predict_proba.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("X must be finite and have the same feature count as training.")
        return sigmoid(self._add_bias(X) @ self.w)
