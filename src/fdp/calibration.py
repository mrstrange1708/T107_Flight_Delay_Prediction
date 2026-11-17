"""Phase 2 · calibration: turning scores into probabilities a planner can trust.
Owner: Nakkina Lakshmi Praneeth (Praneeth).

Guo et al. (2017) showed that modern neural networks are often OVER-confident: their probabilities
are more extreme than the observed frequencies. Two fixes, both fitted on October (a month no model
is trained on) and both leave the ranking - and so PR AUC and ROC AUC - unchanged:
  temperature scaling  p = sigmoid(logit / T)           one number
  Platt scaling        p = sigmoid(a * logit + b)       two numbers (Phase 1 used this)
Platt's extra intercept b also shifts the overall level, which helps when the model is biased,
but it also copies October's base rate into the probabilities.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression


class TemperatureScaler:
    """Guo et al. (2017): divide the logit by one number T, fitted on a held-out month by
    minimising log loss. T > 1 softens over-confident probabilities; ranking (PR AUC) is unchanged."""

    def fit(self, logits, y):
        logits, y = np.asarray(logits, float), np.asarray(y, float)

        def nll(t):
            p = np.clip(1 / (1 + np.exp(-logits / t)), 1e-7, 1 - 1e-7)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        self.T = float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)
        return self

    def transform(self, logits):
        return 1 / (1 + np.exp(-np.asarray(logits, float) / self.T))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))




class PlattScaler:
    def fit(self, logits, y):
        self.m = LogisticRegression(C=1e6).fit(np.asarray(logits, float)[:, None], y)
        return self

    def transform(self, logits):
        return self.m.predict_proba(np.asarray(logits, float)[:, None])[:, 1]


def reliability(y, p, n_bins=10, min_count=100):
    """Points for a reliability diagram: mean predicted vs observed rate per probability bin."""
    y, p = np.asarray(y), np.asarray(p)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    keep = [i for i in range(n_bins) if (b == i).sum() >= min_count]
    return [float(p[b == i].mean()) for i in keep], [float(y[b == i].mean()) for i in keep]
