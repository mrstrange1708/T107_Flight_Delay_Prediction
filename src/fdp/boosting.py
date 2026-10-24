"""Phase 8 gradient boosting: from-scratch implementation, XGBoost tuning with rolling-origin
validation, and Platt calibration."""
import itertools

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingScratch:
    """Gradient boosting for binary log loss, written from scratch.

    F_0(x) = log-odds of the training base rate.
    Round m: p = sigmoid(F), negative gradient of log loss r = y - p.
             Fit a small regression tree to r (it groups rows that need similar corrections).
             Leaf value = sum(r) / sum(p(1-p))   (one Newton step for log loss in that leaf).
             F <- F + learning_rate * leaf value.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_leaf=50, seed=42):
        self.n_estimators, self.lr, self.max_depth, self.min_leaf, self.seed = n_estimators, learning_rate, max_depth, min_samples_leaf, seed

    def fit(self, X, y, X_val=None, y_val=None):
        X, y = np.asarray(X, float), np.asarray(y, float)
        p0 = y.mean()
        self.f0 = np.log(p0 / (1 - p0))
        F = np.full(len(y), self.f0)
        self.trees, self.leaf_values, self.train_loss, self.val_loss = [], [], [], []
        Fv = None if X_val is None else np.full(len(X_val), self.f0)
        for m in range(self.n_estimators):
            p = 1 / (1 + np.exp(-F))
            r = y - p
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_leaf,
                                         random_state=self.seed + m).fit(X, r)
            leaf = tree.apply(X)
            values = {}
            for l in np.unique(leaf):
                mask = leaf == l
                values[l] = r[mask].sum() / max((p[mask] * (1 - p[mask])).sum(), 1e-12)
            F += self.lr * np.vectorize(values.get)(leaf)
            self.trees.append(tree)
            self.leaf_values.append(values)
            self.train_loss.append(self._logloss(y, F))
            if Fv is not None:
                Fv += self.lr * np.vectorize(lambda l: values.get(l, 0.0))(tree.apply(np.asarray(X_val, float)))
                self.val_loss.append(self._logloss(np.asarray(y_val, float), Fv))
        return self

    @staticmethod
    def _logloss(y, F):
        p = np.clip(1 / (1 + np.exp(-F)), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def decision_function(self, X):
        X = np.asarray(X, float)
        F = np.full(len(X), self.f0)
        for tree, values in zip(self.trees, self.leaf_values):
            F += self.lr * np.vectorize(lambda l: values.get(l, 0.0))(tree.apply(X))
        return F

    def predict_proba(self, X):
        return 1 / (1 + np.exp(-self.decision_function(X)))


# ---------------------------------------------------------------- XGBoost
def as_xgb_frame(X: pd.DataFrame, numeric, categorical, categories: dict):
    """Numeric columns as float (NaN allowed: XGBoost learns where missing values go) and
    categoricals as pandas 'category' with a FIXED category list, so codes match across splits."""
    out = X[list(numeric)].astype(float).copy()
    for c in categorical:
        out[c] = pd.Categorical(X[c].astype(str), categories=categories[c])
    return out


def make_xgb(params: dict, n_estimators: int, seed: int = 42, early_stopping: int | None = None):
    kw = dict(objective="binary:logistic", tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
              n_estimators=n_estimators, random_state=seed, n_jobs=-1, eval_metric="aucpr", **params)
    if early_stopping:
        kw["early_stopping_rounds"] = early_stopping
    return xgb.XGBClassifier(**kw)


def rolling_eval(Xf: pd.DataFrame, y: np.ndarray, month: np.ndarray, params: dict, months=(7, 8, 9),
                 margin_fn=None, early_stopping: int = 100, max_trees: int = 3000, seed: int = 42):
    """Honest rolling-origin score for one configuration.

    For each test month m:
      1. train on months < m-1 and early-stop on month m-1 (picks the number of trees),
      2. refit on ALL months < m with that number of trees,
      3. score month m, which played no part in steps 1-2.
    (Early-stopping on month m and then scoring month m would be optimistic: the stopping rule
    has already seen the answers.)

    margin_fn(train_idx, [idx, ...]) -> [logit arrays]: optional starting prediction for each row
    (e.g. logistic regression's logit). Boosting then learns only corrections on top of it.
    """
    scores, trees = [], []
    for m in months:
        inner, stop = np.where(month < m - 1)[0], np.where(month == m - 1)[0]
        full, test = np.where(month < m)[0], np.where(month == m)[0]
        bi = bs = bf = bt = None
        if margin_fn:
            bi, bs = margin_fn(inner, [inner, stop])
        model = make_xgb(params, max_trees, seed, early_stopping)
        model.fit(Xf.iloc[inner], y[inner], base_margin=bi, eval_set=[(Xf.iloc[stop], y[stop])],
                  base_margin_eval_set=None if bs is None else [bs], verbose=False)
        n = model.best_iteration + 1
        if margin_fn:
            bf, bt = margin_fn(full, [full, test])
        final = make_xgb(params, n, seed).fit(Xf.iloc[full], y[full], base_margin=bf)
        scores.append(float(average_precision_score(y[test], final.predict_proba(Xf.iloc[test], base_margin=bt)[:, 1])))
        trees.append(int(n))
    return scores, trees


def tune_xgb(Xf, y, month, grid: dict, n_iter: int, months=(7, 8, 9), margin_fn=None, seed: int = 42, verbose=True):
    """Random search over `grid`; each configuration is scored with rolling_eval (mean PR AUC)."""
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    picks = np.random.default_rng(seed).choice(len(combos), size=min(n_iter, len(combos)), replace=False)
    rows = []
    for i, ci in enumerate(picks):
        params = dict(zip(keys, combos[ci]))
        scores, trees = rolling_eval(Xf, y, month, params, months, margin_fn, seed=seed)
        rows.append({**params, "mean_pr_auc": float(np.mean(scores)), "fold_pr_auc": [round(v, 4) for v in scores], "trees_per_fold": trees})
        if verbose:
            print(f"[{i + 1:2d}/{len(picks)}] {params} -> folds {rows[-1]['fold_pr_auc']} mean {rows[-1]['mean_pr_auc']:.4f} trees {trees}", flush=True)
    return pd.DataFrame(rows).sort_values("mean_pr_auc", ascending=False).reset_index(drop=True)


class Platt:
    """Platt scaling: fit p_cal = sigmoid(a * logit(p) + b) on a held-out month."""

    def fit(self, p, y):
        self.m = LogisticRegression(C=1e6).fit(self._logit(p)[:, None], y)
        return self

    def transform(self, p):
        return self.m.predict_proba(self._logit(p)[:, None])[:, 1]

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
