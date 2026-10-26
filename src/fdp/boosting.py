"""Phase 8 gradient boosting with XGBoost: input format, honest rolling-month scoring and
Platt calibration. The matching %%writefile cell in notebook 05 is the source of truth."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


def as_xgb_frame(X, numeric, categorical, categories):
    """Numeric columns as float (NaN allowed: XGBoost learns where missing values go) and
    categoricals as pandas 'category' with a FIXED category list, so codes match across splits."""
    out = X[list(numeric)].astype(float).copy()
    for c in categorical:
        out[c] = pd.Categorical(X[c].astype(str), categories=categories[c])
    return out


def new_xgb(params, n_trees, early_stopping_rounds=None, seed=42):
    """An XGBClassifier with our fixed settings plus the hyper-parameters in `params`."""
    return xgb.XGBClassifier(n_estimators=n_trees, early_stopping_rounds=early_stopping_rounds,
                             objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
                             enable_categorical=True, max_cat_to_onehot=1,
                             random_state=seed, n_jobs=-1, **params)


def rolling_eval(Xf, y, month, params, months=(7, 8, 9), lr_logits=None, seed=42):
    """Honest rolling-month PR AUC for one XGBoost configuration.

    For each scored month m:
      1. train on months before m-1 and early-stop on month m-1 (this picks the number of trees),
      2. refit on ALL months before m with that number of trees,
      3. score month m, which played no part in steps 1-2.
    (Early-stopping on month m and then scoring month m would be optimistic: the stopping rule
    would already have seen the answers.)

    lr_logits: optional {k: logistic-regression logits for every row, from a model fitted on
    months < k}. If given, XGBoost starts each row at that logit (base_margin) and the trees
    only learn corrections on top of logistic regression (the hybrid model).
    """
    scores, trees = [], []
    for m in months:
        inner = month < m - 1    # training rows for step 1
        stop = month == m - 1    # early-stopping month
        full = month < m         # training rows for step 2
        test = month == m        # scored month

        if lr_logits is None:
            start_inner = start_stop = start_full = start_test = None
        else:
            start_inner, start_stop = lr_logits[m - 1][inner], lr_logits[m - 1][stop]
            start_full, start_test = lr_logits[m][full], lr_logits[m][test]

        # 1. early stopping on month m-1 picks the number of trees
        model = new_xgb(params, 3000, early_stopping_rounds=100, seed=seed)
        model.fit(Xf[inner], y[inner], base_margin=start_inner,
                  eval_set=[(Xf[stop], y[stop])], base_margin_eval_set=[start_stop], verbose=False)
        n_trees = model.best_iteration + 1

        # 2. refit on all earlier months with that many trees, 3. score month m
        final = new_xgb(params, n_trees, seed=seed).fit(Xf[full], y[full], base_margin=start_full)
        p = final.predict_proba(Xf[test], base_margin=start_test)[:, 1]
        scores.append(float(average_precision_score(y[test], p)))
        trees.append(int(n_trees))
    return scores, trees


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(p, y):
    """Platt scaling: a one-feature logistic regression on logit(p), fitted on a held-out month."""
    return LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y)


def apply_platt(platt, p):
    return platt.predict_proba(logit(p).reshape(-1, 1))[:, 1]
