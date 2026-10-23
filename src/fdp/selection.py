"""Phase 6 feature selection: filter (correlation / mutual information), wrapper (SFS),
multicollinearity check (VIF), dimensionality reduction check (PCA)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def vif(frame: pd.DataFrame) -> pd.Series:
    """VIF_j = 1 / (1 - R^2_j), R^2_j from regressing feature j on all the others (computed via the
    inverse correlation matrix: VIF_j is its j-th diagonal element)."""
    corr = np.corrcoef(frame.to_numpy(dtype=float), rowvar=False)
    return pd.Series(np.diag(np.linalg.pinv(corr)), index=frame.columns)


def vif_prune(frame: pd.DataFrame, limit: float = 10.0):
    """Repeatedly drop the highest-VIF feature until every VIF <= limit."""
    kept, dropped = list(frame.columns), []
    while True:
        v = vif(frame[kept])
        if v.max() <= limit:
            return kept, dropped
        worst = v.idxmax()
        dropped.append((worst, float(v.max())))
        kept.remove(worst)


def lr_score(Xtr, ytr, Xva, yva, cols, C=1.0):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    m.fit(Xtr[cols], ytr)
    return float(average_precision_score(yva, m.predict_proba(Xva[cols])[:, 1]))


def sequential_forward_selection(Xtr, ytr, Xva, yva, candidates, max_features=15, verbose=True):
    """Wrapper method from scratch: start empty, add the feature that most improves validation PR AUC."""
    selected, remaining, path = [], list(candidates), []
    for step in range(max_features):
        scores = {f: lr_score(Xtr, ytr, Xva, yva, selected + [f]) for f in remaining}
        best = max(scores, key=scores.get)
        selected.append(best)
        remaining.remove(best)
        path.append({"step": step + 1, "added": best, "valid_pr_auc": scores[best]})
        if verbose:
            print(f"step {step + 1:2d}: + {best:<28s} valid PR AUC {scores[best]:.4f}")
    return path
