"""Phase 4 preprocessing: missing values, scaling, encoding.

Rule from the course (Phase 5 warning): every statistic used for imputation or scaling is FIT ON
THE TRAINING ROWS ONLY and then applied to validation/test. Wrapping the steps in a scikit-learn
Pipeline guarantees that, because .fit() only ever sees the rows it is given.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler


def linear_preprocessor(numeric, categorical, scaler: str = "standard"):
    """For gradient-based / linear models (logistic regression).

    numeric:     median imputation + missing-indicator column, then scaling (z-score or robust).
    categorical: one-hot; categories seen < 50 times in training are grouped as 'infrequent',
                 and unseen categories at test time map to that group instead of crashing.
    """
    sc = StandardScaler() if scaler == "standard" else RobustScaler()
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), sc), list(numeric)),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), list(categorical)),
    ])


def tree_preprocessor(numeric, categorical):
    """For tree models (decision tree, random forest).

    Trees split on thresholds, so scaling changes nothing and is skipped. Label/ordinal codes are
    safe for trees (no distances are computed). Unseen categories get code -1.
    """
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), list(numeric)),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), list(categorical)),
    ])


def iqr_outlier_share(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    q1, q3 = np.percentile(x, [25, 75])
    return float(((x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))).mean())
