"""Phase 7 learned baseline: scikit-learn logistic regression on the shared linear preprocessing.

Notebooks 04 and 05 both build the model here, so they always use the same one.
The matching %%writefile cell in notebook 04 is the source of truth.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .preprocess import linear_preprocessor


def logistic_regression(numeric, categorical, C=1.0):
    """Median imputation, scaling and one-hot encoding, then L2-regularised logistic regression.
    C is the inverse penalty strength: smaller C = stronger regularisation."""
    return make_pipeline(linear_preprocessor(numeric, categorical), LogisticRegression(C=C, max_iter=3000))
