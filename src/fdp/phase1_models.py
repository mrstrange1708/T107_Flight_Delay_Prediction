"""Phase 2 · Phase 1's three final models, refitted with the settings we reported.
Owner: Rishik Chowdary Karuturi (Rishik).

Phase 2 must compare against EXACTLY what Phase 1 reported, on the same rows, in the same run:
  logistic regression  C = 0.001
  XGBoost              tuned settings, 40 trees
  hybrid               XGBoost starting from LR's logit (base_margin), 16 trees
(the settings live in config.py, copied from the Phase 1 report).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from . import config
from .boosting import as_xgb_frame, make_xgb
from .preprocess import linear_preprocessor


class Phase1Models:
    def __init__(self, numeric, categorical, categories: dict):
        self.num, self.cat, self.cats = list(numeric), list(categorical), categories

    def fit(self, frame: pd.DataFrame):
        y = frame["delayed"].astype(int).to_numpy()
        self.lr = make_pipeline(linear_preprocessor(self.num, self.cat),
                                LogisticRegression(C=config.PHASE1_LR_C, max_iter=3000)).fit(frame[self.num + self.cat], y)
        F = as_xgb_frame(frame, self.num, self.cat, self.cats)
        self.xgb = make_xgb(config.PHASE1_XGB, config.PHASE1_XGB_TREES, config.SEED).fit(F, y)
        self.hyb = make_xgb(config.PHASE1_XGB, config.PHASE1_HYBRID_TREES, config.SEED).fit(
            F, y, base_margin=self.lr.decision_function(frame[self.num + self.cat]))
        return self

    def logits(self, frame: pd.DataFrame) -> dict:
        """Scores on the logit scale (what calibration expects)."""
        F = as_xgb_frame(frame, self.num, self.cat, self.cats)
        lr_logit = self.lr.decision_function(frame[self.num + self.cat])
        return {"Logistic regression": lr_logit,
                "XGBoost (Phase 1)": self.xgb.predict(F, output_margin=True),
                "Hybrid: XGBoost from LR (Phase 1)": self.hyb.predict(F, base_margin=lr_logit, output_margin=True)}

    def lr_logit(self, frame: pd.DataFrame) -> np.ndarray:
        return self.lr.decision_function(frame[self.num + self.cat])
