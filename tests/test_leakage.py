"""Leakage guard: features for day D must not change if outcomes on/after D change.
Run from the repo root:  python -m pytest -q tests"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fdp import data, features  # noqa: E402
from fdp.config import POST_DEPARTURE_COLUMNS  # noqa: E402

CUTOFF = pd.Timestamp("2013-06-15")


def _small():
    f, p = data.load_raw()
    return data.build_dataset(f[f["month"].between(5, 7)], p)


def test_features_ignore_same_day_and_future_outcomes():
    df = _small()
    X1, g = features.build_features(df)
    scrambled = df.copy()
    future = scrambled["date"] >= CUTOFF
    rng = np.random.default_rng(0)
    for col in POST_DEPARTURE_COLUMNS:
        scrambled.loc[future, col] = rng.permutation(scrambled.loc[future, col].to_numpy())
    scrambled.loc[future, "delayed"] = 1.0 - scrambled.loc[future, "delayed"]
    X2, _ = features.build_features(scrambled)
    cols = features.numeric_columns(g)
    day = X1["date"] == CUTOFF
    pd.testing.assert_frame_equal(X1.loc[day, cols], X2.loc[day, cols])


def test_no_post_departure_column_is_a_feature():
    _, g = features.build_features(_small())
    assert not set(features.numeric_columns(g)) & set(POST_DEPARTURE_COLUMNS)
