"""Project-wide settings: paths, label definition, decision time, chronological splits.
Owner: Rishik (Phase 1 problem definition + Phase 5 data splitting)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

SEED = 42

# Phase 1 - the target. A flight is "delayed" if it arrives 15+ minutes late (US BTS definition).
DELAY_THRESHOLD_MIN = 15

# Decision time: 05:00 on day D, before the first departure. Allowed inputs: the published
# schedule for day D, static aircraft data, and outcomes of flights on days before D.
# Columns that only exist after a flight departs are NEVER inputs:
POST_DEPARTURE_COLUMNS = ["dep_time", "dep_delay", "arr_time", "arr_delay", "air_time"]

# Phase 5 - chronological split (the "golden barrier"). Jan 1-14 is a warm-up period that only
# feeds the history features, so it is not used for training.
WARMUP_END = pd.Timestamp("2013-01-14")
SPLITS = {
    "train": (pd.Timestamp("2013-01-15"), pd.Timestamp("2013-08-31")),
    "valid": (pd.Timestamp("2013-09-01"), pd.Timestamp("2013-09-30")),   # hyper-parameter tuning
    "calib": (pd.Timestamp("2013-10-01"), pd.Timestamp("2013-10-31")),   # calibration / threshold
    "test":  (pd.Timestamp("2013-11-01"), pd.Timestamp("2013-12-31")),   # touched once, at the end
}
# Rolling-origin folds for tuning (train on everything before the month, validate on that month).
CV_MONTHS = [7, 8, 9]

CATEGORICAL = ["carrier", "origin", "dest"]
