"""Phase 2 · everything a network reads at 05:00.
Owner: Balanagu Sesha Srikar (Srikar).

Part 1 - tabular inputs: TabularEncoder turns the 41 Phase 1 features and the 3 categories into
arrays (numeric block, category indices for the embedding tables).

Part 2 - ordered delay histories for the temporal model:

For a flight on day D the temporal model reads the LAST L DAYS (D-L ... D-1) of delay outcomes
for the groups the flight belongs to (all flights, its origin, carrier, carrier at origin, route,
flight number, aircraft). Phase 1 squeezed these days into a few hand-made averages (1/7/28-day
rates); here the network sees the whole ordered sequence and learns how to weigh the days.

Leakage rule (same as Phase 1): the window ends on day D-1. Day D's own outcomes are never read.
"""
import numpy as np
import pandas as pd

MISSING_FLAG_COLS = ["plane_age", "plane_seats", "plane_engines"]


class TabularEncoder:
    """Fit on training rows only: medians, means, stds and category vocabularies.

    Categories seen fewer than `min_count` times in training share index 1 ('rare');
    index 0 is reserved for categories never seen in training ('unknown')."""

    def __init__(self, numeric, categorical, min_count: int = 50):
        self.numeric, self.categorical, self.min_count = list(numeric), list(categorical), min_count

    def fit(self, X: pd.DataFrame):
        num = X[self.numeric].astype(float)
        self.median = num.median()
        filled = num.fillna(self.median)
        self.mean, self.std = filled.mean(), filled.std().replace(0, 1.0)
        self.vocab, self.seen = {}, {}
        for c in self.categorical:
            counts = X[c].astype(str).value_counts()
            self.vocab[c] = {v: i + 2 for i, v in enumerate(sorted(counts[counts >= self.min_count].index))}
            self.seen[c] = set(counts.index)
        return self

    def cardinalities(self):
        return [len(self.vocab[c]) + 2 for c in self.categorical]

    def transform(self, X: pd.DataFrame):
        num = X[self.numeric].astype(float)
        flags = num[MISSING_FLAG_COLS].isna().any(axis=1).astype(float).to_numpy()[:, None]
        z = ((num.fillna(self.median) - self.mean) / self.std).to_numpy()
        x_num = np.hstack([z, flags]).astype(np.float32)
        cols = []
        for c in self.categorical:
            s = X[c].astype(str)
            code = s.map(self.vocab[c]).fillna(0).astype(int).to_numpy()
            rare = (code == 0) & s.isin(self.seen[c]).to_numpy()
            cols.append(np.where(rare, 1, code))       # seen but rare -> 1; never seen -> 0
        return x_num, np.stack(cols, axis=1).astype(np.int64)


def emb_dim(n: int) -> int:
    """Rule of thumb for embedding size: grows slowly with the number of categories."""
    return int(min(16, round(1.6 * n ** 0.56)))



# ----------------------------------------------------------------- ordered histories

# (name, grouping columns). "all" is every flight from NYC that day.
SEQ_GROUPS = [
    ("origin", ["origin"]),
    ("carrier", ["carrier"]),
    ("carrier_origin", ["carrier", "origin"]),
    ("route", ["origin", "dest"]),
    ("flight", ["carrier", "flight"]),
    ("tail", ["tailnum"]),
]
SEQ_LEN = 14       # two weeks: the ACF said yesterday matters most, but small groups need more days
SEQ_ALPHA = 5.0    # daily counts are small, so shrink each day's rate towards that day's overall rate


def _key(df, cols):
    k = df[cols[0]].astype(str)
    for c in cols[1:]:
        k = k + "|" + df[c].astype(str)
    return k


def channel_names():
    names = ["all_rate", "all_logn"]
    for name, _ in SEQ_GROUPS:
        names += [f"{name}_rel", f"{name}_logn"]
    return names + ["dow_sin", "dow_cos"]


def build_sequences(df: pd.DataFrame, rows: pd.DataFrame, seq_len: int = SEQ_LEN, alpha: float = SEQ_ALPHA,
                    relative: bool = True):
    """Return an array (len(rows), seq_len, n_channels), float32.

    df   : every scheduled flight (labelled or not) - the source of past outcomes.
    rows : the flights we need sequences for (any subset of df, any order).

    Channels per past day t:
      all_rate        overall NYC late rate that day
      all_logn        log(1 + labelled flights that day)
      <group>_rel     group's smoothed late rate MINUS the overall rate that day
      <group>_logn    log(1 + the group's labelled flights that day)  (0 = no evidence)
      dow_sin/cos     weekday of that past day
    "_rel" is the Phase 1 lesson: absolute rates drift with the season, a rate measured against
    the same day's overall rate means the same thing in July and in November.
    relative=False keeps the absolute rates instead (used only for the ablation in notebook 09).
    """
    dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    d_index = {d: i for i, d in enumerate(dates)}
    obs = df[df["delayed"].notna()]
    obs_day = obs["date"].map(d_index).to_numpy()
    T = len(dates)

    k_all = np.bincount(obs_day, weights=obs["delayed"].to_numpy(), minlength=T)
    n_all = np.bincount(obs_day, minlength=T).astype(float)
    p_all = np.where(n_all > 0, k_all / np.maximum(n_all, 1), np.nan)
    p_all = pd.Series(p_all).ffill().fillna(0.2).to_numpy()   # a day with no labelled flight borrows the last one

    row_day = rows["date"].map(d_index).to_numpy()
    lag = row_day[:, None] - seq_len + np.arange(seq_len)[None, :]    # days D-L ... D-1
    valid = lag >= 0
    lag_c = np.clip(lag, 0, T - 1)

    out = [p_all[lag_c], np.log1p(n_all[lag_c])]
    for _, cols in SEQ_GROUPS:
        g_obs = _key(obs, cols)
        codes, uniq = pd.factorize(g_obs)
        G = len(uniq)
        flat = obs_day * G + codes
        k = np.bincount(flat, weights=obs["delayed"].to_numpy(), minlength=T * G).reshape(T, G)
        n = np.bincount(flat, minlength=T * G).reshape(T, G).astype(float)
        rate = (k + alpha * p_all[:, None]) / (n + alpha)
        g_row = pd.Index(uniq).get_indexer(_key(rows, cols))          # -1 = group never seen
        known = (g_row >= 0)[:, None]
        g_c = np.clip(g_row, 0, G - 1)[:, None]
        base = p_all[lag_c] if relative else 0.0
        out.append(np.where(known, rate[lag_c, g_c] - base, 0.0 if relative else p_all[lag_c]))
        out.append(np.where(known, np.log1p(n[lag_c, g_c]), 0.0))
    dow = dates.day_of_week.to_numpy()[lag_c]
    out += [np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)]
    seq = np.stack(out, axis=-1).astype(np.float32)
    seq[~valid] = 0.0
    return seq
