"""Phase 6 feature engineering (creation). Every feature is known at 05:00 on day D.

Groups
  schedule   : cyclic departure/arrival time, distance, overnight arrival
  calendar   : cyclic day of week, weekend, days to nearest US federal holiday
  aircraft   : plane age, seats, engines (static)
  congestion : how many flights are SCHEDULED at the origin in the hour / +-1 h / day, the carrier's
               flights at the origin that day, NYC arrivals scheduled into the destination that hour
  history    : smoothed delay rate + log volume over trailing windows that END ON DAY D-1
"""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

# (name, grouping columns, window in days). Windows come from the ACF analysis in notebook 02.
HISTORY_SPECS = [
    ("all", [], 1), ("all", [], 7),
    ("carrier", ["carrier"], 7), ("carrier", ["carrier"], 28),
    ("origin", ["origin"], 1), ("origin", ["origin"], 7),
    ("dest", ["dest"], 28), ("route", ["origin", "dest"], 28),
    ("carrier_origin", ["carrier", "origin"], 7),
    ("origin_hour", ["origin", "sched_hour"], 28),
    ("flight", ["carrier", "flight"], 28),
    ("tail", ["tailnum"], 7),
]
COLD_START_RATE = 0.20   # used only before any history exists (first days of January)


def _key(df, cols):
    if not cols:
        return pd.Series("ALL", index=df.index)
    k = df[cols[0]].astype(str)
    for c in cols[1:]:
        k = k + "|" + df[c].astype(str)
    return k


def trailing_counts(df, cols, window):
    """(delayed count, flight count) per (date, group) over the `window` days BEFORE each date.

    .rolling(window).sum() covers days [D-window+1, D]; .shift(1) moves it to [D-window, D-1],
    so outcomes of day D itself are never included. This one line is what prevents leakage.
    """
    obs = df[df["delayed"].notna()].assign(_g=lambda d: _key(d, cols))
    dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    k = obs.pivot_table(index="date", columns="_g", values="delayed", aggfunc="sum", fill_value=0)
    n = obs.pivot_table(index="date", columns="_g", values="delayed", aggfunc="size", fill_value=0)
    roll = lambda m: m.reindex(dates, fill_value=0).rolling(window, min_periods=1).sum().shift(1).fillna(0)
    k, n = roll(k), roll(n)
    k.index.name = n.index.name = "date"
    return k.stack(), n.stack()


def build_features(df: pd.DataFrame, alpha: float = 10.0):
    """Return (X, groups). X = df plus feature columns; groups = {group name: [columns]}."""
    X = df.copy()
    g = {"schedule": [], "calendar": [], "aircraft": ["plane_age", "plane_seats", "plane_engines"],
         "congestion": [], "history": []}

    # schedule: time of day lives on a circle (23:59 is next to 00:00), so encode with sin/cos
    for col, name in (("sched_dep_min", "dep"), ("sched_arr_min", "arr")):
        ang = 2 * np.pi * X[col] / 1440
        X[f"{name}_sin"], X[f"{name}_cos"] = np.sin(ang), np.cos(ang)
    X["log_distance"] = np.log(X["distance"])
    X["overnight_arrival"] = (X["sched_arr_min"] < X["sched_dep_min"]).astype(int)
    g["schedule"] = ["dep_sin", "dep_cos", "arr_sin", "arr_cos", "sched_hour", "log_distance", "overnight_arrival"]

    # calendar (no month / day-of-year: test months never appear in training, so they act as date IDs)
    dow = X["date"].dt.dayofweek
    X["dow_sin"], X["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    X["is_weekend"] = (dow >= 5).astype(int)
    hol = USFederalHolidayCalendar().holidays("2012-12-01", "2014-01-31").values
    days = np.abs(X["date"].values[:, None] - hol[None, :]).min(axis=1) / np.timedelta64(1, "D")
    X["days_to_holiday"] = np.minimum(days, 14)
    g["calendar"] = ["dow_sin", "dow_cos", "is_weekend", "days_to_holiday"]

    # congestion from the published schedule (cancelled flights included: they were scheduled)
    per_hour = X.groupby(["origin", "date", "sched_hour"])["flight"].transform("size")
    X["origin_hour_sched"] = per_hour - 1
    oh = X.groupby(["origin", "date", "sched_hour"]).size()
    idx = lambda h: pd.MultiIndex.from_arrays([X["origin"], X["date"], h])
    X["origin_3h_sched"] = sum(oh.reindex(idx(X["sched_hour"] + s)).fillna(0).to_numpy() for s in (-1, 0, 1)) - 1
    X["origin_day_sched"] = X.groupby(["origin", "date"])["flight"].transform("size") - 1
    X["carrier_origin_day_sched"] = X.groupby(["carrier", "origin", "date"])["flight"].transform("size") - 1
    X["dest_arr_hour_sched"] = X.groupby(["dest", "date", "arr_hour"])["flight"].transform("size") - 1
    g["congestion"] = ["origin_hour_sched", "origin_3h_sched", "origin_day_sched", "carrier_origin_day_sched", "dest_arr_hour_sched"]

    # history: smoothed past delay rates. rate = (k + alpha * prior) / (n + alpha)
    priors = {}
    for name, cols, w in HISTORY_SPECS:
        if w not in priors:
            pk, pn = trailing_counts(df, [], w)
            priors[w] = (pk.droplevel(1) / pn.droplevel(1).replace(0, np.nan)).fillna(COLD_START_RATE)
        k, n = trailing_counts(df, cols, w)
        ix = pd.MultiIndex.from_arrays([X["date"], _key(X, cols)])
        kk = k.reindex(ix).fillna(0).to_numpy()
        nn = n.reindex(ix).fillna(0).to_numpy()
        prior = priors[w].reindex(X["date"]).fillna(COLD_START_RATE).to_numpy()
        X[f"h_{name}_{w}d_rate"] = (kk + alpha * prior) / (nn + alpha)
        g["history"].append(f"h_{name}_{w}d_rate")
        if cols:  # volume tells the model how much evidence the rate is based on
            X[f"h_{name}_{w}d_logn"] = np.log1p(nn)
            g["history"].append(f"h_{name}_{w}d_logn")
    return X, g


def numeric_columns(groups, include=("schedule", "calendar", "aircraft", "congestion", "history")):
    return [c for name in include for c in groups[name]]
