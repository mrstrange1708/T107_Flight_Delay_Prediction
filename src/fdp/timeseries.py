"""Phase 2 · classical time-series check (ARIMA) on the daily late rate.
Owner: Balanagu Sesha Srikar (Srikar).

The evaluator's question: "your data is a time series, why no ARIMA?"
ARIMA forecasts ONE series (here: the share of NYC flights that arrive late each day). It cannot
tell two flights on the same day apart, and our task is to rank ~900 flights WITHIN each day.
So ARIMA cannot be the model, but it can be (1) a forecaster of the day's overall level and
(2) one extra feature. This module measures both instead of arguing.

Leakage rule, same as everywhere: parameters are fitted on days up to `fit_end` only, and the
forecast for day D uses the observed rates of days before D (one-step-ahead).
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

CANDIDATES = [(1, 0, 0), (1, 0, 1), (2, 0, 1), (0, 1, 1)]
# (0,1,1) is simple exponential smoothing: it has no fixed mean, so it follows a drifting level.


def daily_rate(df: pd.DataFrame) -> pd.Series:
    """Share of labelled flights that arrived late, per calendar day (complete by 05:00 next day)."""
    obs = df[df["delayed"].notna()]
    return obs.groupby("date")["delayed"].mean().asfreq("D").interpolate()


def one_step_forecasts(series: pd.Series, order, fit_end) -> tuple:
    """Fit ARIMA(order) on days <= fit_end, then give the one-step-ahead forecast for EVERY day
    with those fixed parameters. Returns (forecast series, fitted model)."""
    fitted = ARIMA(series[:fit_end], order=order).fit()
    return fitted.apply(series).predict(), fitted


def mae(actual: pd.Series, forecast: pd.Series, start, end) -> float:
    return float((actual[start:end] - forecast[start:end]).abs().mean())


def naive_forecasts(series: pd.Series, fit_end) -> dict:
    """The simple forecasts a classical model must beat."""
    return {"yesterday's rate": series.shift(1),
            "mean of the last 7 days": series.shift(1).rolling(7, min_periods=1).mean(),
            "training mean (no memory)": pd.Series(series[:fit_end].mean(), index=series.index)}
