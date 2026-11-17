"""Phase 2 · "Your data is a time series. Why not ARIMA?" - the classical check, measured.

Writes reports/06_arima.json, which the Phase 2 report reads. Uses fdp/timeseries.py.
Run from the repo root:  python report/arima_check.py   (about 3 minutes on a laptop CPU)

1. ARIMA as a FORECASTER of the daily late rate: parameters fitted on days up to 31 Aug,
   one-step forecasts judged on September and October against simple forecasts.
2. ARIMA as a SCORE and as a FEATURE for ranking flights, on the rolling months Jul/Aug/Sep.
   For a scored month m the parameters are fitted on days before m, and the forecast for day D
   uses only days before D, so nothing from month m leaks in.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fdp import baselines, config, data, features, timeseries as ts  # noqa: E402

df = data.build_dataset(*data.load_raw())
X, groups = features.build_features(df)
NUM, CAT = features.numeric_columns(groups), config.CATEGORICAL
S = data.split(X)
rate = ts.daily_rate(df)
FIT_END = config.SPLITS["train"][1]

# 1. forecasting the daily level
forecasts = []
for order in ts.CANDIDATES:
    fc, fitted = ts.one_step_forecasts(rate, order, FIT_END)
    forecasts.append({"forecast": f"ARIMA{order}", "aic": float(fitted.aic),
                      "mae_sep": ts.mae(rate, fc, "2013-09-01", "2013-09-30"),
                      "mae_oct": ts.mae(rate, fc, "2013-10-01", "2013-10-31")})
for name, fc in ts.naive_forecasts(rate, FIT_END).items():
    forecasts.append({"forecast": name, "aic": None, "mae_sep": ts.mae(rate, fc, "2013-09-01", "2013-09-30"),
                      "mae_oct": ts.mae(rate, fc, "2013-10-01", "2013-10-31")})

# 2. ranking flights within the day
lab = pd.concat([S["train"], S["valid"]])
month = lab["date"].dt.month.to_numpy()
y = lab["delayed"].astype(int).to_numpy()
lr_scores = []
for m in config.CV_MONTHS:
    tr, te = month < m, month == m
    lr = baselines.logistic_regression(NUM, CAT, config.PHASE1_LR_C).fit(lab[tr], y[tr])
    lr_scores.append(float(ap(y[te], lr.decision_function(lab[te]))))

ranking = []
for order in [(1, 0, 1), (0, 1, 1)]:
    alone, with_feature, fc_rows, yday_rows = [], [], [], []
    for m in config.CV_MONTHS:
        fc, _ = ts.one_step_forecasts(rate, order, pd.Timestamp(2013, m, 1) - pd.Timedelta(days=1))
        part = lab.assign(arima_fc=lab["date"].map(fc).to_numpy())
        tr, te = month < m, month == m
        alone.append(float(ap(y[te], part.loc[te, "arima_fc"])))
        lr = baselines.logistic_regression(NUM + ["arima_fc"], CAT, config.PHASE1_LR_C).fit(part[tr], y[tr])
        with_feature.append(float(ap(y[te], lr.decision_function(part[te]))))
        fc_rows.append(part.loc[te, "arima_fc"].to_numpy())
        yday_rows.append(part.loc[te, "h_all_1d_rate"].to_numpy())
    ranking.append({"order": list(order), "alone": alone, "alone_mean": float(np.mean(alone)),
                    "lr_plus_arima": with_feature, "lr_plus_arima_mean": float(np.mean(with_feature)),
                    "gain": float(np.mean(with_feature) - np.mean(lr_scores)),
                    "corr_with_yesterday_rate": float(np.corrcoef(np.concatenate(fc_rows), np.concatenate(yday_rows))[0, 1])})
    print(ranking[-1], flush=True)

out = {"fit_end": str(FIT_END.date()), "forecasts": forecasts,
       "mean_daily_rate": {"September": float(rate["2013-09"].mean()), "October": float(rate["2013-10"].mean())},
       "months": config.CV_MONTHS, "lr_rolling": lr_scores, "lr_rolling_mean": float(np.mean(lr_scores)), "ranking": ranking}
(ROOT / "reports" / "06_arima.json").write_text(json.dumps(out, indent=2))
print("saved reports/06_arima.json")
