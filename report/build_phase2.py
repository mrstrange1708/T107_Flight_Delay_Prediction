"""Fill report/phase2.tex.tmpl with numbers from reports/06-10_*.json and write report/phase2.tex.

Same rule as the Phase 1 report: every number comes from the JSON files the notebooks (and
report/arima_check.py) saved; none is typed by hand. <<expression>> in the template is replaced by
the value of a Python expression evaluated in the namespace built below.

Run from the repo root:  python report/build_phase2.py   (then: tectonic report/phase2.tex)
"""
import re

from build_report import HERE, f2, f3, f4, load, n, table, tex   # Phase 1 helpers

P1 = {r["model"]: r for r in load("05_boosting_evaluation.json")["test_table"]}
d6, ar, d7, d8, d9, d10 = (load(f) for f in ["06_dl_data.json", "06_arima.json", "07_mlp.json",
                                               "08_temporal_hybrid.json", "09_ablation_calibration.json",
                                               "10_final_rescheduling.json"])
T = {r["model"]: r for r in d10["test_table"]}
MLP, GRU, HYB = "MLP + embeddings", "Temporal: GRU replaces hand history", "Hybrid: LR + temporal correction"
LR, XGB, XHY = "Logistic regression", "XGBoost (Phase 1)", "Hybrid: XGBoost from LR (Phase 1)"
NETS = [MLP, GRU, HYB]
roll = {r["model"]: r for r in d8["rolling"]}
seeds = {r["model"]: r for r in d8["seed_spread"]}
epochs = d8["epochs_per_fold"]
abl = {r["ablation"]: r for r in d9["ablation"]}
cal = {r["model"]: r for r in d9["calibration"]}
biz = {(r["model"], r["rule"][:3]): r for r in d10["business"]}
pol = {r["policy"]: r for r in d10["policies"]}
sweep = {r["buffer (min)"]: r for r in d10["sweep"]}
fc = {r["forecast"]: r for r in ar["forecasts"]}
rank = {tuple(r["order"]): r for r in ar["ranking"]}
cats = {r["column"]: r for r in d6["categories"]}
drift = d6["drift_monthly"]
ABS, REL = "absolute: share above July's 75th percentile", "relative: share above July's 75th percentile"
lr_roll = d8["lr_rolling_mean"]
seed_range = d9["seed_range_full"]
base = d9["base_rates"]

nan = lambda x: x != x
pm = lambda x: f"{float(x):+.3f}"
ci = lambda r: f"[{f3(r['ci_low'])}, {f3(r['ci_high'])}]"
dci = lambda r: f"{pm(r['diff_vs_lr'])} [{pm(r['diff_ci_low'])}, {pm(r['diff_ci_high'])}]"
cap = lambda s: s[0].upper() + s[1:]
names = lambda xs: ", ".join(xs[:-1]) + (" and " if len(xs) > 1 else "") + xs[-1] if xs else "none"
NICK = {MLP: "the MLP", GRU: "the temporal model", HYB: "the hybrid", LR: "logistic regression",
        XGB: "Phase 1's XGBoost", XHY: "Phase 1's XGBoost hybrid"}
SHORT = {LR: "LR", XGB: "XGBoost (P1)", XHY: "XGB from LR (P1)", MLP: "MLP", GRU: "Temporal", HYB: "Hybrid"}
ROLL_SHORT = {LR: "Logistic regression", MLP: "MLP + embeddings", "Temporal: GRU + all features": "GRU + all features",
              GRU: "GRU, no hand history", HYB: "Hybrid: LR + GRU", "XGBoost, tuned (Phase 1 report)": "XGBoost (P1 report)",
              "Hybrid: XGBoost from LR (Phase 1 report)": "XGB from LR (P1 report)"}

# Sentences whose wording depends on how the results came out, so the text cannot contradict the numbers.
sig = [m for m in NETS if T[m]["diff_ci_low"] > 0]
not_sig = [m for m in NETS if m not in sig]
sig_sentence = (f"All three networks beat logistic regression on the test months, and every paired interval excludes zero"
                if not not_sig else
                f"{cap(names([NICK[m] for m in sig])) if sig else 'No network'} beat{'s' if len(sig) == 1 else ''} logistic regression with a paired interval above zero; "
                f"for {names([NICK[m] for m in not_sig])} the interval contains zero")
ahead_every_seed = [m for m in NETS if min(seeds[m]["rolling means"]) > lr_roll]
DEPLOY = HYB if HYB in ahead_every_seed else (ahead_every_seed[0] if ahead_every_seed else LR)
best_test = max(NETS, key=lambda m: T[m]["pr_auc"])
cal_better = [m for m in [LR, XGB, XHY] + NETS if T[m]["ece"] < T[m]["ece_raw"]]
cal_worse = [m for m in [LR, XGB, XHY] + NETS if T[m]["ece"] >= T[m]["ece_raw"]]
T_values = [cal[m]["T"] for m in cal]
conf_word = "under-confident" if max(T_values) < 1 else ("over-confident" if min(T_values) > 1 else "mixed")
gain_full = abl["full model"]["mean"]
abl_rows = sorted((r for k, r in abl.items() if k != "full model"), key=lambda r: r["change vs full"])
inside_band = [r["ablation"] for r in abl_rows if abs(r["change vs full"]) <= seed_range]
hyb_b, lr_b = biz[(HYB, "(b)")], biz[(LR, "(b)")]

LOG = [
    ("06", "Same label, 05:00 decision time, split and test-once rule as Phase 1", "Phase 2 is judged against Phase 1 on equal terms"),
    ("06", "z-score numeric block + missing-aircraft flag, fitted on training rows",
     f"{d6['n_numeric']} features + 1 flag = {d6['numeric_block_width']} inputs"),
    ("06", "Category indices: 0 unknown, 1 rare ($<$50), 2... common",
     f"Embedding sizes {cats['carrier']['embedding size']} / {cats['origin']['embedding size']} / {cats['dest']['embedding size']} (carrier / origin / dest)"),
    ("06", f"Ordered {d6['seq_len']}-day histories ending on D$-$1, {d6['n_channels']} channels",
     "Sequence leakage test passes" if d6["leak_tests_passed"] else "Leakage test FAILED"),
    ("06", "History stored relative to the day's overall rate",
     f"July rule fires {f3(min(r[REL] for r in drift))}--{f3(max(r[REL] for r in drift))} (relative) vs {f3(min(r[ABS] for r in drift))}--{f3(max(r[ABS] for r in drift))} (absolute)"),
    ("06", "ARIMA measured as forecaster, score and feature",
     f"Alone {f3(rank[(1, 0, 1)]['alone_mean'])} vs LR {f3(ar['lr_rolling_mean'])}; as a feature {rank[(1, 0, 1)]['gain']:+.4f}"),
    ("07", "Compact MLP, learned embeddings, log loss", f"{n(d7['n_params'])} weights, {n(d7['emb_params'])} in embeddings"),
    ("07", "Dropout, AdamW weight decay, early stopping on PR AUC", "Without them September peaks early (Fig.~\\ref{fig:overfit})"),
    ("07", "Epochs chosen like trees: stop on $m-1$, refit, score $m$",
     f"{len(d7['search'])} settings within {f3(d7['search'][0]['mean'] - d7['search'][-1]['mean'])}"),
    ("08", "GRU (not LSTM/Transformer) over the 14 days", f"Rolling mean {f4(roll[GRU]['mean'])} with no hand-made history"),
    ("08", "Hybrid: LR logit as fixed offset, zero-initialised correction, epoch 0 scored",
     f"Rolling mean {f4(roll[HYB]['mean'])} vs LR {f4(lr_roll)}"),
    ("08", "Seed spread before any claim", f"Ranges {f4(seeds[MLP]['range'])} / {f4(seeds[GRU]['range'])} / {f4(seeds[HYB]['range'])}"),
    ("08", "Decision rule: ahead of LR only if every seed is", f"Met by {names([NICK[m] for m in ahead_every_seed])}"),
    ("09", "One-change ablations, same protocol and seed", f"$-$GRU: {pm(abl[next(k for k in abl if 'GRU branch' in k)]['change vs full'])}"),
    ("09", "Calibration method chosen on Sep $\\rightarrow$ Oct", f"{d9['calibration_choice'].capitalize()} (T = {f3(min(T_values))}--{f3(max(T_values))})"),
    ("10", "Final fit Jan 15 -- Sep 30, median fold epochs",
     ", ".join(f"{NICK[m]} {e}" for m, e in d10["final_epochs"].items())),
    ("10", "Test months scored once; paired day-block bootstrap vs LR", f"Hybrid {dci(T[HYB])}"),
    ("10", "Flag rule fixed in Phase 1: calibrated $p \\geq 0.5$",
     f"Hybrid precision {f3(hyb_b['precision'])}, recall {f3(hyb_b['recall'])}"),
    ("10", f"What-if: same budget ({n(d10['k_flagged'])} flights $\\times$ {d10['buffer_min']} min) for every policy",
     f"Hybrid avoids {pol['hybrid flags']['late arrivals avoided']:.0f}, random {pol['random flights']['late arrivals avoided']:.0f}"),
]

ns = dict(globals())
ns.update(
    inputs_table=table([[c, r["categories in training"], r["table rows (incl. unknown + rare)"], r["embedding size"], r["one-hot width"]]
                        for c, r in cats.items()]),
    drift_table=table([["absolute", *(f2(r[ABS]) for r in drift)], ["relative", *(f2(r[REL]) for r in drift)]]),
    forecast_table=table([[tex(r["forecast"]), "--" if r["aic"] is None else f"{r['aic']:.1f}", f3(r["mae_sep"]), f3(r["mae_oct"])]
                          for r in ar["forecasts"]]),
    arima_rank_table=table([[f"ARIMA{tuple(r['order'])}", f4(r["alone_mean"]), f4(ar["lr_rolling_mean"]), f4(r["lr_plus_arima_mean"]),
                             f"{r['gain']:+.4f}", f2(r["corr_with_yesterday_rate"])] for r in ar["ranking"]]),
    search_table=table([[r["hidden"].replace(", ", "-").strip("()"), r["dropout"], r["lr"], r["weight_decay"], f4(r["mean"]),
                         str(r["epochs per fold"]).replace(" ", "")] for r in d7["search"]]),
    rolling_table=table([[ROLL_SHORT.get(m, tex(m)), *(f4(r[k]) for k in ("Jul", "Aug", "Sep", "mean")),
                          str(epochs[m]).replace(" ", "") if m in epochs else "--"] for m, r in roll.items()]),
    seed_table=table([[SHORT[m], ", ".join(f4(v) for v in r["rolling means"]), f4(r["mean over seeds"]), f4(r["range"])] for m, r in seeds.items()]),
    ablation_table=table([[tex(r["ablation"]).replace("−", "$-$"), f4(r["mean"]), f"{r['change vs full']:+.4f}", str(r["epochs"]).replace(" ", "")]
                          for r in d9["ablation"]]),
    calib_table=table([[SHORT[m], f3(r["T"]), f3(r["October ECE raw"]), f3(r["after temperature"]), f3(r["after Platt"])] for m, r in cal.items()]),
    test_table=table([[tex(m), f3(r["pr_auc"]), "--" if nan(r["ci_low"]) else ci(r),
                       "--" if nan(r["diff_ci_low"]) else dci(r), f3(r["roc_auc"]), f2(r["lift"]), f3(r["brier"]),
                       f3(r["ece"]) if nan(r["ece_raw"]) else f"{f3(r['ece_raw'])} $\\rightarrow$ {f3(r['ece'])}"] for m, r in T.items()]),
    biz_table=table([[SHORT[m], r, f3(r_["precision"]), f3(r_["recall"]),
                      f"{r_['flagged per 1,000']:.1f}", f"{r_['really late among them']:.1f}", f"{r_['random list would catch']:.1f}"]
                     for (m, r), r_ in biz.items()]),
    policy_table=table([[p, f"{r['late arrivals avoided']:.0f}", f2(r["avoided per 1,000 flights"]),
                         f"{100 * r['share of all late arrivals avoided']:.1f}\\%", f"{100 * r['share of buffer minutes used']:.1f}\\%"]
                        for p, r in pol.items()]),
    sweep_table=table([[b, r["late arrivals avoided"], r["still late after the buffer"], r["padded but on time anyway"],
                        f"{r['avoided per 1,000 buffer minutes']:.1f}", f"{100 * r['share of buffer minutes used']:.1f}\\%"] for b, r in sweep.items()]),
    decision_log=table([[i, nb, d, o] for i, (nb, d, o) in enumerate(LOG, 1)]),
)


def fill(template: str) -> str:
    return re.sub(r"<<(.+?)>>", lambda m: str(eval(m.group(1), ns)), template, flags=re.S)   # our own template


if __name__ == "__main__":
    (HERE / "phase2.tex").write_text(fill((HERE / "phase2.tex.tmpl").read_text()))
    print("wrote report/phase2.tex")
