"""Fill report/main.tex.tmpl with numbers from reports/*.json and write report/main.tex.

Every number in the report comes from the JSON files the notebooks saved; none is typed by hand.
In the template, <<expression>> is replaced by the value of a Python expression evaluated in the
namespace built below (d1..d5 = the five JSON files, T = test table by model, helpers f2/f3/...).

Run from the repo root:  python report/build_report.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "report"


def load(name):
    return json.loads((ROOT / "reports" / name).read_text())   # Python's json accepts NaN


d1, d2, d3, d4, d5 = (load(n) for n in ["01_problem_data.json", "02_eda_preprocessing.json", "03_features.json",
                                         "04_model_selection.json", "05_boosting_evaluation.json"])
f2, f3, f4 = (lambda x: f"{float(x):.2f}"), (lambda x: f"{float(x):.3f}"), (lambda x: f"{float(x):.4f}")
n = lambda x: f"{int(x):,}"
tex = lambda s: str(s).replace("\\", "/").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")

T = {r["model"]: r for r in d5["test_table"]}
LR, GB, HY = T["Logistic regression"], T["Gradient boosting"], T["Hybrid (boosting from LR)"]
RULE, MAJ = T["Rule: flight's 28-day delay rate"], T["Majority class"]
split = {r["block"]: r for r in d1["split_table"]}
sfs = d3["sfs"]
grp = d3["group_table"]
ms4 = {r["model"]: r for r in d4["model_selection"]}
cv = d5["cv_compare"]
trap = d5["early_stopping_trap"]
opt, hon = trap["early-stop on the scored month (optimistic)"], trap["our protocol (honest)"]
search = d5["search"]
best = d5["best_params"]
abl = {r["features"]: r["october_pr_auc"] for r in d5["ablation_october"]}
biz = d5["business"]
biz_a = [r for r in biz if r["rule"].startswith("(a)")]
biz_b = [r for r in biz if r["rule"].startswith("(b)")]
lift = {r["slice"]: r for r in d5["lift_by_slice"]}
depth = {r["max_depth"]: r for r in d4["tree_depth"]}
trees_gb = json.loads(search[0]["trees_per_fold"])
rng = lambda rows, k, fmt=f3: f"{fmt(min(r[k] for r in rows))}--{fmt(max(r[k] for r in rows))}"
ci = lambda r: f"[{f3(r['ci_low'])}, {f3(r['ci_high'])}]"
monthly = d1["monthly_delay_rate"]
leak = re.sub(r"\x1b\[[0-9;]*m", "", d3["leakage_tests"])

# Sentences whose wording depends on how the results came out, so the text cannot contradict the numbers.
ranked = sorted(ms4.items(), key=lambda kv: -kv[1]["mean_pr_auc"])
short = lambda m: "untuned XGBoost" if m.startswith("Gradient boosting") else m.lower()
family_sentence = "Mean PR AUC ranks them: " + ", ".join(f"{short(m)} ({f3(r['mean_pr_auc'])})" for m, r in ranked) + "."
lr_cv = cv["Logistic regression"]["mean"]
search_vs_lr = (f", all below logistic regression ({f3(lr_cv)})" if all(float(r["mean_pr_auc"]) < lr_cv for r in search)
                else f" (logistic regression: {f3(lr_cv)})")
learned = {"logistic regression": LR, "boosting": GB, "the hybrid": HY}
ece_change = lambda k: f"{k} ({f3(learned[k]['ece'])} $\\rightarrow$ {f3(learned[k]['ece_after_platt'])})"
better = [k for k, r in learned.items() if r["ece_after_platt"] < r["ece"]]
worse = [k for k, r in learned.items() if r["ece_after_platt"] >= r["ece"]]
platt_sentence = "Platt scaling on October " + " and ".join(
    part for part in [("lowered the calibration error of " + " and ".join(map(ece_change, better))) if better else "",
                      ("raised it for " + " and ".join(map(ece_change, worse))) if worse else ""] if part) + "."
if worse:
    platt_sentence += " October's lower base rate pulls calibrated probabilities down, which hurts models that were already well calibrated."

# ------------------------------------------------------------------ decision log: (notebook, decision, outcome)
LOG = [
    ("01", "Binary label: arrives $\\geq$15 min late",
     f"{n(d1['summary']['labelled'])} labelled flights, {f3(d1['summary']['delayed_share_of_labelled'])} delayed"),
    ("01", "Decision time 05:00; history ends on D$-$1; post-departure columns banned", "Both leakage tests pass" if leak.startswith("2 passed") else f"Leakage tests: {leak}"),
    ("01", "Cancelled/diverted flights unlabelled but kept on the schedule",
     f"{n(d1['summary']['cancelled'])} cancelled + {n(d1['summary']['diverted'])} diverted not scored"),
    ("01", "PR AUC as main metric, not accuracy",
     f"``Always on time'' has {f3(d1['valid_always_on_time_accuracy'])} accuracy and zero recall"),
    ("01", "Chronological split; test Nov--Dec used once",
     f"Monthly delay rate {f3(min(monthly.values()))}--{f3(max(monthly.values()))}"),
    ("01", "Day-block bootstrap confidence intervals", f"LR test PR AUC CI {ci(LR)}"),
    ("02", "EDA and preprocessing statistics from training rows only", "All fitted inside scikit-learn pipelines"),
    ("02", "Impute aircraft fields + missing indicator, keep rows",
     f"{d2['missing_pct']['plane_year']}\\% missing, mostly MQ and AA; no rows lost"),
    ("02", "Keep delay outliers", f"{100 * d2['iqr_outlier_share']:.1f}\\% IQR outliers, all real delays"),
    ("02", "History windows from ACF/PACF", f"ACF lag 1 = {d2['acf']['lag1']}, lag 7 = {d2['acf']['lag7']}"),
    ("02", "No SMOTE / re-sampling", f"LR raw test ECE {f3(LR['ece'])}"),
    ("02", "Standard scaler for linear models",
     f"Tie with robust scaler ({d2['scaler_comparison']['standard']['valid_pr_auc']} vs {d2['scaler_comparison']['robust']['valid_pr_auc']})"),
    ("03", f"{d3['n_numeric']} numeric features in 5 groups + 3 categorical",
     f"All groups best: Sep {f3(grp[4]['valid_pr_auc'])}, Oct {f3(grp[4]['calib_pr_auc'])}"),
    ("03", f"Shrink history rates towards the overall rate ($\\alpha$ = {d3['alpha']})", "Small, noisy groups pulled to the mean"),
    ("03", "No month or day-of-year feature", "Test months never seen in training"),
    ("03", "Keep all features unless a method wins on Sep and Oct",
     f"None did (SFS: Oct {f3(sfs['calib_sfs'])} vs {f3(sfs['calib_all'])})"),
    ("04", "Baselines before any learned model",
     f"Sep PR AUC: majority {f3(d4['baselines']['valid']['majority_class'])}, 28-day rule {f3(d4['baselines']['valid']['rule_flight_28d_rate'])}"),
    ("04", "Learning-rate demo with batch gradient descent",
     f"Log loss after 100 epochs: lr 0.01 {d4['lr_effect_logloss']['0.01']}, 0.5 {d4['lr_effect_logloss']['0.5']}, 8 {d4['lr_effect_logloss']['8.0']}"),
    ("04", "L2 strength C chosen on September only", f"C = {d4['best_C']}"),
    ("04", "Decision-tree depth sweep",
     f"Unlimited depth: train {f3(depth['None']['train_pr_auc'])}, Sep {f3(depth['None']['valid_pr_auc'])}"),
    ("04", "Families compared on rolling Jul/Aug/Sep, fixed settings",
     " $>$ ".join(f"{short(m)} {f3(r['mean_pr_auc'])}" for m, r in ranked)),
    ("05", "Honest early stopping: stop on $m-1$, refit, score $m$",
     f"Optimistic {f3(opt['mean'])} vs honest {f3(hon['mean'])}"),
    ("05", f"Random search over {len(search)} of {d5['grid_size']} XGBoost configurations",
     f"Best mean {f3(search[0]['mean_pr_auc'])} (LR {f3(lr_cv)})"),
    ("05", "Hybrid: XGBoost starts from LR's logit", f"Rolling mean {f3(cv['Hybrid: boosting from LR']['mean'])}"),
    ("05", "Final tree count = median of fold counts",
     f"{d5['n_trees_final']} trees (September alone: {d5['sept_only_trees']})"),
    ("05", "Platt calibration on October", f"Boosting ECE {f3(GB['ece'])} $\\rightarrow$ {f3(GB['ece_after_platt'])}"),
    ("05", "Test months scored once, all models together",
     f"PR AUC: LR {f3(LR['pr_auc'])}, boosting {f3(GB['pr_auc'])}, hybrid {f3(HY['pr_auc'])}"),
    ("05", "Flag when calibrated $p \\geq 0.5$", f"Precision {rng(biz_b, 'precision')}, recall {rng(biz_b, 'recall')}"),
    ("05", "Importance and ablation on October, never on test",
     f"Ablation: peak {f3(abl['+ aircraft'])}, + history {f3(abl['+ history'])}"),
]


def table(rows):
    return "\n".join(" & ".join(map(str, r)) + r"\\" for r in rows)


ns = dict(globals())
ns.update(
    test_table=table([[tex(m), f3(r["pr_auc"]), "--" if r["ci_low"] != r["ci_low"] else ci(r), f3(r["roc_auc"]), f2(r["lift"]),
                       f3(r["brier"]), f3(r["ece"]) + ("" if r["ece_after_platt"] != r["ece_after_platt"] else " / " + f3(r["ece_after_platt"]))]
                      for m, r in T.items()]),
    rolling_table=table([[m, *(f3(cv[m][k]) for k in ("Jul", "Aug", "Sep", "mean"))] for m in cv]),
    family_table=table([[m.replace(" (XGBoost, untuned)", " (untuned)"), *(f3(r[k]) for k in ("Jul", "Aug", "Sep", "mean_pr_auc")), f"{r['fit_seconds']:.1f}"]
                        for m, r in ms4.items()]),
    selection_table=table([[tex(r["method"]), r["n"], f3(r["valid"]), f3(r["calib"])] for r in d3["decision"]]),
    trap_table=table([[k, f3(opt[k]), f3(hon[k])] for k in ("Jul", "Aug", "Sep", "mean")]),
    business_table=table([[{"Logistic regression": "LR", "Gradient boosting": "Boosting"}.get(r["model"], "Hybrid"), r["rule"][:3],
                           f3(r["precision"]), f3(r["recall"]), f"{r['flagged per 1000']:.1f}",
                           f"{r['truly delayed among them']:.1f}", f"{r['random list would catch']:.1f}"] for r in biz]),
    decision_log=table([[i, nb, d, o] for i, (nb, d, o) in enumerate(LOG, 1)]),
)


def fill(template: str) -> str:
    def value(m):
        return str(eval(m.group(1), ns))   # expressions come from our own template file
    return re.sub(r"<<(.+?)>>", value, template, flags=re.S)


if __name__ == "__main__":
    (HERE / "main.tex").write_text(fill((HERE / "main.tex.tmpl").read_text()))
    print("wrote report/main.tex")
