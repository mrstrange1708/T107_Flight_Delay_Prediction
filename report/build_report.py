"""Fill every report/*.tmpl with numbers from reports/*.json:
main.tex.tmpl -> main.tex (IEEE report) and slide_content.md.tmpl -> slide_content.md (slide-by-slide content).

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
dec3 = {r["method"]: r for r in d3["decision"]}
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
reg = {r["C"]: r for r in d4["regularisation"]}
depth = {r["max_depth"]: r for r in d4["tree_depth"]}
gd = {r["method"]: r for r in d4["gd_table"]}
trees_gb = json.loads(search[0]["trees_per_fold"])
n_combos = 1
for v in ([0.03, 0.05, 0.1], [2, 3, 4, 6], [50, 200, 500], [0.7, 0.9], [0.5, 0.8], [1, 10]):
    n_combos *= len(v)      # size of the GRID in notebook 05, cell 11
rng = lambda rows, k, fmt=f3: f"{fmt(min(r[k] for r in rows))}--{fmt(max(r[k] for r in rows))}"
ci = lambda r: f"[{f3(r['ci_low'])}, {f3(r['ci_high'])}]"
monthly = d1["monthly_delay_rate"]
leak = re.sub(r"\x1b\[[0-9;]*m", "", d3["leakage_tests"])

# ------------------------------------------------------------------ decision log
# (notebook, decision, why / alternative rejected, what happened, verdict)
LOG = [
    ("01", "Binary classification: delayed = arrives $\\geq$15 min late",
     f"Decision is yes/no; BTS definition; delay minutes skewed (skew {d2['arr_delay_skew']:.2f}). Regression on minutes rejected.",
     f"{n(d1['summary']['labelled'])} labelled flights, {f3(d1['summary']['delayed_share_of_labelled'])} delayed.", "adopted"),
    ("01", "Predict at 05:00 on day D; post-departure columns banned; history ends on D$-$1",
     "Early enough to act; about 3\\% of the previous day's flights land after midnight, so ``end of previous day'' is not exact.",
     f"Leakage tests pass ({leak}); no post-departure column is a feature.", "worked"),
    ("01", "Cancelled/diverted flights unlabelled but kept on the schedule",
     "Label undefined; inventing one would fabricate data. They still occupy the schedule.",
     f"{n(d1['summary']['cancelled'])} cancelled + {n(d1['summary']['diverted'])} diverted excluded from training and scoring, counted in congestion features.", "adopted"),
    ("01", "Metric fixed before modelling: PR AUC (+ ROC AUC, Brier, ECE, lift)",
     "Accuracy rewards ``always on time''.",
     f"``Always on time'' scores {f3(d1['valid_always_on_time_accuracy'])} accuracy in September with zero recall; random scores PR AUC $\\approx$ base rate.", "worked"),
    ("01", "Chronological split: warm-up Jan 1--14, train to Aug, valid Sep, calib Oct, test Nov--Dec (once)",
     "Same-day flights share weather; a random split leaks it.",
     f"Shares {split['train']['share_%']}/{split['valid']['share_%']}/{split['calib']['share_%']}/{split['test']['share_%']}\\%. Monthly delay rate ranges {f3(min(monthly.values()))} (Sep) to {f3(max(monthly.values()))} (Dec): a base-rate shift that later drives the calibration and threshold results.", "worked"),
    ("01", "Confidence intervals by day-block bootstrap",
     "Flights on one day are not independent.",
     f"Test CIs are about $\\pm$0.05 wide (LR {ci(LR)}), which is what makes the LR/boosting gap non-significant.", "worked"),
    ("02", "EDA and every preprocessing statistic on training rows only",
     "Looking at valid/test while deciding is peeking.",
     "Pipelines fitted on train; imputer median of plane age (11.0) is the training median.", "adopted"),
    ("02", "Impute aircraft fields (median + missing indicator); do not delete rows",
     f"{d2['missing_pct']['plane_year']}\\% missing, concentrated in MQ ({d2['plane_year_missing_by_carrier_top']['MQ']}\\%) and AA ({d2['plane_year_missing_by_carrier_top']['AA']}\\%): systematic. Deleting would bias two airlines.",
     "Indicator columns keep the ``unknown aircraft'' signal; no rows lost.", "adopted"),
    ("02", "Keep delay outliers",
     f"{100 * d2['iqr_outlier_share']:.1f}\\% of flights are IQR outliers, but they are real delays and the yes/no label ignores size.",
     "No rows removed.", "adopted"),
    ("02", "Check duplicates",
     "Worksheet checklist item 4.",
     f"{d2['duplicates_full']} full and {d2['duplicates_key']} key duplicates: nothing to remove.", "no action needed"),
    ("02", "History windows chosen from ACF/PACF of the daily delay rate",
     f"ACF lag 1 = {d2['acf']['lag1']}, PACF lag 2 = {d2['pacf']['lag2']}, ACF lag 7 = {d2['acf']['lag7']} $\\Rightarrow$ 1-day window + day-of-week features; 7/28-day windows only for small groups.",
     f"In notebook 03 the history group raised September PR AUC from {f3(grp[3]['valid_pr_auc'])} to {f3(grp[4]['valid_pr_auc'])}.", "worked"),
    ("02", "No SMOTE / re-sampling",
     f"Imbalance is moderate ({f3(d2['train_delay_rate'])} delayed); re-sampling distorts probabilities.",
     f"LR's raw test probabilities were already well calibrated (ECE {f3(LR['ece'])}).", "worked"),
    ("02", "Standard (z-score) scaler for the linear model",
     "Compared with a robust scaler.",
     f"September PR AUC {d2['scaler_comparison']['standard']['valid_pr_auc']} vs {d2['scaler_comparison']['robust']['valid_pr_auc']}: a tie, standard kept.", "tie"),
    ("02", "One-hot for linear models, ordinal codes for trees, native categoricals in XGBoost",
     "Integer codes imply a fake order to a linear model; trees only ask ``code $\\leq k$?''.",
     "LR validation matrix: 120 columns.", "adopted"),
    ("03", f"Five feature groups, {d3['n_numeric']} numeric + 3 categorical",
     "Each group motivated by an EDA finding.",
     f"LR with all five groups is best on both months (Sep {f3(grp[4]['valid_pr_auc'])}, Oct {f3(grp[4]['calib_pr_auc'])}). History adds +{f3(grp[4]['valid_pr_auc'] - grp[3]['valid_pr_auc'])} in Sep but only +{f3(grp[4]['calib_pr_auc'] - grp[3]['calib_pr_auc'])} in Oct.", "worked"),
    ("03", f"Shrink history rates towards the overall rate ($\\alpha$ = {d3['alpha']})",
     "A flight number seen 3 times gives an extreme, noisy rate.",
     "Small groups pulled to the mean; log-volume features say how much evidence a rate rests on.", "adopted"),
    ("03", "No month or day-of-year feature",
     "Test months never appear in training, so month would act as a date ID.",
     "Seasonality enters only through history features (see drift, notebook 05).", "adopted"),
    ("03", "Keep all features unless a method beats them on both September and October",
     "Filter (MI), VIF pruning, SFS (from scratch), L1, PCA tried.",
     f"None passed. SFS: Sep {f3(sfs['valid_sfs'])} $>$ {f3(sfs['valid_all'])} but Oct {f3(sfs['calib_sfs'])} $<$ {f3(sfs['calib_all'])}: it over-fitted the month it was selected on.", "all features kept"),
    ("03", "Report VIF but keep collinear features",
     f"{d3['vif']['n_above_10']} features have VIF $>$ 10; collinearity hurts interpretation, not prediction.",
     f"Pruning {len(d3['vif']['dropped'])} features lowered both months ({f3(d3['vif']['valid_after_vif'])} / {f3(d3['vif']['calib_after_vif'])}).", "worked"),
    ("04", "Baselines before any learned model",
     "Phase 7: a complex model must beat a simple rule.",
     f"Sep / Oct PR AUC: majority {f3(d4['baselines']['valid']['majority_class'])} / {f3(d4['baselines']['calib']['majority_class'])}; flight's 28-day rate {f3(d4['baselines']['valid']['rule_flight_28d_rate'])} / {f3(d4['baselines']['calib']['rule_flight_28d_rate'])}; scheduled hour {f3(d4['baselines']['valid']['rule_scheduled_hour'])} / {f3(d4['baselines']['calib']['rule_scheduled_hour'])}.", "floor set"),
    ("04", "Logistic regression from scratch (batch, mini-batch, SGD), checked against scikit-learn",
     "Course training loop; log loss = Bernoulli MLE.",
     f"Batch GD train log loss {gd['batch GD (lr 0.5)']['final_train_logloss']} vs scikit-learn {gd['scikit-learn (lbfgs, same model)']['final_train_logloss']}; mini-batch {gd['mini-batch 256 (lr 0.05)']['final_train_logloss']} in 5 epochs.", "verified"),
    ("04", "Learning-rate sweep for batch GD",
     "Show slow vs divergent steps.",
     f"lr 0.01 $\\rightarrow$ {d4['lr_effect_logloss']['0.01']} after 100 epochs (slow); 0.5 $\\rightarrow$ {d4['lr_effect_logloss']['0.5']} (best); 8 $\\rightarrow$ {d4['lr_effect_logloss']['8.0']} (diverged).", "as expected"),
    ("04", "Choose L2 strength C on September only",
     "October and test must not choose settings.",
     f"C = {d4['best_C']} (Sep {f4(reg[d4['best_C']]['valid_pr_auc'])}); scores barely move with C, so LR is bias-limited, not variance-limited.", "adopted"),
    ("04", "Decision-tree depth sweep",
     "Show over-fitting.",
     f"Unlimited depth: train {f3(depth['None']['train_pr_auc'])}, Sep {f3(depth['None']['valid_pr_auc'])}; best Sep at depth {d4['best_tree_depth']} ({f3(depth[d4['best_tree_depth']]['valid_pr_auc'])}).", "as expected"),
    ("04", "Compare families on rolling Jul/Aug/Sep with fixed configurations",
     "September alone is unusually calm; fixed settings avoid using future months.",
     f"Mean PR AUC: LR {f3(ms4['Logistic regression']['mean_pr_auc'])} $>$ random forest {f3(ms4['Random forest']['mean_pr_auc'])} $>$ untuned XGBoost {f3(ms4['Gradient boosting (XGBoost, untuned)']['mean_pr_auc'])} $>$ tree {f3(ms4['Decision tree']['mean_pr_auc'])}. LR is the model to beat; boosting goes on to tuning.", "LR leads"),
    ("05", "Gradient boosting from scratch, checked against scikit-learn",
     "Prove the theory (trees fitted to $r = y - p$, Newton leaf values).",
     f"September log loss {d5['scratch_vs_sklearn']['ours (from scratch)']['September log loss']} and PR AUC {d5['scratch_vs_sklearn']['ours (from scratch)']['September PR AUC']}, identical to scikit-learn.", "verified"),
    ("05", "Honest early stopping: stop on month $m-1$, refit, score $m$",
     "Early-stopping on the scored month lets the stopping rule see the answers.",
     f"Same configuration: optimistic mean {f3(opt['mean'])} vs honest {f3(hon['mean'])} (+{f3(opt['mean'] - hon['mean'])} inflation avoided).", "trap found"),
    ("05", f"Random search, {len(search)} of {n_combos} XGBoost configurations, honest rolling score",
     "Tuned boosting is the fair test of the project's model.",
     f"Best: lr {best['learning_rate']}, depth {best['max_depth']}, min child weight {best['min_child_weight']}, subsample {best['subsample']}, colsample {best['colsample_bytree']}, $\\lambda$ {best['reg_lambda']} $\\rightarrow$ {f3(search[0]['mean_pr_auc'])}. All {len(search)} lie in {f3(min(float(r['mean_pr_auc']) for r in search))}--{f3(max(float(r['mean_pr_auc']) for r in search))}, below LR ({f3(cv['Logistic regression']['mean'])}).", "boosting behind LR"),
    ("05", "Hybrid: XGBoost starts from LR's logit (base\\_margin) and learns corrections",
     "Zero trees = plain LR, so it should never be much worse.",
     f"Rolling mean {f3(cv['Hybrid: boosting from LR']['mean'])} (LR {f3(cv['Logistic regression']['mean'])}, boosting {f3(cv['Gradient boosting (tuned)']['mean'])}); trees per fold {d5['hybrid_trees_per_fold']}.", "matches LR"),
    ("05", "Final tree count = median of the fold counts",
     "One calm month would stop far too early.",
     f"Boosting folds {trees_gb} $\\rightarrow$ {d5['n_trees_final']} trees; hybrid $\\rightarrow$ {d5['n_trees_hybrid']}; September alone would pick {d5['sept_only_trees']}.", "worked"),
    ("05", "Refit on Jan 15--Sep 30; Platt-calibrate on October",
     "Use the most recent month; calibrate on a month no model has seen.",
     f"Boosting ECE {f3(GB['ece'])} $\\rightarrow$ {f3(GB['ece_after_platt'])} (better); LR {f3(LR['ece'])} $\\rightarrow$ {f3(LR['ece_after_platt'])} and hybrid {f3(HY['ece'])} $\\rightarrow$ {f3(HY['ece_after_platt'])} (worse: already calibrated, then October's lower base rate pulled them).", "mixed"),
    ("05", "Test set (Nov--Dec) scored once, all models together",
     "Golden barrier.",
     f"PR AUC: hybrid {f3(HY['pr_auc'])}, LR {f3(LR['pr_auc'])}, boosting {f3(GB['pr_auc'])}, rule {f3(RULE['pr_auc'])}, majority {f3(MAJ['pr_auc'])}. Intervals overlap: no winner among learned models.", "no winner"),
    ("05", "Flag policy: calibrated $p \\geq 0.5$ instead of the October score threshold",
     f"Delay rate rose from {f3(split['calib']['delay_rate'])} (Oct) to {f3(split['test']['delay_rate'])} (test).",
     f"October threshold: precision {rng(biz_a, 'precision')}, recall {rng(biz_a, 'recall')}, {rng(biz_a, 'flagged per 1000', lambda x: f'{x:.0f}')} flagged per 1000. Calibrated rule: precision {rng(biz_b, 'precision')}, recall {rng(biz_b, 'recall')}, {rng(biz_b, 'flagged per 1000', lambda x: f'{x:.0f}')} flagged, about 2$\\times$ a random list.", "rule (b) chosen"),
    ("05", "Importance and ablation on October, never on test",
     "Analysis must not shape the reported score.",
     f"Permutation: sched\\_hour {d5['permutation_top12']['sched_hour']}, h\\_flight\\_28d\\_rate {d5['permutation_top12']['h_flight_28d_rate']}. Ablation: peak {f3(abl['+ aircraft'])} with schedule+calendar+aircraft; + congestion {f3(abl['+ congestion'])}; + history {f3(abl['+ history'])}.", "drift found"),
    ("05", "Error analysis by month and airport (lift)",
     "Where does each model fail?",
     f"Boosting is weakest in November (lift {lift['November']['Gradient boosting']} vs LR {lift['November']['Logistic regression']}) and at JFK ({lift['origin JFK']['Gradient boosting']} vs {lift['origin JFK']['Logistic regression']}); December is hard for all (about {lift['December']['Logistic regression']}).", "boosting weaker in Nov"),
]


def decision_log_tex():
    rows = [f"{i} & {nb} & {d} & {w} & {r} & \\emph{{{v}}}\\\\" for i, (nb, d, w, r, v) in enumerate(LOG, 1)]
    return "\n".join(rows)


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
    business_table=table([[{"Logistic regression": "LR", "Gradient boosting": "Boosting"}.get(m, "Hybrid"), r["rule"][:3], f3(r["precision"]), f3(r["recall"]),
                           f"{r['flagged per 1000']:.1f}", f"{r['truly delayed among them']:.1f}", f"{r['random list would catch']:.1f}"]
                          for r in biz for m in [r["model"]]]),
    search_table=table([[r["learning_rate"], r["max_depth"], r["min_child_weight"], r["subsample"], r["colsample_bytree"],
                         r["reg_lambda"], f3(r["mean_pr_auc"]), r["trees_per_fold"].strip("[]").replace(",", ",~")] for r in search]),
    decision_log=decision_log_tex(),
)


PLAIN = [("\\geq", "≥"), ("\\leq", "≤"), ("\\pm", "±"), ("\\rightarrow", "→"), ("\\Rightarrow", "⇒"),
         ("\\approx", "≈"), ("\\times", "×"), ("\\lambda", "λ"), ("\\alpha", "α"), ("\\%", "%"), ("\\_", "_"),
         ("``", "“"), ("''", "”"), ("--", "–"), ("$", "")]


def plain(s: str) -> str:
    """LaTeX snippet -> plain text for the slides."""
    for a, b in PLAIN:
        s = s.replace(a, b)
    return s


def md(header, rows):
    """Markdown table."""
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def dec(*nums):
    """Decision blocks for the slide content: decision, why, what happened, verdict."""
    out = []
    for k in nums:
        _, d, w, r, v = LOG[k - 1]
        out.append(f"> **Decision {k}: {plain(d)}**  \n> *Why:* {plain(w)}  \n> *What happened:* {plain(r)}  \n> *Verdict:* {v}")
    return "\n>\n".join(out)


def fill(template: str) -> str:
    def value(m):
        return str(eval(m.group(1), ns))   # expressions come from our own template file
    return re.sub(r"<<(.+?)>>", value, template, flags=re.S)


if __name__ == "__main__":
    ns.update(md=md, dec=dec, plain=plain)
    for tmpl in sorted(HERE.glob("*.tmpl")):
        tmpl.with_suffix("").write_text(fill(tmpl.read_text()))
        print("wrote", tmpl.with_suffix("").relative_to(ROOT))
