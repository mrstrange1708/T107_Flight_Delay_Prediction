// Build the Phase 1 slide deck from reports/*.json and report/report_values.json.
// Run after build_report.py, from the repo root:
//   npm install pptxgenjs && node report/build_slides.js     -> report/T107_Phase1_Slides.pptx
// Numbers are read from the JSON files, never typed by hand.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const FIG = (f) => path.join(ROOT, "reports", "figures", f);
const load = (f) => JSON.parse(fs.readFileSync(path.join(ROOT, f), "utf8").replace(/\bNaN\b/g, "null"));
const [d1, d2, d3, d4, d5] = ["01_problem_data", "02_eda_preprocessing", "03_features", "04_model_selection", "05_boosting_evaluation"]
  .map((n) => load(`reports/${n}.json`));
const LOG = load("report/report_values.json").decision_log;
const f3 = (x) => Number(x).toFixed(3);
const n = (x) => Number(x).toLocaleString("en-US");

// Palette: night-sky navy (dominant), runway amber (accent), slate text.
const NAVY = "0B1F3A", AMBER = "F2A900", SLATE = "4A5568", MUTED = "7A8699", TINT = "F1F4F8", WHITE = "FFFFFF";
const GOOD = "2E7D4F", FIND = "9A5B00", NEUTRAL = "5B6B82";
const VERDICT = { good: ["adopted", "worked", "verified", "as expected", "floor set", "trap found", "rule (b) chosen", "all features kept"],
                  find: ["mixed", "no winner", "boosting behind LR", "drift found", "boosting weaker in Nov"] };
const verdictColor = (v) => (VERDICT.good.includes(v) ? GOOD : VERDICT.find.includes(v) ? FIND : NEUTRAL);
const HEAD = "Cambria", BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.333 x 7.5 in
pres.title = "T107 Phase 1: Flight delay prediction at 05:00";
pres.author = "Team T107";
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };

const footer = [
  { text: { text: "Team T107 · Flight delay prediction at 05:00", options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: MUTED, fontFace: BODY } } },
];
pres.defineSlideMaster({ title: "DARK", background: { color: NAVY }, objects: [
  { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 11.7, h: 1.8, fontFace: HEAD, fontSize: 40, bold: true, color: WHITE, align: "left", valign: "bottom" }, text: "" } },
  { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.2, w: 11.7, h: 1.6, fontFace: BODY, fontSize: 18, color: "C9D3E3", valign: "top" }, text: "" } },
] });
pres.defineSlideMaster({ title: "CONTENT", background: { color: WHITE }, objects: [
  { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 10.2, h: 0.95, fontFace: HEAD, fontSize: 28, bold: true, color: NAVY, align: "left", valign: "top", margin: 0 }, text: "" } },
  ...footer,
], slideNumber: { x: 12.2, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: MUTED, fontFace: BODY } });

// ---------------------------------------------------------------- helpers
const lines = (text, w, pt, bold) => Math.max(1, Math.ceil((text.length * pt * (bold ? 0.0078 : 0.0071)) / w));
const textH = (text, w, pt, bold) => (lines(text, w, pt, bold) * pt * 1.22) / 72;

function addSlide(title, section, owner) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  if (owner) {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 10.95, y: 0.42, w: 1.8, h: 0.42, rectRadius: 0.21, fill: { color: NAVY }, line: { color: NAVY }, objectName: "owner pill" });
    s.addText(owner, { x: 10.95, y: 0.42, w: 1.8, h: 0.42, fontSize: 11, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0, isTextBox: true });
  }
  return s;
}

function pill(s, text, x, y, w) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.3, rectRadius: 0.15, fill: { color: verdictColor(text) }, line: { color: verdictColor(text) } });
  s.addText(text, { x, y, w, h: 0.3, fontSize: 10, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0, isTextBox: true });
}

// A decision card: number badge, decision, what happened, verdict pill. Returns its height.
function card(s, num, x, y, w, fixedH) {
  const e = LOG[num - 1];
  const tw = w - 0.85;
  const dh = textH(e.decision, tw - 1.45, 14, true);
  const rh = textH(e.result, tw, 12, false);
  const h = fixedH || Math.max(0.75, dh) + rh + 0.38;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: TINT }, line: { color: TINT }, objectName: `decision ${num}` });
  s.addShape(pres.shapes.OVAL, { x: x + 0.15, y: y + 0.15, w: 0.45, h: 0.45, fill: { color: AMBER }, line: { color: AMBER } });
  s.addText(String(num), { x: x + 0.15, y: y + 0.15, w: 0.45, h: 0.45, fontSize: 13, bold: true, color: NAVY, align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addText(e.decision, { x: x + 0.72, y: y + 0.14, w: tw - 1.45, h: dh, fontSize: 14, bold: true, color: NAVY, valign: "top", margin: 0, isTextBox: true });
  pill(s, e.verdict, x + w - 1.5, y + 0.15, 1.35);
  const ry = y + Math.max(0.75, dh + 0.24);
  s.addText([{ text: "→ ", options: { bold: true, color: AMBER } }, { text: e.result }],
    { x: x + 0.72, y: ry, w: tw, h: y + h - ry - 0.08, fontSize: 12, color: SLATE, valign: "top", margin: 0, isTextBox: true });
  return h;
}

function stack(s, nums, x, y, w, maxY = 6.85) {
  const gap = 0.18;
  for (const k of nums) y += card(s, k, x, y, w) + gap;
  if (y - gap > maxY) console.warn(`cards overflow (${y.toFixed(2)} in) on: ${nums}`);
}

function image(s, file, x, y, w, h, alt) {
  s.addImage({ path: FIG(file), x, y, w, h, sizing: { type: "contain", w, h }, altText: alt });
}

const CHART = { catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, catAxisLabelColor: SLATE, valAxisLabelColor: MUTED,
  catAxisLabelFontFace: BODY, valAxisLabelFontFace: BODY, dataLabelFontSize: 10, dataLabelFontFace: BODY, dataLabelColor: SLATE,
  valGridLine: { color: "E3E8EF", size: 0.5 }, catGridLine: { style: "none" }, legendFontSize: 11, legendFontFace: BODY,
  legendColor: SLATE, titleFontSize: 13, titleFontFace: BODY, titleColor: NAVY, showTitle: true };
const chart = (extra) => Object.assign({}, CHART, extra);

function table(s, header, rows, x, y, w, colW, opts = {}) {
  const head = header.map((t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY } } }));
  const body = rows.map((r, i) => r.map((c) => (typeof c === "object" ? c : { text: String(c), options: { fill: { color: i % 2 ? WHITE : TINT } } })));
  s.addTable([head, ...body], { x, y, w, colW, fontSize: opts.fontSize || 13, fontFace: BODY, color: SLATE, border: { type: "solid", pt: 0.5, color: "D5DCE6" }, valign: "middle", margin: 0.06, ...opts });
}

const T = Object.fromEntries(d5.test_table.map((r) => [r.model, r]));
const LR = T["Logistic regression"], GB = T["Gradient boosting"], HY = T["Hybrid (boosting from LR)"];
const RULE = T["Rule: flight's 28-day delay rate"], MAJ = T["Majority class"];
const split = Object.fromEntries(d1.split_table.map((r) => [r.block, r]));

// ---------------------------------------------------------------- 1 title
pres.addSection({ title: "Opening" });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Opening" });
s.addText("Flight delay prediction at 05:00", { placeholder: "title" });
s.addText([
  { text: "Gradient boosting against strong baselines · NYC 2013 flights", options: { breakLine: true } },
  { text: "Team T107 · Section D · Project 21 · Phase 1", options: { breakLine: true } },
  { text: "Rishik · Praneeth · Srikar · Sami · Mani Kanta", options: { color: AMBER } },
], { placeholder: "body" });
s.addNotes("Rishik: introduce the team and the question. We predict at 05:00 on the day of travel whether a New York departure will arrive 15+ minutes late.");

// ---------------------------------------------------------------- 2 verdict (forest plot of test PR AUC with 95% CIs)
s = addSlide("All learned models beat the baselines; none clearly beats the others", "Opening", "Verdict");
const models = [["Majority class", MAJ], ["Planner's rule (flight's 28-day rate)", RULE], ["Logistic regression", LR], ["Gradient boosting (tuned XGBoost)", GB], ["Hybrid (boosting from LR)", HY]];
const X0 = 4.6, X1 = 12.4, lo = 0.2, hi = 0.5, px = (v) => X0 + ((v - lo) / (hi - lo)) * (X1 - X0);
s.addText("Test PR AUC, November–December (used once) · bars = 95% day-block CI", { x: 0.6, y: 1.45, w: 11, h: 0.4, fontSize: 14, color: SLATE, margin: 0, isTextBox: true });
models.forEach(([name, r], i) => {
  const y = 2.2 + i * 0.78, learned = r.ci_low !== null;
  s.addText(name, { x: 0.6, y: y - 0.2, w: 3.8, h: 0.4, fontSize: 15, bold: learned, color: learned ? NAVY : SLATE, align: "right", valign: "middle", margin: 0, isTextBox: true });
  if (learned) s.addShape(pres.shapes.LINE, { x: px(r.ci_low), y, w: px(r.ci_high) - px(r.ci_low), h: 0, line: { color: NAVY, width: 3 } });
  s.addShape(pres.shapes.OVAL, { x: px(r.pr_auc) - 0.12, y: y - 0.12, w: 0.24, h: 0.24, fill: { color: learned ? AMBER : MUTED }, line: { color: learned ? NAVY : MUTED, width: 1.5 } });
  s.addText(f3(r.pr_auc), { x: px(r.pr_auc) - 0.4, y: y - 0.48, w: 0.8, h: 0.3, fontSize: 12, bold: true, color: NAVY, align: "center", margin: 0, isTextBox: true });
});
const axisY = 2.2 + 5 * 0.78 - 0.25;
s.addShape(pres.shapes.LINE, { x: X0, y: axisY, w: X1 - X0, h: 0, line: { color: MUTED, width: 1 } });
for (let v = lo; v <= hi + 1e-9; v += 0.05) {
  s.addShape(pres.shapes.LINE, { x: px(v), y: axisY, w: 0, h: 0.08, line: { color: MUTED, width: 1 } });
  s.addText(v.toFixed(2), { x: px(v) - 0.35, y: axisY + 0.1, w: 0.7, h: 0.3, fontSize: 11, color: MUTED, align: "center", margin: 0, isTextBox: true });
}
s.addShape(pres.shapes.LINE, { x: px(d5.test_base_rate), y: 1.95, w: 0, h: axisY - 1.95, line: { color: MUTED, width: 1, dashType: "dash" } });
s.addText(`random = base rate ${f3(d5.test_base_rate)}`, { x: px(d5.test_base_rate) + 0.08, y: axisY - 0.35, w: 2.6, h: 0.3, fontSize: 10, color: MUTED, margin: 0, isTextBox: true });
s.addText("At 05:00 the useful signal is recent delay history, which a linear model uses as well as boosting does — and transfers better across seasons.",
  { x: 0.6, y: 6.3, w: 12.1, h: 0.55, fontSize: 15, italic: true, color: NAVY, margin: 0, isTextBox: true });
s.addNotes(`Rishik: the answer first. Hybrid ${f3(HY.pr_auc)}, logistic regression ${f3(LR.pr_auc)}, boosting ${f3(GB.pr_auc)}; planner's rule ${f3(RULE.pr_auc)}; random ${f3(MAJ.pr_auc)}. The CIs overlap heavily, so we claim no winner among learned models.`);

// ---------------------------------------------------------------- 3 pipeline
s = addSlide("Five notebooks, one per person, following the course life cycle", "Opening", "Overview");
const steps = [["01", "Rishik", "Problem, data, metric, split", "Phases 1, 2, 5"], ["02", "Praneeth", "EDA and preprocessing", "Phases 3, 4"],
  ["03", "Srikar", "Features, selection, leakage test", "Phase 6"], ["04", "Sami", "Baselines, GD from scratch, model selection", "Phases 7, 8A"],
  ["05", "Mani Kanta", "Boosting: tuning, training, test", "Phases 8B, 8C, 9"]];
steps.forEach(([nb, who, what, ph], i) => {
  const x = 0.6 + i * 2.5;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.9, w: 2.2, h: 2.6, rectRadius: 0.1, fill: { color: TINT }, line: { color: TINT } });
  s.addShape(pres.shapes.OVAL, { x: x + 0.75, y: 2.1, w: 0.7, h: 0.7, fill: { color: AMBER }, line: { color: AMBER } });
  s.addText(nb, { x: x + 0.75, y: 2.1, w: 0.7, h: 0.7, fontSize: 18, bold: true, color: NAVY, align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addText(who, { x: x + 0.1, y: 2.95, w: 2.0, h: 0.4, fontSize: 17, bold: true, color: NAVY, align: "center", margin: 0, isTextBox: true });
  s.addText(what, { x: x + 0.15, y: 3.4, w: 1.9, h: 0.7, fontSize: 13, color: SLATE, align: "center", valign: "top", margin: 0, isTextBox: true });
  s.addText(ph, { x: x + 0.1, y: 4.1, w: 2.0, h: 0.3, fontSize: 11, color: MUTED, align: "center", margin: 0, isTextBox: true });
  if (i < 4) s.addShape(pres.shapes.CHEVRON, { x: x + 2.24, y: 3.05, w: 0.22, h: 0.3, fill: { color: AMBER }, line: { color: AMBER } });
});
s.addText([
  { text: "Each notebook writes its module to src/fdp/ and its numbers to reports/*.json.", options: { breakLine: true, bullet: true } },
  { text: "The report and these slides are generated from those JSON files, so no number is typed by hand.", options: { breakLine: true, bullet: true } },
  { text: `Every decision below is numbered (1–${LOG.length}); the full log with evidence is at the end of the deck.`, options: { bullet: true } },
], { x: 0.6, y: 4.9, w: 12, h: 1.6, fontSize: 16, color: SLATE, paraSpaceAfter: 8, margin: 0, isTextBox: true });
s.addNotes("Rishik: explain the order of work. Each notebook builds on the modules pushed before it.");

// ---------------------------------------------------------------- 4-5 Rishik
pres.addSection({ title: "01 Problem, data, split" });
s = addSlide("A yes/no question asked at 05:00, with only what is known then", "01 Problem, data, split", "01 · Rishik");
const stats1 = [[n(d1.summary.flights), "scheduled flights"], [n(d1.summary.labelled), "labelled flights"], [f3(d1.summary.delayed_share_of_labelled), "share delayed (≥15 min)"]];
stats1.forEach(([v, l], i) => {
  s.addText(v, { x: 0.6, y: 1.6 + i * 1.6, w: 4.4, h: 0.9, fontSize: 48, bold: true, color: NAVY, fontFace: HEAD, margin: 0, isTextBox: true });
  s.addText(l, { x: 0.6, y: 2.45 + i * 1.6, w: 4.4, h: 0.4, fontSize: 14, color: MUTED, margin: 0, isTextBox: true });
});
stack(s, [1, 2, 3], 5.4, 1.5, 7.35);
s.addNotes("Rishik: classification because the decision is yes/no and BTS uses 15 minutes. 05:00 because ~3% of the previous day's flights land after midnight. Cancelled flights have no label; we do not invent one.");

s = addSlide("Metric before model, and a split that follows time", "01 Problem, data, split", "01 · Rishik");
image(s, "01_split_delay_rate.png", 0.6, 1.5, 5.8, 2.6, "Monthly delay rate coloured by split block");
s.addText([{ text: f3(d1.valid_always_on_time_accuracy), options: { fontSize: 40, bold: true, color: NAVY, fontFace: HEAD, breakLine: true } },
  { text: "accuracy of “always on time” in September — and zero delays caught. So: PR AUC, not accuracy.", options: { fontSize: 14, color: SLATE } }],
  { x: 0.6, y: 4.4, w: 5.6, h: 1.6, margin: 0, valign: "top", isTextBox: true });
stack(s, [4, 5, 6], 6.6, 1.5, 6.15);
s.addNotes(`Rishik: the split. Train ${split.train.from}–${split.train.to}, valid September, calibration October, test November–December once. Monthly delay rate moves from ${f3(d1.monthly_delay_rate["9"])} to ${f3(d1.monthly_delay_rate["12"])}: this base-rate shift comes back in every later result.`);

// ---------------------------------------------------------------- 6-7 Praneeth
pres.addSection({ title: "02 EDA, preprocessing" });
s = addSlide("EDA on training rows only: missing data is systematic, outliers are real", "02 EDA, preprocessing", "02 · Praneeth");
image(s, "02_delay_rate_relationships.png", 0.6, 1.45, 6.0, 3.85, "Delay rate by hour, carrier, weekday and origin");
s.addText(`Delay rate climbs from ${f3(d2.delay_rate_by_hour["5"])} at 05:00 to ${f3(d2.delay_rate_by_hour["20"])} at 20:00: delays propagate through the day.`,
  { x: 0.6, y: 5.45, w: 6.0, h: 0.8, fontSize: 14, color: SLATE, margin: 0, isTextBox: true });
stack(s, [8, 9, 10], 6.9, 1.5, 5.85);
s.addNotes(`Praneeth: aircraft year missing ${d2.missing_pct.plane_year}% — MQ ${d2.plane_year_missing_by_carrier_top.MQ}%, AA ${d2.plane_year_missing_by_carrier_top.AA}%. Systematic, so impute + indicator. Keep outliers. No duplicates.`);

s = addSlide("Yesterday matters, last week does not — so we chose the history windows from the ACF", "02 EDA, preprocessing", "02 · Praneeth");
image(s, "02_daily_acf_pacf.png", 0.6, 1.45, 12.1, 2.9, "Daily delay rate, ACF and PACF");
[11, 12, 13].forEach((k, i) => card(s, k, 0.6 + i * 4.1, 4.5, 3.9, 2.35));
s.addNotes(`Praneeth: ACF lag 1 ${d2.acf.lag1}, PACF cuts off after lag 1, lag 7 ${d2.acf.lag7}. So a 1-day window and day-of-week features. Decision 14: one-hot for linear models, ordinal codes for trees, native categoricals in XGBoost.`);

// ---------------------------------------------------------------- 8-9 Srikar
pres.addSection({ title: "03 Features, selection" });
s = addSlide("Five feature groups, all known at 05:00 — and a test that proves it", "03 Features, selection", "03 · Srikar");
const gt = d3.group_table;
s.addChart(pres.charts.BAR, [
  { name: "September", labels: gt.map((r) => "+ " + r.features.split(" + ").pop()), values: gt.map((r) => r.valid_pr_auc) },
  { name: "October", labels: gt.map((r) => "+ " + r.features.split(" + ").pop()), values: gt.map((r) => r.calib_pr_auc) },
], chart({ x: 0.6, y: 1.45, w: 6.2, h: 4.6, barDir: "col", barGrouping: "clustered", chartColors: [NAVY, AMBER], title: "Logistic regression PR AUC as groups are added",
  showLegend: true, legendPos: "b", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0, valAxisMaxVal: 0.35 }));
s.addText(`rate = (k + α·prior) / (n + α),  α = ${d3.alpha}   ·   windows end on day D−1 (.rolling().sum().shift(1))`,
  { x: 0.6, y: 6.2, w: 6.2, h: 0.6, fontSize: 12, color: SLATE, margin: 0, isTextBox: true });
stack(s, [15, 16, 17], 7.1, 1.5, 5.65);
s.addNotes(`Srikar: ${d3.n_numeric} numeric features in five groups. History is the big September jump, but in October calendar and congestion matter more. Leakage tests: ${LOG[1].result}`);

s = addSlide("No selection method beat “all features” on both months, so we kept all", "03 Features, selection", "03 · Srikar");
const dec = d3.decision;
s.addChart(pres.charts.BAR, [
  { name: "September (selection month)", labels: dec.map((r) => r.method.replace(" (24 comps, 95% var)", " (24)").replace("embedded: ", "")), values: dec.map((r) => r.valid) },
  { name: "October (honest check)", labels: dec.map((r) => r.method.replace(" (24 comps, 95% var)", " (24)").replace("embedded: ", "")), values: dec.map((r) => r.calib) },
], chart({ x: 0.6, y: 1.45, w: 7.4, h: 5.3, barDir: "bar", barGrouping: "clustered", chartColors: [NAVY, AMBER], title: "PR AUC by selection method",
  showLegend: true, legendPos: "b", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0, valAxisMaxVal: 0.36, catAxisOrientation: "maxMin" }));
stack(s, [18, 19], 8.3, 1.5, 4.45);
s.addNotes(`Srikar: SFS looked best on September (${f3(d3.sfs.valid_sfs)}) — the month it chose on — and lost on October (${f3(d3.sfs.calib_sfs)} vs ${f3(d3.sfs.calib_all)}). That is selection over-fitting.`);

// ---------------------------------------------------------------- 10-11 Sami
pres.addSection({ title: "04 Baselines, model selection" });
s = addSlide("The floor: a planner's rule, and logistic regression written from scratch", "04 Baselines, model selection", "04 · Sami");
const bl = d4.baselines, blNames = { majority_class: "Majority class", rule_flight_28d_rate: "Flight's 28-day rate", rule_scheduled_hour: "Scheduled hour" };
s.addChart(pres.charts.BAR, [
  { name: "September", labels: Object.values(blNames), values: Object.keys(blNames).map((k) => bl.valid[k]) },
  { name: "October", labels: Object.values(blNames), values: Object.keys(blNames).map((k) => bl.calib[k]) },
], chart({ x: 0.6, y: 1.45, w: 5.0, h: 2.7, barDir: "col", barGrouping: "clustered", chartColors: [NAVY, AMBER], title: "Baseline PR AUC",
  showLegend: true, legendPos: "b", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0, valAxisMaxVal: 0.3 }));
image(s, "04_learning_rate.png", 0.6, 4.25, 5.0, 2.65, "Training log loss by learning rate");
stack(s, [20, 21, 22], 5.9, 1.5, 6.85);
s.addNotes("Sami: baselines first. Gradient = Xᵀ(σ(Xw) − y)/n + λw. Batch GD matches scikit-learn to the fourth decimal. Learning rate 8 diverges. Decision 23: C = 0.001 chosen on September only.");

s = addSlide("Model families on rolling months: logistic regression led, untuned boosting was third", "04 Baselines, model selection", "04 · Sami");
const ms = d4.model_selection;
s.addChart(pres.charts.BAR, ["Jul", "Aug", "Sep"].map((m) => ({ name: m, labels: ms.map((r) => r.model.replace(" (XGBoost, untuned)", " (untuned)")), values: ms.map((r) => r[m]) })),
  chart({ x: 0.6, y: 1.45, w: 7.2, h: 3.6, barDir: "col", barGrouping: "clustered", chartColors: [NAVY, "5B7BA6", AMBER], title: "PR AUC per rolling month (fixed settings)",
    showLegend: true, legendPos: "b", showValue: false, valAxisMinVal: 0, valAxisMaxVal: 0.6 }));
table(s, ["Model", "Mean", "Fit s"], ms.map((r) => [r.model.replace(" (XGBoost, untuned)", " (untuned)"), f3(r.mean_pr_auc), r.fit_seconds.toFixed(1)]),
  0.6, 5.2, 7.2, [4.6, 1.3, 1.3], { fontSize: 12, rowH: 0.3 });
stack(s, [24, 25], 8.1, 1.5, 4.65);
s.addNotes("Sami: a deep tree memorises (train 1.0). On three rolling months LR is best; untuned XGBoost loses to it — so boosting has to earn its place through honest tuning in notebook 05.");

// ---------------------------------------------------------------- 12-16 Mani Kanta
pres.addSection({ title: "05 Gradient boosting, evaluation" });
s = addSlide("We found an early-stopping trap that inflates boosting's score", "05 Gradient boosting, evaluation", "05 · Mani Kanta");
const trap = d5.early_stopping_trap, opt = trap["early-stop on the scored month (optimistic)"], hon = trap["our protocol (honest)"];
s.addChart(pres.charts.BAR, [
  { name: "Early-stop on the scored month", labels: ["Jul", "Aug", "Sep", "Mean"], values: ["Jul", "Aug", "Sep", "mean"].map((k) => opt[k]) },
  { name: "Honest: stop on m−1, refit, score m", labels: ["Jul", "Aug", "Sep", "Mean"], values: ["Jul", "Aug", "Sep", "mean"].map((k) => hon[k]) },
], chart({ x: 0.6, y: 1.45, w: 6.6, h: 4.6, barDir: "col", barGrouping: "clustered", chartColors: [MUTED, NAVY], title: "Same XGBoost configuration, two protocols (PR AUC)",
  showLegend: true, legendPos: "b", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0, valAxisMaxVal: 0.65 }));
s.addText(`+${f3(opt.mean - hon.mean)} of the optimistic mean came from the stopping rule seeing the answers.`,
  { x: 0.6, y: 6.2, w: 6.6, h: 0.6, fontSize: 14, italic: true, color: NAVY, margin: 0, isTextBox: true });
stack(s, [26, 27], 7.5, 1.5, 5.25);
s.addNotes("Mani Kanta: boosting fits trees to r = y − p. Our from-scratch version equals scikit-learn. Then the trap: early-stopping on the month you report is optimistic. All tuning uses the honest protocol.");

s = addSlide("Tuned boosting stayed below logistic regression; the hybrid caught up", "05 Gradient boosting, evaluation", "05 · Mani Kanta");
const cv = d5.cv_compare;
table(s, ["Rolling PR AUC", "Jul", "Aug", "Sep", "Mean"], Object.entries(cv).map(([m, r]) => [m, ...["Jul", "Aug", "Sep", "mean"].map((k) => f3(r[k]))]),
  0.6, 1.55, 6.6, [3.0, 0.9, 0.9, 0.9, 0.9], { fontSize: 14, rowH: 0.42 });
const tgb = JSON.parse(d5.search[0].trees_per_fold);
s.addText("Trees chosen by early stopping, per fold", { x: 0.6, y: 3.45, w: 6.6, h: 0.35, fontSize: 14, bold: true, color: NAVY, margin: 0, isTextBox: true });
[["Boosting", tgb, d5.n_trees_final], ["Hybrid", d5.hybrid_trees_per_fold, d5.n_trees_hybrid]].forEach(([name, folds, med], i) => {
  const y = 3.95 + i * 0.95;
  s.addText(name, { x: 0.6, y, w: 1.3, h: 0.6, fontSize: 14, color: SLATE, valign: "middle", margin: 0, isTextBox: true });
  folds.forEach((t, j) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 1.95 + j * 1.25, y, w: 1.1, h: 0.6, rectRadius: 0.08, fill: { color: TINT }, line: { color: TINT } });
    s.addText(String(t), { x: 1.95 + j * 1.25, y, w: 1.1, h: 0.6, fontSize: 18, bold: true, color: NAVY, align: "center", valign: "middle", margin: 0, isTextBox: true });
  });
  s.addText(`median → ${med}`, { x: 5.8, y, w: 1.5, h: 0.6, fontSize: 15, bold: true, color: FIND, valign: "middle", margin: 0, isTextBox: true });
});
s.addText(`September alone would stop at ${d5.sept_only_trees} tree.`, { x: 0.6, y: 5.95, w: 6.6, h: 0.4, fontSize: 13, italic: true, color: SLATE, margin: 0, isTextBox: true });
stack(s, [28, 29, 30], 7.5, 1.5, 5.25);
s.addNotes(`Mani Kanta: random search over ${d5.search.length} configurations; best lr ${d5.best_params.learning_rate}, depth ${d5.best_params.max_depth}. Shallow trees win. The tree count swings between folds — the structure that transfers is not stable. The hybrid starts from LR's logit and adds corrections.`);

s = addSlide("The final exam: test months scored once", "05 Gradient boosting, evaluation", "05 · Mani Kanta");
table(s, ["Model", "PR AUC", "95% CI", "ROC AUC", "Lift", "ECE raw → Platt"],
  d5.test_table.map((r) => [r.model.replace("Rule: flight's 28-day delay rate", "Rule: flight's 28-day rate"), f3(r.pr_auc), r.ci_low === null ? "—" : `[${f3(r.ci_low)}, ${f3(r.ci_high)}]`,
    f3(r.roc_auc), r.lift.toFixed(2), r.ece_after_platt === null ? f3(r.ece) : `${f3(r.ece)} → ${f3(r.ece_after_platt)}`]),
  0.6, 1.5, 12.1, [3.7, 1.4, 2.2, 1.4, 1.1, 2.3], { fontSize: 14, rowH: 0.38 });
image(s, "05_test_pr_reliability.png", 0.6, 4.05, 6.0, 2.85, "Test precision-recall curves and reliability diagram");
stack(s, [32], 6.8, 4.15, 5.95, 6.95);
s.addNotes("Mani Kanta: compare PR AUC and the intervals together. They overlap heavily: no winner. Platt helped boosting's calibration but slightly hurt LR and the hybrid, which were already calibrated.");

s = addSlide("For planners: a flag list where about half the flags are real delays", "05 Gradient boosting, evaluation", "05 · Mani Kanta");
const biz = d5.business;
table(s, ["Model", "Rule", "Precision", "Recall", "Flagged / 1000", "Really delayed", "Random list"],
  biz.map((r) => [r.model.replace(" (boosting from LR)", ""), r.rule, f3(r.precision), f3(r.recall), r["flagged per 1000"].toFixed(1), r["truly delayed among them"].toFixed(1), r["random list would catch"].toFixed(1)]),
  0.6, 1.5, 12.1, [2.6, 2.6, 1.2, 1.1, 1.6, 1.6, 1.4], { fontSize: 13, rowH: 0.36 });
card(s, 33, 0.6, 4.3, 6.0, 2.55);
card(s, 31, 6.8, 4.3, 5.95, 2.55);
s.addNotes(`Mani Kanta: the delay rate rose from ${f3(split.calib.delay_rate)} in October to ${f3(split.test.delay_rate)} in the test months, so a score cut-off fixed in October lands in a different place. Out of every 1,000 flights the model hands over a few dozen, about half really late — roughly twice what a random list of the same size would catch.`);

s = addSlide("Why boosting gains little: tree thresholds learned in summer drift in autumn", "05 Gradient boosting, evaluation", "05 · Mani Kanta");
image(s, "05_importance.png", 0.6, 1.45, 6.6, 2.5, "XGBoost gain and permutation importance on October");
const abl = d5.ablation_october;
s.addChart(pres.charts.LINE, [
  { name: "Gradient boosting (nb 05)", labels: abl.map((r) => r.features), values: abl.map((r) => r.october_pr_auc) },
  { name: "Logistic regression (nb 03)", labels: abl.map((r) => r.features), values: d3.group_table.map((r) => r.calib_pr_auc) },
], chart({ x: 0.6, y: 4.0, w: 6.6, h: 2.9, chartColors: [AMBER, NAVY], title: "October PR AUC as feature groups are added", lineSize: 2, lineDataSymbolSize: 7,
  showLegend: true, legendPos: "r", valAxisMinVal: 0.2, valAxisMaxVal: 0.35, valAxisLabelFormatCode: "0.00" }));
stack(s, [34, 35], 7.5, 1.5, 5.25);
s.addNotes("Mani Kanta: on October, adding congestion and then history lowers boosting while LR keeps gaining. Trees split on absolute rate levels from a high-delay summer. Note: the two lines use different training windows (LR Jan–Aug, boosting Jan–Sep).");

// ---------------------------------------------------------------- 17 scorecard
pres.addSection({ title: "Decisions and outcomes" });
s = addSlide(`${LOG.length} decisions: what worked, what we found, what we would revisit`, "Decisions and outcomes", "All");
const groups = [["Worked / verified", GOOD, LOG.map((e, i) => [i + 1, e]).filter(([, e]) => verdictColor(e.verdict) === GOOD)],
  ["Findings", FIND, LOG.map((e, i) => [i + 1, e]).filter(([, e]) => verdictColor(e.verdict) === FIND)],
  ["Ties / neutral", NEUTRAL, LOG.map((e, i) => [i + 1, e]).filter(([, e]) => verdictColor(e.verdict) === NEUTRAL)]];
let gx = 0.6;
const gw = [5.4, 3.6, 2.9];
groups.forEach(([label, color, items], gi) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: gx, y: 1.5, w: gw[gi], h: 0.5, rectRadius: 0.08, fill: { color }, line: { color } });
  s.addText(`${label} · ${items.length}`, { x: gx, y: 1.5, w: gw[gi], h: 0.5, fontSize: 15, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0, isTextBox: true });
  const cols = gi === 0 ? 2 : 1, cw = (gw[gi] - (cols - 1) * 0.15) / cols;
  items.forEach(([k, e], j) => {
    const col = Math.floor(j / Math.ceil(items.length / cols)), row = j % Math.ceil(items.length / cols);
    s.addText([{ text: `${k}  `, options: { bold: true, color } }, { text: e.decision.length > 40 ? e.decision.slice(0, 38).replace(/\s+\S*$/, "") + " …" : e.decision }],
      { x: gx + col * (cw + 0.15), y: 2.15 + row * 0.27, w: cw, h: 0.27, fontSize: 10, color: SLATE, margin: 0, valign: "middle", isTextBox: true });
  });
  gx += gw[gi] + 0.2;
});
s.addNotes("All: one line per decision, grouped by outcome. The full log with evidence follows.");

// ---------------------------------------------------------------- 18-21 full decision log
const PER = 9;
for (let p = 0; p * PER < LOG.length; p++) {
  const rows = LOG.slice(p * PER, (p + 1) * PER);
  s = addSlide(`Decision log (${p * PER + 1}–${p * PER + rows.length} of ${LOG.length})`, "Decisions and outcomes", "Appendix");
  table(s, ["#", "NB", "Decision", "What happened", "Verdict"],
    rows.map((e, i) => [String(p * PER + i + 1), e.notebook, { text: e.decision, options: { bold: true, color: NAVY, fill: { color: i % 2 ? WHITE : TINT } } }, e.result,
      { text: e.verdict, options: { bold: true, color: verdictColor(e.verdict), fill: { color: i % 2 ? WHITE : TINT } } }]),
    0.6, 1.4, 12.1, [0.4, 0.5, 3.3, 6.6, 1.3], { fontSize: 11, valign: "top" });
  s.addNotes("Read-ahead slide: every decision with the evidence from the saved results. The report's appendix also gives the reason for each.");
}

// ---------------------------------------------------------------- caveats, conclusion, contributions
s = addSlide("What we would flag before anyone else does", "Decisions and outcomes", "All");
const cvLR = d5.cv_compare["Logistic regression"].Sep, cv1 = d4.model_selection.find((r) => r.model === "Logistic regression").Sep;
const caveats = [
  ["Small selection optimism", `C = ${d4.best_C} was picked on September and reused in the rolling comparison whose September fold is the same month (${f3(cvLR)} vs ${f3(cv1)} with C = 1). Boosting's configuration was also picked on the folds it is reported on. The test set is unaffected.`],
  ["Small search", `${d5.search.length} configurations were tried; their spread was only ${f3(d5.search[0].mean_pr_auc - d5.search[d5.search.length - 1].mean_pr_auc)}, so a larger search is unlikely to close a ${f3(cv["Logistic regression"].mean - cv["Gradient boosting (tuned)"].mean)} gap, but this is untested.`],
  ["Unstable hybrid fold", `The hybrid's August fold stopped at ${d5.hybrid_trees_per_fold[1]} trees against ${d5.hybrid_trees_per_fold[0]} and ${d5.hybrid_trees_per_fold[2]}; the median rule absorbs it.`],
  ["One month, one year", "Ablation uses October only and a fixed tree count; one year of data cannot separate season from trend."],
];
caveats.forEach(([h, t], i) => {
  const x = 0.6 + (i % 2) * 6.15, y = 1.5 + Math.floor(i / 2) * 2.65;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 5.95, h: 2.45, rectRadius: 0.08, fill: { color: TINT }, line: { color: TINT } });
  s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: y + 0.2, w: 0.45, h: 0.45, fill: { color: FIND }, line: { color: FIND } });
  s.addText("!", { x: x + 0.2, y: y + 0.2, w: 0.45, h: 0.45, fontSize: 16, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addText(h, { x: x + 0.8, y: y + 0.2, w: 4.9, h: 0.45, fontSize: 17, bold: true, color: NAVY, valign: "middle", margin: 0, isTextBox: true });
  s.addText(t, { x: x + 0.8, y: y + 0.8, w: 4.9, h: 1.5, fontSize: 14, color: SLATE, valign: "top", margin: 0, isTextBox: true });
});
s.addNotes("Rishik: honest limitations. None of them touches the one-time test result.");

pres.addSection({ title: "Close" });
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Close" });
s.addText("At 05:00, boosting does not beat a well-tuned linear model", { placeholder: "title" });
s.addText([
  { text: `All learned models beat the planner's rule (${f3(RULE.pr_auc)}) and random (${f3(MAJ.pr_auc)}); their test intervals overlap.`, options: { bullet: true, breakLine: true } },
  { text: "Honest tuning mattered: the early-stopping shortcut would have overstated boosting.", options: { bullet: true, breakLine: true } },
  { text: "Phase 2: drift-robust (relative) history features, weather forecasts known at 05:00, more years of data, temporal deep learning.", options: { bullet: true } },
], { placeholder: "body", fontSize: 18, paraSpaceAfter: 8 });
s.addNotes("Rishik: conclusion and Phase 2 plan.");

s = addSlide("Contributions", "Close", "All");
const people = [["Rishik Chowdary Karuturi", "2401010381", "01 · Problem, data, metrics, split; report and slides"],
  ["Nakkina Lakshmi Praneeth", "2401010288", "02 · EDA, preprocessing"], ["Balanagu Sesha Srikar", "2401010124", "03 · Features, selection, leakage test"],
  ["Shaik Junaid Sami", "2401010435", "04 · Baselines, gradient descent, model selection"], ["Siraparapu Shiva Sankar Mani Kanta", "2401010457", "05 · Gradient boosting, tuning, evaluation"]];
people.forEach(([name, id, what], i) => {
  const y = 1.55 + i * 1.02;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 12.1, h: 0.85, rectRadius: 0.08, fill: { color: TINT }, line: { color: TINT } });
  s.addShape(pres.shapes.OVAL, { x: 0.8, y: y + 0.15, w: 0.55, h: 0.55, fill: { color: AMBER }, line: { color: AMBER } });
  s.addText(`0${i + 1}`, { x: 0.8, y: y + 0.15, w: 0.55, h: 0.55, fontSize: 14, bold: true, color: NAVY, align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addText(name, { x: 1.6, y, w: 4.6, h: 0.85, fontSize: 17, bold: true, color: NAVY, valign: "middle", margin: 0, isTextBox: true });
  s.addText(id, { x: 6.2, y, w: 1.6, h: 0.85, fontSize: 13, color: MUTED, valign: "middle", margin: 0, isTextBox: true });
  s.addText(what, { x: 7.8, y, w: 4.8, h: 0.85, fontSize: 14, color: SLATE, valign: "middle", margin: 0, isTextBox: true });
});
s.addNotes("All: each person's work is in their own commits.");

pres.writeFile({ fileName: path.join(__dirname, "T107_Phase1_Slides.pptx") }).then((f) => console.log("wrote", f));
