# T107_Flight_Delay_Prediction

## How to run
1. Open the notebooks in `notebooks/` in order (01 → 10), in Colab or locally. Notebooks 08 and 09 are slow on a CPU (use a Colab GPU runtime, or tick `QUICK_RUN` for a check).
2. `pip install -r requirements.txt` (Colab already has most of these).
3. The dataset downloads itself into `data/raw/` on the first run (`fdp.data.download_raw()`). It is gitignored and never committed.
4. Run tests: `pytest tests/`
5. Build the report: `python report/build_report.py` (writes `report/main.tex`; compile it in Overleaf or with `tectonic report/main.tex`)

## Local setup (one environment for everyone)
```
python3.12 -m venv .venv          # 3.12 = Colab's version
.venv/bin/pip install -r requirements.txt
```
Pick `.venv` as the notebook kernel. Notebooks are committed with the generic `python3`
kernelspec; don't commit a personal kernel name. On macOS XGBoost also needs `brew install libomp`.
`requirements-lock.txt` lists the exact package versions used for the committed results.

## Folder map
```
├── README.md                     how to run, folder map, decisions
├── requirements.txt              Colab-compatible
├── requirements-lock.txt         exact versions of the final run
├── .gitignore                    data/ never committed
├── notebooks/
│   ├── 01_problem_data_split.ipynb
│   ├── 02_eda_preprocessing.ipynb
│   ├── 03_feature_engineering_selection.ipynb
│   ├── 04_baselines_model_selection.ipynb
│   ├── 05_gradient_boosting_evaluation.ipynb
│   ├── 06_dl_dataset_sequences_Srikar.ipynb          Phase 2
│   ├── 07_mlp_embeddings_ManiKanta.ipynb
│   ├── 08_temporal_hybrid_Rishik.ipynb
│   ├── 09_ablation_calibration_Praneeth.ipynb
│   └── 10_final_evaluation_rescheduling_Sami.ipynb
├── src/fdp/                      written by the notebooks (%%writefile)
│   ├── __init__.py
│   ├── config.py  data.py  metrics.py
│   ├── preprocess.py
│   ├── features.py  selection.py
│   ├── baselines.py
│   ├── boosting.py
│   ├── dl_inputs.py  timeseries.py               Phase 2
│   ├── deep.py  phase1_models.py
│   └── calibration.py  rescheduling.py
├── tests/test_leakage.py
├── reports/
│   ├── 01_problem_data.json … 10_final_rescheduling.json
│   └── figures/0N_*.png
└── report/
    └── main.tex.tmpl  build_report.py  main.tex  main.pdf     IEEE report (numbers from reports/)
```

## Decisions
- `data/` is never committed.
- `src/fdp/` modules are written from the notebooks with `%%writefile`, so the notebooks are the source of truth.
- Report numbers come only from `reports/*.json`, never typed by hand.

## Phase 2 (Deep Learning)
**Question, unchanged:** same label, same 05:00 decision time, same split, test months still scored once.
**Added:** a compact MLP with learned embeddings (carrier, origin, destination), a temporal model (GRU over the last 14 days of ordered delay outcomes), an ML + DL hybrid (logistic regression + a temporal network that learns only a correction), ablations, temperature/Platt calibration, and an offline what-if rescheduling analysis.

| # | Notebook | Owner | Phase 2 rubric | Module(s) it writes |
|---|---|---|---|---|
| 06 | `06_dl_dataset_sequences_Srikar.ipynb` | Balanagu Sesha Srikar | DL dataset, leakage validation | `dl_inputs.py`, `timeseries.py`, one new test in `tests/test_leakage.py` |
| 07 | `07_mlp_embeddings_ManiKanta.ipynb` | Siraparapu Shiva Sankar Mani Kanta | architecture, regularisation | `deep.py` |
| 08 | `08_temporal_hybrid_Rishik.ipynb` | Rishik Chowdary Karuturi | temporal model, hybrid, model selection | `phase1_models.py` |
| 09 | `09_ablation_calibration_Praneeth.ipynb` | Nakkina Lakshmi Praneeth | ablation, calibration | `calibration.py` |
| 10 | `10_final_evaluation_rescheduling_Sami.ipynb` | Shaik Junaid Sami | final test, operational metrics, what-if | `rescheduling.py` |

Each notebook reads the modules and `reports/*.json` written before it, so run 06 → 10 in order.
