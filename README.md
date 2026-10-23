# T107_Flight_Delay_Prediction

## How to run
1. Open the notebooks in `notebooks/` in order (01 → 05), in Colab or locally.
2. `pip install -r requirements.txt` (Colab already has most of these).
3. Put the dataset in `data/`. It is gitignored and never committed.
4. Run tests: `pytest tests/`
5. Build the report: `python report/build_report.py`

## Folder map
```
├── README.md                     how to run, folder map, decisions
├── requirements.txt              Colab-compatible
├── .gitignore                    data/ never committed
├── notebooks/
│   ├── 01_problem_data_split.ipynb
│   ├── 02_eda_preprocessing.ipynb
│   ├── 03_feature_engineering_selection.ipynb
│   ├── 04_baselines_model_selection.ipynb
│   └── 05_gradient_boosting_evaluation.ipynb
├── src/fdp/                      written by the notebooks (%%writefile)
│   ├── __init__.py
│   ├── config.py  data.py  metrics.py
│   ├── preprocess.py
│   ├── features.py  selection.py
│   ├── baselines.py
│   └── boosting.py
├── tests/test_leakage.py
├── reports/
│   ├── 01_problem_data.json … 05_boosting_evaluation.json
│   └── figures/0N_*.png
├── report/
│   ├── main.tex.tmpl  build_report.py  main.tex     IEEE report (numbers from reports/)
└── docs/T107_Phase1_Guide.html
```

## Decisions
- `data/` is never committed.
- `src/fdp/` modules are written from the notebooks with `%%writefile`, so the notebooks are the source of truth.
- Report numbers come only from `reports/*.json`, never typed by hand.
