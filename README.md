# Student Success Predictor 🎓

**A comparative study of machine learning algorithms for early academic risk
prediction** — university project for the *Algorithms for Data Science* course.

An interactive Streamlit application plus a reproducible experiment pipeline
that classifies students of the UCI **Student Performance** dataset
(Mathematics) into two academic-risk categories using attributes available
*before* the prediction time — **without using any exam grades**.

> ⚠️ **Educational prototype.** Predictions are statistical estimates with
> real uncertainty. They must not be used for real academic decisions or
> judged as certain outcomes about a student.

---

## 1. Project overview

- **Research question:** How do Logistic Regression, Decision Tree and
  Random Forest compare when predicting an explicitly defined academic risk
  category from student attributes available before the prediction time?
- **Target (project-defined rule):** `at_risk` if final grade **G3 < 10**,
  else `not_at_risk` (positive class = `at_risk`).
- **Leakage policy:** `G3` is never a feature (it defines the target) and
  `G1`/`G2` are excluded in the *primary early-prediction experiment*.
- **Algorithms compared:** Logistic Regression · Decision Tree ·
  Random Forest (all with `class_weight="balanced"`).
- **Reproducibility:** stratified 80/20 split, `random_state=42`,
  5-fold cross-validated grid search **on training data only**.

Measured results are stored in `results/metrics/` and consumed by the
application — the app does **not** hardcode any experimental numbers.

## 2. Dataset

| Property | Value |
|---|---|
| Source | [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) |
| Files | `student-mat.csv` (395 students, Mathematics — used here), `student-por.csv` (649, kept for extension) |
| Target column | G3 (final grade, 0–20) |
| Missing values | 0 |
| License | CC BY 4.0 |

**Primary reference:** Cortez, P., & Silva, A. M. G. (2008). *Using data
mining to predict secondary school student performance.* In Proceedings of
5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) (pp. 5–12).
Dataset DOI: 10.24432/C5TG7T.

## 3. Prerequisites

- Windows 11 (or any OS with Python)
- **Python 3.11+** (developed and tested on 3.13)
- VS Code (recommended) with the Python extension

## 4. Setup (PowerShell)

```powershell
cd student-success-predictor

# 1. Create & activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# If scripts are blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 2. Install dependencies
python -m pip install -r requirements.txt
```

## 5. Data preparation

The training script **auto-downloads** the official dataset from UCI on
first run (internet permitting). To do it manually:

1. Open https://archive.ics.uci.edu/dataset/320/student+performance
2. Download the ZIP and extract it.
3. Copy `student-mat.csv` (and optionally `student-por.csv`) into `data/raw/`.

No data is ever uploaded or fabricated — if the download fails, the pipeline
stops with clear instructions.

## 6. Train the models

```powershell
python -m src.train
```

What this does (in order):
1. Loads + validates `data/raw/student-mat.csv`.
2. Builds the target (`at_risk = G3 < 10`) and audits excluded columns.
3. Stratified 80/20 split (seed 42).
4. Tunes the three pipelines with 5-fold CV **on training data only**.
5. Evaluates on the held-out test split; the best model by F1 (currently
   Random Forest) is saved as `models/best_model.joblib`.
6. Writes metrics, metadata, figures, processed data and EDA summary.

Optional experiment (documented, clearly separated):

```powershell
python -m src.train --with-grades   # G1/G2 included - NOT comparable
```

## 7. Run the application

```powershell
streamlit run app.py
```

Then open http://localhost:8501. Pages: **Home**, **Student Risk
Assessment**, **Algorithm Comparison**, **Data Insights**, **Research &
Methodology**. All numbers/figures are loaded from generated artifacts.

### Sharing the app as a public link (free, 2 minutes)

Push this folder to GitHub, then deploy on **Streamlit Community Cloud**:

1. Create a GitHub repo and push the project:
   ```powershell
   git init
   git add .
   git commit -m "Student Success Predictor"
   git remote add origin https://github.com/<your-username>/student-success-predictor.git
   git push -u origin main
   ```
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app → Deploy a public app from GitHub**:
   - Repository: `<your-username>/student-success-predictor`
   - Branch: `main`
   - Main file path: `app.py`
4. Click **Deploy** → you get a permanent public URL like
   `https://<your-username>-student-success-predictor-app.streamlit.app`.

Notes for the cloud app:
- The dataset downloads automatically from the official UCI source on the
  first cloud run (internet access is allowed on Streamlit Cloud).
- Models are NOT included in Git (`.gitignore` excludes `models/*.joblib`),
  so after deployment run training once — or add a `setup` step: the app
  will show clear "artifacts not found" instructions, and the recommended
  flow is to commit `models/best_model.joblib` only if your team accepts a
  one-off small binary (justify it in the repo README), or retrain on
  Streamlit Cloud by running the training locally and pushing artifacts.

Fastest reliable flow for the cloud: **commit `models/best_model.joblib`,
`results/metrics/*.json` and `results/figures/*.png`** (remove the
`models/*.joblib` line from `.gitignore` if you choose this route). These
are small files (<1 MB) generated by your own experiments, so no fake
data is involved.

## 8. Run the tests

```powershell
python -m pytest
```

30 automated tests cover: missing-file handling, schema & target rules,
**leakage prevention** (G1/G2/G3 never leaks into primary features), split
stratification & reproducibility, model training, metrics consistency,
artifact loading, input validation and prediction.

## 9. Output artifact locations

| Artifact | Path |
|---|---|
| Trained pipeline (joblib) | `models/best_model.joblib` (+ `*_with_grades` variant) |
| Metadata (schema, seed, config) | `results/metrics/model_metadata.json` |
| Metrics (JSON + CSV) | `results/metrics/model_metrics.{json,csv}` |
| EDA summary | `results/metrics/eda_summary.json` |
| Figures (PNG) | `results/figures/*.png` |
| Processed dataset | `data/processed/student_mat_processed.csv` |
| Research paper / presentation | `research/research_paper.docx`, `research/presentation.pptx` |

## 10. Troubleshooting

- **`python: command not found`** → use `py -m ...` or the full path of
  the interpreter.
- **PowerShell script blocked** → `Set-ExecutionPolicy -Scope Process
  -ExecutionPolicy Bypass`.
- **Dataset missing / download fails** → follow §5 (file goes to
  `data/raw/`).
- **App says "artifacts not found"** → run `python -m src.train`, then
  restart the app.
- **Different results after retraining?** Check the printed random seed and
  library versions in `results/metrics/model_metadata.json` — everything
  except exact float boundaries should match with the same seed/versions.

## 11. Limitations & responsible use

- Small dataset (395 students, 2 Portuguese schools, one subject, 2008).
- The at-risk threshold (G3 < 10) is a **project-defined operational rule**,
  not a universal definition of academic risk.
- Early prediction with no grades is intentionally difficult — measured
  metrics are modest and must be reported honestly.
- The study evaluates **predictive association, not causation**; feature
  importance is not proof that a factor *causes* failure.
- Do not use this tool for real academic decisions about students.

## 12. Project structure

```
student-success-predictor/
├── app.py                    # Streamlit application (5 pages)
├── requirements.txt          # pinned, environment-verified versions
├── README.md                 # this file
├── STUDY_GUIDE_AR.md         # beginner-friendly Arabic study guide
├── DEFENSE_QA_AR.md          # 25+ oral-defense Q&A in Egyptian Arabic
├── TEAM_TASKS.md             # two-member work division
├── .gitignore
├── data/
│   ├── raw/                  # official UCI files (downloaded)
│   └── processed/            # target-labelled dataset (generated)
├── models/                   # fitted pipelines (generated)
├── results/
│   ├── metrics/              # metrics + metadata + EDA (generated)
│   └── figures/              # PNG charts (generated)
├── research/
│   ├── research_paper.docx   # English research paper (generated)
│   ├── presentation.pptx     # English slides (generated)
│   ├── references.md         # verified bibliography
│   └── verify_references.py  # Crossref-citation verification helpers
├── src/
│   ├── config.py             # single source of truth for definitions
│   ├── data_loader.py        # loading, validation, download, leakage audit
│   ├── preprocessing.py      # ColumnTransformer + pipelines
│   ├── train.py              # training, tuning, artifact persistence
│   ├── evaluate.py           # metrics + figures (+ permutation importance)
│   └── predict.py            # artifact loading + single-student prediction
├── tests/
│   ├── test_preprocessing.py
│   ├── test_training.py
│   └── test_prediction.py
└── .streamlit/config.toml    # app theme
```

## 13. Documentation for the team

- `STUDY_GUIDE_AR.md` — شرح المشروع بالكامل بالعربي المصري، أكاديمي
- `DEFENSE_QA_AR.md` — أسئلة الدكتور المتوقعة + إجابات جاهزة
- `TEAM_TASKS.md` — توزيع الشغل بين الطالبين
- `research/references.md` — المراجع المُوثّقة (كله حقيقي ومتحقق منه)
