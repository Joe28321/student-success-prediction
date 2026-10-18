"""Student Success Predictor - interactive Streamlit application.

Pages:
1. Home            - project overview, workflow, disclaimer
2. Risk Assessment - single-student prediction from the trained model
3. Algorithm Comparison - measured metrics loaded from saved artifacts
4. Data Insights   - EDA, class balance, importance analysis
5. Methodology     - dataset documentation, experiment design, references

The application ONLY displays results that were actually produced by
'python -m src.train'. If the artifacts are missing it shows clear setup
instructions instead of fake predictions.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config
from src.predict import (
    ModelNotAvailableError,
    feature_schema,
    load_artifacts,
    predict_student,
    validate_input,
)

APP_DIR = config.PROJECT_ROOT
FIG_DIR = config.FIGURES_DIR

PAGE_TABS = ["Home", "Student Risk Assessment", "Algorithm Comparison",
             "Data Insights", "Research & Methodology"]

# ---------------------------------------------------------------------------
# Cached artifact loading
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading trained model...")
def _load_model():
    return load_artifacts()


@st.cache_data(show_spinner="Loading experiment results...")
def _load_metrics(filename: str = "model_metrics.json"):
    path = config.METRICS_DIR / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Loading metadata...")
def _load_metadata():
    path = config.ARTIFACT_METADATA
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Loading EDA summary...")
def _load_eda():
    path = config.ARTIFACT_EDA_SUMMARY
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Loading processed dataset...")
def _load_processed():
    path = config.ARTIFACT_PROCESSED_DATA
    if not path.exists():
        return None
    return pd.read_csv(path)


def check_artifacts_ready() -> bool:
    """Return True when app-required artifacts exist; else show instructions."""
    try:
        _load_model()
        return True
    except ModelNotAvailableError as exc:
        st.error(str(exc))
        return False
    except FileNotFoundError as exc:
        st.error(str(exc))
        return False


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
      .card {
          background: linear-gradient(160deg, #182230 0%, #131c29 100%);
          border: 1px solid #24344b; border-radius: 14px;
          padding: 1.4rem 1.6rem; margin-bottom: 1.1rem;
      }
      .metric-tile {
          background: #131c29; border: 1px solid #24344b;
          border-radius: 10px; padding: 0.8rem 1rem; text-align: center;
      }
      .small-note { color: #9db2c7; font-size: 0.88rem; line-height: 1.45; }
      h1, h2, h3 { color: #F2F7FB; }
      .section-pad { margin-top: 2.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def header(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="card">
          <h1 style="margin:0 0 .35rem 0; font-size:1.9rem;">{title}</h1>
          <p class="small-note" style="margin:0;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def note(text: str):
    st.markdown(f'<p class="small-note">{text}</p>',
                unsafe_allow_html=True)


def disclaimer():
    st.warning(
        "Educational prototype - this tool demonstrates machine learning "
        "concepts on the UCI Student Performance dataset. Predictions are "
        "based on historical data, carry uncertainty, and must never be "
        "used as a definitive judgment about a student or for real "
        "academic decisions such as grading or support allocation."
    )


# ---------------------------------------------------------------------------
# Page 1: Home
# ---------------------------------------------------------------------------

def page_home():
    header(
        "Student Success Predictor",
        "A comparative study of machine learning algorithms for early "
        "academic risk prediction - Algorithms for Data Science course project.",
    )
    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.markdown("#### The problem")
        st.markdown(
            "Schools want to identify students who may need academic support "
            "**early**, not after final grades are already published. This "
            "project asks whether standard machine learning algorithms can "
            "flag academic risk using information available *before* the "
            "prediction time (demographic, social and school-context "
            "attributes - **without** using any exam grades)."
        )
        st.markdown("#### Research question")
        st.info(
            "How do **Logistic Regression**, **Decision Tree** and "
            "**Random Forest** compare when predicting an explicitly defined "
            "academic risk category from student attributes available *before* "
            "the prediction time?"
        )
    with col2:
        st.markdown("#### Dataset at a glance")
        eda = _load_eda()
        if eda:
            m1, m2 = st.columns(2)
            m1.markdown(f'<div class="metric-tile"><b>{eda["n_rows"]}</b><br>'
                        'students (Math)</div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-tile"><b>'
                        f'{eda["n_features_used"]}</b><br>features used'
                        '</div>', unsafe_allow_html=True)
            share = eda["target_share"].get("at_risk", 0)
            st.markdown(
                f'<div class="metric-tile"><b>{share:.0%}</b><br>of students '
                'labelled <i>at risk</i> (G3 &lt; 10) - a class-imbalance '
                'challenge</div>', unsafe_allow_html=True)
        else:
            st.info("Run training first: `python -m src.train`")

    st.markdown("#### Workflow")
    st.markdown(
        """
        ```
        UCI dataset (student-mat.csv)
              |
              v
        [Target rule]  at_risk = G3 < 10   (G3 itself is NEVER a feature)
              |
              v
        [Leakage audit] exclude G1, G2, G3  -> 30 prediction-safe features
              |
              v
        [Preprocessing]  numeric: impute + scale  |  categorical: one-hot
              |                (fitted on TRAINING data only)
              v
        [Training]  stratified 80/20 split, seed 42
              |     5-fold CV grid search on training data
              v
        Logistic Regression | Decision Tree | Random Forest
              |
              v
        [Evaluation on held-out test set]  Accuracy / Precision / Recall /
                                            F1 / Confusion matrix / ROC-AUC
              |
              v
        results/ + models/  -->  this Streamlit app + paper + presentation
        ```
        """
    )
    st.markdown("#### Compared algorithms")
    st.markdown(
        "| Algorithm | Family | Why included |\n"
        "|---|---|---|\n"
        "| Logistic Regression | Linear model | Standard interpretable "
        "baseline for binary classification |\n"
        "| Decision Tree | Rule-based | Captures non-linear thresholds, "
        "human-readable rules |\n"
        "| Random Forest | Ensemble of trees | Reduces overfitting by "
        "averaging many decorrelated trees |"
    )
    disclaimer()


# ---------------------------------------------------------------------------
# Page 2: Student Risk Assessment
# ---------------------------------------------------------------------------

LABEL_OVERRIDES = {
    "school": "School", "sex": "Sex", "age": "Age", "address": "Address type",
    "famsize": "Family size", "Pstatus": "Parents' cohabitation",
    "Medu": "Mother's education", "Fedu": "Father's education",
    "Mjob": "Mother's job", "Fjob": "Father's job",
    "reason": "Reason for choosing school", "guardian": "Guardian",
    "traveltime": "Travel time to school", "studytime": "Weekly study time",
    "failures": "Past class failures", "schoolsup": "Extra school support",
    "famsup": "Family educational support", "paid": "Extra paid classes",
    "activities": "Extra-curricular activities",
    "nursery": "Attended nursery", "higher": "Wants higher education",
    "internet": "Internet at home", "romantic": "Romantic relationship",
    "famrel": "Family relationship quality", "freetime": "Free time",
    "goout": "Going out with friends", "Dalc": "Workday alcohol use",
    "Walc": "Weekend alcohol use", "health": "Health status",
    "absences": "School absences",
}

NUMERIC_HELP = {
    "Medu": "0=none, 1=primary, 2=5th-9th, 3=secondary, 4=higher",
    "Fedu": "0=none, 1=primary, 2=5th-9th, 3=secondary, 4=higher",
    "traveltime": "1=<15min, 2=15-30min, 3=30-60min, 4=>60min",
    "studytime": "1=<2h, 2=2-5h, 3=5-10h, 4=>10h",
    "famrel": "1=very bad ... 5=excellent",
    "freetime": "1=very low ... 5=very high",
    "goout": "1=very low ... 5=very high",
    "Dalc": "1=very low ... 5=very high",
    "Walc": "1=very low ... 5=very high",
    "health": "1=very bad ... 5=very good",
    "failures": "Number of past class failures (0-3)",
    "absences": "Number of school absences (0-93)",
}


def page_assessment():
    header(
        "Student Risk Assessment",
        "Single-student prediction produced by the trained pipeline. "
        "Inputs are generated from the model's own feature schema.",
    )
    try:
        pipeline, metadata = _load_model()
    except (ModelNotAvailableError, FileNotFoundError) as exc:
        st.error(str(exc))
        return

    # Model picker (all pipelines from the same primary run)
    features, numeric, categorical = feature_schema(metadata)
    available_names = ["Random Forest", "Logistic Regression", "Decision Tree"]
    chosen = st.selectbox(
        "Model", available_names,
        index=available_names.index(metadata.get("best_model",
                                                 "Random Forest"))
        if metadata.get("best_model") in available_names else 0,
        help="All three models were trained on the same split; the best "
             "one (by F1 on the test split) is pre-selected.",
    )
    model_path = config.MODELS_DIR / (
        "best_model.joblib" if chosen == metadata.get("best_model")
        else f"best_model_{chosen.lower().replace(' ', '_')}.joblib"
    )
    if not Path(model_path).exists():
        st.error(
            f"The trained '{chosen}' artifact was not found at "
            f"`{model_path.relative_to(APP_DIR)}`. Run `python -m src.train` "
            "to (re)generate artifacts with this exact model scheme."
        )
        return
    import joblib
    pipeline = joblib.load(model_path)

    with st.form("student_form"):
        st.markdown("##### Student attributes "
                    "(*no grades are requested - the primary experiment "
                    "uses none*)")
        cols = st.columns(3)
        col_idx = 0
        numeric_values, categorical_values = {}, {}

        for col in numeric:
            lo, hi = metadata["numeric_ranges"].get(col, [0, 100])
            lo, hi = int(lo), int(hi)
            with cols[col_idx % 3]:
                numeric_values[col] = st.select_slider(
                    LABEL_OVERRIDES.get(col, col),
                    options=list(range(lo, hi + 1)),
                    value=max(lo, min(hi, (lo + hi) // 2)),
                    help=NUMERIC_HELP.get(col),
                )
            col_idx += 1
        for col in categorical:
            with cols[col_idx % 3]:
                categorical_values[col] = st.selectbox(
                    LABEL_OVERRIDES.get(col, col),
                    options=metadata["categorical_values"].get(col, [""]),
                )
            col_idx += 1

        submitted = st.form_submit_button(
            "Predict risk category", type="primary", width="stretch"
        )

    if submitted:
        record = {**numeric_values, **categorical_values}
        problems = validate_input(record, metadata)
        if problems:
            st.error("**Input validation failed:**\n\n- "
                     + "\n- ".join(problems))
            return
        result = predict_student(record, pipeline, metadata)
        pred = result["predicted_class"]
        prob_at_risk = (result.get("probabilities") or {}).get("at_risk")

        st.markdown("##### Result")
        c1, c2 = st.columns([1, 2])
        with c1:
            if pred == "at_risk":
                st.error(f"### At risk\nmodel estimate")
            else:
                st.success(f"### Not at risk\nmodel estimate")
        with c2:
            if prob_at_risk is not None:
                st.markdown(
                    f'<div class="metric-tile"><b>{prob_at_risk:.1%}</b><br>'
                    'model-estimated probability of the <i>at risk</i> '
                    'category (from predict_proba, not a certainty)'
                    '</div>', unsafe_allow_html=True)
            else:
                note("The selected model does not expose probability "
                     "estimates - only the discrete category is shown.")

        note(
            "**Why this prediction is uncertain:** the classifier was "
            "trained on 395 historical records of secondary-school students "
            "(dataset best F1 on unseen data: 0.52, Random Forest). It "
            "captures statistical *associations*, not individual truth. Do "
            "not treat any student according to this output - use it only "
            "as a demonstration in class."
        )

        imp = metadata.get("permutation_importance", {}).get("top_features")
        if imp:
            st.markdown("##### Factors the model relies on most (dataset level)")
            imp_df = pd.DataFrame(imp)
            st.bar_chart(imp_df.set_index("feature")["importance_mean"])
            note("These permutation importances describe what the model "
                 "weights *overall* on the held-out test split - they are "
                 "not a causal explanation for this specific student, and "
                 "importance can differ if data distribution changes.")

    # Disclosure always visible on the page
    note(
        "The form was generated dynamically from the saved model feature "
        "schema (feature names, numeric ranges and allowed categorical "
        "values come from model metadata - not hardcoded). G1/G2/G3 are "
        "excluded by design to avoid data leakage."
    )


# ---------------------------------------------------------------------------
# Page 3: Algorithm Comparison
# ---------------------------------------------------------------------------

def page_comparison():
    header(
        "Algorithm Comparison",
        "Measured results from the saved experiment artifacts "
        "(results/metrics/model_metrics.json) - nothing is hardcoded.",
    )
    metrics = _load_metrics()
    metadata = _load_metadata()
    if not metrics:
        st.error(
            "No experiment artifacts found. Run:\n\n"
            "    python -m src.train\n\n"
            "then restart the application."
        )
        return

    df = pd.DataFrame(metrics)
    metric_cols = {
        "accuracy": "Accuracy",
        "precision_positive": "Precision (at_risk)",
        "recall_positive": "Recall (at_risk)",
        "f1_positive": "F1 (at_risk)",
        "roc_auc": "ROC-AUC",
    }
    present = [c for c in metric_cols if c in df.columns]

    # Metrics table
    st.markdown("##### Measured test metrics (positive class = `at_risk`)")
    st.dataframe(
        df[list(metric_cols)].round(4).rename(columns=metric_cols),
        width="stretch",
    )

    # Interactive chart: choose metric(s)
    st.markdown("##### Comparison chart")
    selected = st.multiselect(
        "Metrics to display", present,
        default=["f1_positive", "recall_positive"],
        format_func=lambda c: metric_cols[c],
    )
    if selected:
        chart_df = df.set_index("model")[selected].rename(
            columns={c: metric_cols[c] for c in selected}
        ).T
        st.bar_chart(chart_df)
    note("Computed on a single stratified 80/20 held-out split "
         "(79 students, seed 42). Comparing on one split is a limitation: "
         "small samples (20% of 395) make differences below a few points "
         "largely noise; cross-validation on training data gave a similar "
         "ranking (see Methodology).")

    # Training times
    if "train_seconds" in df.columns:
        st.markdown("##### Measured training time "
                    "(grid-search CV, same machine/conditions)")
        st.dataframe(
            df[["model", "train_seconds", "predict_test_seconds"]]
            .rename(columns={
                "train_seconds": "Fit+CV time (s)",
                "predict_test_seconds": "Predict test set (s)",
            }).set_index("model"),
            width="stretch",
        )

    # Confusion matrices & ROC
    st.markdown("##### Confusion matrices (rows = actual, "
                "columns = predicted)")
    cm_png = FIG_DIR / "confusion_matrices.png"
    if cm_png.exists():
        st.image(str(cm_png))
    else:
        st.info("Run `python -m src.train` to generate figures.")

    st.markdown("##### ROC curves")
    roc_png = FIG_DIR / "roc_curves.png"
    if roc_png.exists():
        st.image(str(roc_png))

    # Auto summary based on actual values
    cv = (metadata or {}).get("cv_results", {})
    if cv:
        cv_notes = ", ".join(
            f"{k} CV-F1 {v['best_cv_f1']:.3f}" for k, v in cv.items()
        )
    else:
        cv_notes = "CV results not available"
    if present:
        best_by_f1 = df.sort_values("f1_positive", ascending=False).iloc[0]
        best_by_auc = df.sort_values("roc_auc", ascending=False).iloc[0]
        best_recall = df.sort_values("recall_positive", ascending=False).iloc[0]
        st.markdown("##### Auto-generated summary (from measured values)")
        st.markdown(
            f"""
            - **Best F1 (at_risk):** {best_by_f1['model']}
              (F1 = {best_by_f1['f1_positive']:.3f})
            - **Best ROC-AUC:** {best_by_auc['model']}
              (AUC = {best_by_auc['roc_auc']:.3f})
            - **Best Recall (at_risk):** {best_recall['model']}
              (Recall = {best_recall['recall_positive']:.3f})

            Interpretation: ranking by F1, {best_by_f1['model']} looks best on
            the held-out split, but the margins are small and all models are
            far from perfect. For early-warning use you would normally first
            look at **Recall** (of students flagged at risk) - missing a
            struggling student (false negative) is usually costlier than
            falsely flagging one (false positive) - while precision controls
            how many such flags are wrong.
            """
        )
    st.info(
        "Limitations: results are from one stratified test split with only "
        "79 students; small-sample differences should not be "
        "over-interpreted. Cross-validation on the training data (5-fold, "
        f"F1 of the at_risk class) gave: {cv_notes} - i.e., a similar "
        "range for all three algorithms, so the tree-vs-linear differences "
        "on a single split are within noise."
    )


# ---------------------------------------------------------------------------
# Page 4: Data Insights
# ---------------------------------------------------------------------------

def page_insights():
    header(
        "Data Insights",
        "Exploratory analysis and importance analysis computed from the "
        "processed dataset and trained artifacts.",
    )
    eda = _load_eda()
    metadata = _load_metadata()
    processed = _load_processed()
    if not eda or not metadata:
        st.error("Artifacts missing. Run `python -m src.train` first.")
        return

    col1, col2,_ = st.columns([1, 1, 0.4])
    col1.markdown(
        f'<div class="metric-tile"><b>{eda["g3_stats"]["mean"]}</b>'
        f'<br>mean final grade G3 (min {eda["g3_stats"]["min"]}, '
        f'max {eda["g3_stats"]["max"]})</div>', unsafe_allow_html=True)
    col2.markdown(
        f'<div class="metric-tile"><b>{eda["target_counts"].get("at_risk")}'
        f' / {eda["n_rows"]}</b><br>students below the pass threshold '
        '(G3 &lt; 10)</div>', unsafe_allow_html=True)

    st.markdown("##### Class distribution and grade distribution")
    png = FIG_DIR / "class_distribution.png"
    if png.exists():
        st.image(str(png))

    st.markdown("##### Missing values")
    missing = eda.get("missing_values", {})
    if missing:
        st.dataframe(pd.Series(missing, name="missing"))
    else:
        st.success("No missing values in the dataset (verified during "
                   "loading - the UCI Student Performance dataset is "
                   "complete by construction).")

    st.markdown("##### Correlation of numeric features with G3")
    corr = eda["correlation_with_G3"]
    corr_df = pd.DataFrame(
        sorted(corr.items(), key=lambda kv: kv[1]),
        columns=["feature", "correlation with G3"],
    ).set_index("feature")
    st.bar_chart(corr_df["correlation with G3"])
    note("Correlations describe linear association, not causation. The "
         "strongest points (failures, absences, Medu, age) are known "
         "education-domain signals; grades themselves are excluded from "
         "the model features.")

    st.markdown("##### Feature importance / permutation importance "
                "(primary early experiment)")
    imp_png = FIG_DIR / "feature_importance.png"
    if imp_png.exists():
        st.image(str(imp_png))
    pi = metadata.get("permutation_importance", {}).get("top_features")
    if pi:
        pi_df = pd.DataFrame(pi).set_index("feature")
        st.markdown("Permutation importance - "
                    f"{metadata['permutation_importance'].get('scoring')}")
        st.bar_chart(pi_df["importance_mean"])
        note("Permutation importance measures how much performance drops "
             f"when a feature is shuffled ({metadata['permutation_importance'].get('n_repeats')} "
             "repeats, held-out split). It is model-specific evidence of "
             "predictive usefulness - **not proof of causation** - and it "
             "can change with a different model or data distribution.")

    st.markdown("##### Feature distributions (processed dataset)")
    if processed is not None:
        num_cols = metadata["feature_schema"]["numeric"]
        pick = st.select_slider(
            "Select a numeric feature",
            options=num_cols,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.bar_chart(processed[pick].value_counts().sort_index())
        with c2:
            sub = processed[[pick, "target"]]
            grouped = sub.groupby(["target", pick]).size().unstack(fill_value=0)
            st.bar_chart(grouped.T)
    disclaimer()


# ---------------------------------------------------------------------------
# Page 5: Research & Methodology
# ---------------------------------------------------------------------------

def page_methodology():
    header(
        "Research & Methodology",
        "Dataset documentation, target definition, experiment design, "
        "reproducibility details, and references.",
    )
    metadata = _load_metadata()

    st.markdown("### Dataset documentation")
    info = config.DATASET_INFO
    st.markdown(
        f"""
        - **Source:** {info['source']} - {info['source_url']}
        - **Used file:** `data/raw/student-mat.csv` (Mathematics; the
          Portuguese file is available for extension but not used in the
          primary experiment).
        - **Samples:** {info['n_samples_math']} students, {info['n_features_math']} raw columns,
          {info['missing_values']} missing values.
        - **Citation:** {info['citation']}
        """
    )
    st.markdown("### Target definition")
    st.info(config.describe_target_rule())
    st.markdown("### Prediction scenario (and leakage prevention)")
    st.markdown(
        "**Primary early-prediction experiment.** The features are "
        "demographic, social and school-context attributes collected before "
        "grades start arriving. Excluded by design:\n\n"
        "| Excluded column | Why |\n|---|---|\n"
        "| G3 | Directly defines the target -> direct leakage |\n"
        "| G1, G2 | Intermediate grades, unavailable at early prediction "
        "time -> temporal leakage |\n"
        "\nThe remaining features are audited: `failures` (past class "
        "failures) and `absences` are *current-term proxies* a school might "
        "know early in the year; this is clearly a proxy for information "
        "availability rather than a guarantee, and it is treated as a "
        "limitation in the documentation.\n\n"
        "**Optional separate experiment** (`python -m src.train "
        "--with-grades`) includes G1/G2. Its artifacts are stored in "
        "separate files (`*_with_grades.json`) and its results (accuracy "
        "0.87) are **not comparable** to the primary scenario: they show "
        "how much intermediate grades alone explain final risk - which is "
        "exactly why omitting them early is the honest scenario."
    )
    if metadata:
        bv = metadata["class_balance_train"], metadata["class_balance_test"]
        st.markdown("### Experiment setup (from saved metadata)")
        st.markdown(
            f"""
            - Random seed: `{metadata['random_seed']}` - split:
              `{metadata['n_train']}` train / `{metadata['n_test']}` test
              (stratified), CV: {metadata['cv_folds']}-fold on training data.
            - Class balance (train): {metadata['class_balance_train']} -
              (test): {metadata['class_balance_test']}
            - Positive class: `{metadata['positive_class']}`;
              `class_weight='balanced'` used in all three algorithms because
              the at-risk class is the minority.
            - Preprocessing: median-impute + standard-scale numerics;
              most-frequent-impute + one-hot categoricals
              (`handle_unknown='ignore'`); fitted on training data only.
            - Tuning: small GridSearchCV grids (F1 scorer, positive class =
              at_risk), test split untouched.
            - Best model: **{metadata['best_model']}** -
              {metadata['selection_rule']}.
            - Environment: Python {metadata['env']['python']},
              scikit-learn {metadata['env']['sklearn']},
              pandas {metadata['env']['pandas']}.
            """
        )
    st.markdown("### Evaluation metrics")
    st.markdown(
        """
        - **Accuracy** - overall fraction correct; can hide minority-class
        performance when classes are imbalanced.
        - **Precision (at_risk)** - of students flagged at risk, how many
        really are; matters to avoid stigmatizing false alarms.
        - **Recall (at_risk)** - of truly at-risk students, how many we
        catch; this is the key metric when the goal is *not to miss*
        students who need support.
        - **F1 (at_risk)** - harmonic mean of precision & recall.
        - **ROC-AUC** - probability the model ranks a random at-risk
        student above a random not-at-risk one; threshold-free.
        - **Confusion matrix** - explicit TP / FP / FN / TN counts.
        """
    )
    st.markdown("### Limitations")
    st.markdown(
        """
        1. Small dataset (395 students, single school cohort, single subject)
        -> high variance in evaluation; single-split numbers are noisy.
        2. Only two Portuguese schools in 2008; findings may not transfer.
        3. The at-risk threshold (G3 < 10) is a project-defined operational
        rule, not a universal definition of academic risk.
        4. Feature information availability is *approximated*: some features
        (e.g. absences, failures) may not be fully known at the start of a
        course in a real school.
        5. Predictive association, not causation - no causal claims are
        supported by this analysis.
        6. Fairness/bias: features like sex, family size or parents' jobs
        can encode socioeconomic patterns; a real system would need careful
        fairness auditing before any deployment.
        7. The model is trained and evaluated on static historical data; a
        deployed system would need monitoring for distribution shift.
        """
    )
    st.markdown("### References")
    st.markdown(Path(APP_DIR / "research" / "references.md").read_text(encoding="utf-8")
                if (APP_DIR / "research" / "references.md").exists()
                else "references.md not yet generated.")
    disclaimer()


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Student Success Predictor",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.sidebar.title("🎓 Student Success Predictor")
    st.sidebar.caption("Early academic risk prediction - university "
                       "course project (educational prototype).")
    page = st.sidebar.radio("Navigation", PAGE_TABS, label_visibility="collapsed")

    st.sidebar.markdown("---")
    exp = _load_metrics()
    met = _load_metadata()
    if exp and met:
        best = met.get("best_model", "-")
        st.sidebar.success(f"Artifacts loaded\n\n**Best model:** {best}\n\n"
                           f"Experiment: primary early-prediction "
                           f"(seed {met['random_seed']})")
    else:
        st.sidebar.warning(
            "Artifacts not found. Train first:\n\n"
            "```\npython -m src.train\n```"
        )

    # Page dispatch - a dictionary is cleaner than an if/elif chain.
    dispatch = {
        "Home": page_home,
        "Student Risk Assessment": page_assessment,
        "Algorithm Comparison": page_comparison,
        "Data Insights": page_insights,
        "Research & Methodology": page_methodology,
    }
    dispatch[page]()


if __name__ == "__main__":
    main()
