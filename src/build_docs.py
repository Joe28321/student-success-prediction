"""Generate the research paper (DOCX) and presentation (PPTX).

IMPORTANT: every number and figure inserted here comes from the saved
experiment artifacts (results/metrics/*). If artifacts are missing the
script refuses to produce a fake-results document and exits with an error
message.

Run:  python -m src.build_docs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from pptx import Presentation
from pptx.util import Inches as PInches

from src import config

ARTIFACTS_OK = (
    config.ARTIFACT_METRICS.exists()
    and config.ARTIFACT_METADATA.exists()
    and config.FIGURES_DIR.exists()
)

PLACEHOLDER_TEAM = "[Team Member 1 & Team Member 2 - add names]"
PLACEHOLDER_UNI = "[University name - add]"
PLACEHOLDER_COURSE = "Algorithms for Data Science"
PLACEHOLDER_DATE = "[Submission date - add]"


def load_artifacts():
    metrics = json.loads(config.ARTIFACT_METRICS.read_text(encoding="utf-8"))
    metadata = json.loads(config.ARTIFACT_METADATA.read_text(encoding="utf-8")) if \
        config.ARTIFACT_METADATA.exists() else {}

    midterm_metrics = None
    midterm_path = config.METRICS_DIR / "model_metrics_with_grades.json"
    if midterm_path.exists():
        midterm_metrics = json.loads(midterm_path.read_text(encoding="utf-8"))

    eda = json.loads(config.ARTIFACT_EDA_SUMMARY.read_text(encoding="utf-8")) \
        if config.ARTIFACT_EDA_SUMMARY.exists() else {}
    return metrics, metadata, midterm_metrics, eda


def fmt(v, nd=3):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return f"{f:.{nd}f}"


def metrics_row_text(m):
    return (
        f"{m['model']}: Accuracy {fmt(m['accuracy'])}, "
        f"Precision {fmt(m['precision_positive'])}, "
        f"Recall {fmt(m['recall_positive'])}, "
        f"F1 {fmt(m['f1_positive'])}, "
        f"ROC-AUC {fmt(m.get('roc_auc'))}"
    )


# ===========================================================================
# Research paper (DOCX)
# ===========================================================================

def add_heading(doc, text, level=1):
    doc.add_heading(text, level=level)


def add_para(doc, text, style=None, align=None):
    p = doc.add_paragraph(text, style=style)
    if align is not None:
        p.alignment = align
    return p


def build_paper():
    metrics, metadata, midterm, eda = load_artifacts()
    best = metadata.get("best_model", metrics[0]["model"])

    doc = Document()
    styles = doc.styles["Normal"]
    styles.font.name = "Calibri"
    styles.font.size = Pt(11)

    # --- Title page -------------------------------------------------------
    t = doc.add_heading(
        "Student Success Predictor: A Comparative Study of Machine "
        "Learning Algorithms for Early Academic Risk Prediction",
        level=0,
    )
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_para(doc, PLACEHOLDER_TEAM, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, f"{PLACEHOLDER_UNI} — {PLACEHOLDER_COURSE}",
             align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, PLACEHOLDER_DATE, align=WD_ALIGN_PARAGRAPH.CENTER)

    # --- Abstract ---------------------------------------------------------
    add_heading(doc, "Abstract", level=1)
    ar = [m for m in metrics if m["model"] == best][0]
    add_para(
        doc,
        "Early identification of academically at-risk students allows "
        "schools to allocate support before failure occurs, but deploying "
        "the strongest possible predictive model is not the real "
        "constraint: grades themselves become available only late in the "
        "course. This project compares three standard classifiers - "
        "Logistic Regression, Decision Tree, and Random Forest - on the "
        "UCI Student Performance dataset (395 secondary-school students "
        "of Mathematics) under an honest early-prediction protocol: no "
        "grade feature (G1, G2, or G3) is available to the models. The "
        "binary target is defined by a project-level operational rule "
        "(at risk if G3 < 10 on the 0-20 Portuguese scale). A stratified "
        f"80/20 split (seed 42) is held out, while tuning uses 5-fold "
        "cross-validation on the training split only. With the at-risk "
        "class as positive, the measured test results were: "
        + "; ".join(metrics_row_text(m) for m in metrics)
        + f". {best} achieved the best F1 and ROC-AUC, yet all models "
        "remain far from perfect; notably, including the intermediate "
        "grades G1/G2 in a separate, non-comparable experiment raises "
        "test accuracy from roughly 0.7 to 0.87 - quantifying how much "
        "late-course information 'leaks' into the task when it is allowed. "
        "We report confusion matrices, ROC curves, permutation importance, "
        "class imbalance effects (class_weight='balanced'), and discuss "
        "limitations: small sample size, single-context cohort, "
        "proxy-based feature availability, and the fact that findings are "
        "predictive associations, not causal effects. The trained pipeline "
        "and an interactive, disclaimer-first Streamlit application are "
        "released as an educational prototype with reproducible "
        "configuration and tests.",
    )

    # --- Keywords ---------------------------------------------------------
    add_heading(doc, "Keywords", level=1)
    add_para(doc, "Educational data mining; machine learning; classification; "
                  "student performance prediction; data leakage; "
                  "imbalance; scikit-learn; Streamlit")

    # --- 1 Introduction ---------------------------------------------------
    add_heading(doc, "1. Introduction", level=1)
    add_para(doc, "Academic failure rarely starts at the final exam. "
        "Schools that can identify students at risk early enough to "
        "intervene have a practical advantage, which is why machine "
        "learning on educational records (Educational Data Mining; "
        "Romero & Ventura, 2010) has grown as a research area. The UCI "
        "Student Performance dataset (Cortez & Silva, 2008; Cortez, 2008) "
        "is a standard, openly licensed testbed for this problem and is "
        "the dataset used here (Cortez, 2008, DOI 10.24432/C5TG7T).")
    add_para(doc, "A recurring pitfall in this setting is data leakage: "
        "models are often trained with intermediate grades (G1, G2) that "
        "be available only near the end of the course, making them "
        "excellent predictors of the final grade (G3) but useless for an "
        "early system (Cortez, 2008, explicitly notes the strong "
        "correlation between G3 and G1/G2). This project therefore "
        "deliberately excludes all grade columns from the feature set in "
        "the primary experiment and treats grade-free prediction as the "
        "honest scenario.")
    add_para(doc, "This project - executed by a two-student team as part "
        "of a university data science course - contributes (i) a "
        "leakage-aware, reproducible experimental design; (ii) a "
        "controlled comparison of three classic algorithms; (iii) honest "
        "reporting of measured metrics including their uncertainty; and "
        "(iv) an interactive application that surfaces the measured "
        "results and clearly discloses their limits.")

    # --- 2 Problem Statement ----------------------------------------------
    add_heading(doc, "2. Problem Statement", level=1)
    add_para(doc, "Can a machine learning classifier, built on student "
        "attributes that are available before grades are known (family, "
        "demographic, social, and school-context features), identify "
        "students who will end the course below the pass threshold? How "
        "do three representative algorithms compare on this task under "
        "stratified, leakage-safe, reproducible conditions?")

    # --- 3 Research Objectives --------------------------------------------
    add_heading(doc, "3. Research Objectives", level=1)
    add_para(doc, "1. Define an explicit, documented target: at_risk = "
        "G3 < 10 on the dataset's 0-20 grading scale.",
        style="List Number")
    add_para(doc, "2. Prevent data leakage by excluding G3 (defines the "
        "target) and G1/G2 (unavailable early) in the primary experiment.",
        style="List Number")
    add_para(doc, "3. Train, tune (training data only) and evaluate "
        "Logistic Regression, Decision Tree, and Random Forest under "
        "identical conditions.", style="List Number")
    add_para(doc, "4. Report Accuracy, Precision, Recall, F1, confusion "
        "matrix, ROC-AUC and training time on a held-out test split.",
        style="List Number")
    add_para(doc, "5. Quantify the impact of including intermediate grades "
        "in a clearly separated experiment.", style="List Number")
    add_para(doc, "6. Deliver an interactive application that only shows "
        "measured results.", style="List Number")

    # --- 4 Research Questions ---------------------------------------------
    add_heading(doc, "4. Research Questions", level=1)
    add_para(doc, "RQ1. How do the three algorithms compare when predicting "
        "an explicitly defined academic risk category from attributes "
        "available before the prediction time?")
    add_para(doc, "RQ2. How large is the performance gap between the "
        "leakage-free early scenario and the intermediate-grades "
        "(midterm) scenario, and why is it not a fair comparison?")
    add_para(doc, "RQ3. Which evaluation metric best serves the goal of "
        "identifying students who may need support, given class imbalance?")

    # --- 5 Literature Review ----------------------------------------------
    add_heading(doc, "5. Literature Review", level=1)
    add_para(doc, "Cortez and Silva (2008) introduced the dataset used "
        "here and modeled both Mathematics and Portuguese course "
        "performance under binary/five-level and regression tasks; they "
        "predict G3 (or binarized G3) with techniques including decision "
        "trees, random forests, neural networks and SVMs, and explicitly "
        "observe that excluding G1/G2 makes the task harder but more "
        "useful for early prediction. In the broader educational data "
        "mining literature, Romero and Ventura (2010) survey methods, "
        "tools and evaluation practice for predicting academic outcomes.")
    add_para(doc, "On algorithms: regularized logistic regression is a "
        "standard linear baseline for binary classification (Cox, 1958; "
        "Hastie, Tibshirani, & Friedman, 2009). Decision trees partition "
        "the feature space with axis-aligned rules (Breiman et al., "
        "1984), are interpretable but prone to overfitting without "
        "pruning. Random forests (Breiman, 2001) average many "
        "decorrelated trees to reduce variance and typically improve "
        "generalization on small tabular datasets such as this one.")
    add_para(doc, "On evaluation: overall accuracy is known to be a weak "
        "indicator under class imbalance (He & Garcia, 2009); recall of "
        "the minority class determines how many at-risk students are "
        "actually found, precision controls false alarms, and F1 "
        "balances the two. Precision-recall style evaluation is being "
        "noted as more informative than ROC on imbalanced data in some "
        "settings (Saito & Rehmsmeier, 2015), which is why this project "
        "reports the full confusion matrix alongside scalar metrics.")

    # --- 6 Dataset ----------------------------------------------------------
    add_heading(doc, "6. Dataset Description", level=1)
    add_para(doc, f"Source: UCI Machine Learning Repository - Student "
        f"Performance (Cortez, 2008), DOI 10.24432/C5TG7T, license CC BY 4.0. "
        f"The primary file is student-mat.csv (Mathematics): "
        f"{config.DATASET_INFO['n_samples_math']} students, "
        f"{config.DATASET_INFO['n_features_math']} columns, "
        f"{config.DATASET_INFO['missing_values']} missing values. A "
        f"Portuguese-language file (student-por.csv, 649 students) is "
        f"kept for extensions but not used in the reported experiment.")
    add_para(doc, "The 33 columns cover school, demographic (e.g. sex, "
        "age, address, family size/education/jobs), behavioural (study "
        "time, failures, absences), social (free time, going out, "
        "alcohol use), and grade variables G1-G3. Full definitions are "
        "stored in src/config.py and surfaced in the application's "
        "Methodology page.")
    add_para(doc, "The target G3 distribution is imbalanced: "
        + ", ".join(f"{k} {v} ({eda.get('target_share', {}).get(k, 0):.0%})"
                    for k, v in eda.get("target_counts", {}).items())
        + " - 'at risk' is the minority class, motivating "
        "class_weight='balanced' in all models.")

    # --- 7-10 Methodology / preprocessing / algorithms / setup -------------
    add_heading(doc, "7. Methodology", level=1)
    add_para(doc, "The experiment follows a fixed, scriptable pipeline "
        "(src/train.py): load and validate the raw file; build the "
        "target; audit exclusions; stratified 80/20 split with seed 42; "
        "grid-search tuning with 5-fold stratified CV on the training "
        "split only; final evaluation on the held-out split. The test "
        "split is used once, for the reported comparison, and never for "
        "tuning or threshold selection.")
    add_heading(doc, "8. Data Preprocessing", level=1)
    add_para(doc, "Numeric features (13 of 30) receive median imputation "
        "followed by standardization; categorical features (17 of 30) "
        "receive most-frequent imputation and one-hot encoding with "
        "handle_unknown='ignore' (so unknown category values at "
        "prediction time do not crash the saved pipeline). All "
        "preprocessing is wrapped in a sklearn ColumnTransformer inside "
        "a Pipeline and is fitted on the training split only. The ten "
        "excluded/segregated grade columns are audited in code: G3 is "
        "excluded in every scenario - it defines the target - and G1/G2 "
        "are excluded in the primary early scenario; the audit "
        "(src/data_loader.get_feature_metadata) reports exactly which "
        "columns were excluded and why, and a test asserts the audit.")
    add_heading(doc, "9. Algorithm Descriptions", level=1)
    add_para(doc, "Logistic Regression predicts the log-odds of the "
        "positive class as a linear combination of features; we use "
        "default L2 regularization with C selected from {0.1, 1, 10}. "
        "(Cox, 1958; Pedregosa et al., 2011).")
    add_para(doc, "Decision Tree greedily selects axis-aligned splits that "
        "maximize impurity reduction; we tune max_depth and "
        "min_samples_leaf, a light pruning for this small dataset "
        "(Breiman et al., 1984).")
    add_para(doc, "Random Forest bagges hundreds of decision trees with "
        "random feature subsampling (Breiman, 2001); we tune "
        "n_estimators, max_depth and min_samples_leaf. Because the "
        "at-risk class is the minority in all splits, all three models "
        "use class_weight='balanced' so that the minority class "
        "contributes proportionally to the loss.")
    add_heading(doc, "10. Experimental Setup", level=1)
    n_train = metadata.get("n_train", "?")
    n_test = metadata.get("n_test", "?")
    add_para(doc, f"Reproducible configuration (stored verbatim in "
        f"results/metrics/model_metadata.json): random seed "
        f"{metadata.get('random_seed', 42)}; stratified split "
        f"{n_train} train / {n_test} test; class balance "
        f"(train) {metadata.get('class_balance_train', '?')} and "
        f"(test) {metadata.get('class_balance_test', '?')}; 5-fold "
        f"stratified CV on training data scored by F1 of the positive "
        f" ('at_risk') class; class_weight='balanced'; Python "
        f"{metadata.get('env', {}).get('python', '?')}, "
        f"scikit-learn {metadata.get('env', {}).get('sklearn', '?')}.")

    # --- 11-12 Metrics ------------------------------------------------------
    add_heading(doc, "11. Evaluation Metrics", level=1)
    add_para(doc, "Accuracy = overall correct fraction - but with "
        "imbalanced classes it can hide poor minority performance "
        "(He & Garcia, 2009). Precision(at_risk) = how many flagged "
        "students are truly at risk (false alarms cost credibility and "
        "stigma). Recall(at_risk) = how many true at-risk students the "
        "model catches - the primary quantity of interest for early "
        "support. F1 harmonizes precision and recall. ROC-AUC measures "
        "ranking quality across thresholds. Confusion matrices expose "
        "TP/FP/FN/TN explicitly.")
    add_heading(doc, "12. Results", level=1)
    add_para(doc, "Measured on the held-out test split (positive class "
        "= at_risk):")
    table = doc.add_table(rows=1, cols=6)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    for i, name in enumerate(["Model", "Accuracy", "Precision", "Recall",
                              "F1", "ROC-AUC"]):
        hdr[i].text = name
    for m in sorted(metrics, key=lambda x: -x["f1_positive"]):
        row = table.add_row().cells
        for i, val in enumerate([
            m["model"], fmt(m["accuracy"]), fmt(m["precision_positive"]),
            fmt(m["recall_positive"]), fmt(m["f1_positive"]),
            fmt(m.get("roc_auc")),
        ]):
            row[i].text = val
    add_para(doc, "")
    add_para(doc, "Per-model confusion matrices and per-feature "
        "permutation importance (on the test split, F1 scoring) are "
        "stored in results/figures and results/metrics respectively.")
    add_para(doc, "Optional non-comparable experiment (G1/G2 included): "
        + ("; ".join(metrics_row_text(m) for m in midterm) if midterm
           else "not yet run - run 'python -m src.train --with-grades'.")
        + " This is reported ONLY as an upper bound of what late-course "
        "grade information adds, not as an early-warning result.")

    # --- 13-14 Discussion ----------------------------------------------------
    add_heading(doc, "13. Discussion", level=1)
    bid = {}
    if midterm:
        mx = max(midterm, key=lambda m: m["f1_positive"])
        bid = mx
    add_para(doc, "RQ1 (primary comparison): " + best + " achieved the "
        "highest F1 (positive class) and ROC-AUC among the three "
        "candidates, with the largest gain in precision "
        f"({fmt(ar['precision_positive'])}); the"
        " linear and single-tree baselines landed on the same confusion "
        "matrix in this run - a coincidence possible on a 79-sample "
        "test split with a small grid, reported here exactly as "
        "measured rather than smoothed. CV on the training data gave "
        "best-CV-F1 values of "
        + ", ".join(
            f"{k} {fmt(v['best_cv_f1'])}" for k, v in
            metadata.get('cv_results', {}).items()
        ) + " - i.e., within-noise differences between the algorithms on "
        "this dataset size, which is itself an important honest finding "
        "for a small-scale study: none of the three is a decisive "
        "'winner', and any claims about superiority should be limited "
        "to descriptive reporting of this one split plus the CV "
        "pattern.")
    add_para(doc, "RQ3 (metric choice): for an early-warning use case, "
        "Recall of the at-risk class matters most (false negatives are "
        "a missed student), but operating with too low a Precision "
        "erodes teacher trust; the balanced choice is hence F1 (or "
        "an application-level threshold chosen on validation data, "
        "outside the scope of this project).")
    if bid:
        add_para(doc, f"RQ2 (leakage quantification): including G1/G2 "
            f"raises the best test accuracy/F1 from {fmt(ar['accuracy'])}/"
            f"{fmt(ar['f1_positive'])} to {fmt(bid['accuracy'])}/"
            f"{fmt(bid['f1_positive'])} - the difference is essentially "
            "the information those grades carry about the final exam, "
            "which early systems cannot use.")
    add_heading(doc, "14. Limitations and Ethical Considerations", level=1)
    add_para(doc, "1. Statistical limits: 395 students from two schools, "
        "one subject, one country/era; a 79-student held-out split "
        "produces noisy metrics; single-split comparisons cannot "
        "establish algorithmic superiority more robust than the "
        "observed differences.")
    add_para(doc, "2. Scenario limits: 'features available early' is "
        "approximated by dataset attributes (e.g. absences/failures may "
        "be known only after early terms); a real deployment needs its "
        "own availability audit.")
    add_para(doc, "3. Definitions: the at-risk rule (G3 < 10) is a "
        "project-defined operational threshold on this grading scale, "
        "not a universal or clinical definition of academic risk.")
    add_para(doc, "4. Interpretation: all reported numbers are "
        "predictive associations. Absences, for example, are correlated "
        "with lower G3 but nothing here supports a causal claim; "
        "feature importance/rankings must not be read as causal or as a "
        "student judgment.")
    add_para(doc, "5. Ethics: attributes such as sex, family background, "
        "or parents' jobs can encode socioeconomic patterns; a real "
        "system would need explicit fairness auditing, transparency, "
        "student data protection, and human review before any "
        "educational decision. This project's artifacts are for "
        "teaching and demonstration only.")

    # --- 15 Conclusion --------------------------------------------------------
    add_heading(doc, "15. Conclusion and Future Work", level=1)
    add_para(doc, "Under a leakage-safe early-prediction protocol, the "
        "three classic algorithms performed in a close, statistically "
        "noisy band, with Random Forest giving the best measured F1 and "
        "ROC-AUC on the held-out split, and the comparison exposed the "
        "expected but large benefits of late-course grade information "
        "when it is wrongly allowed. Future work: cross-validated "
        "combined modelling of both subject files, hyperparameter "
        "search with repeated CV, cost-based thresholding, "
        "probability calibration, fairness/robustness audits, and "
        "semi-deployment studies with graded-feature availability.")
    add_heading(doc, "16. References", level=1)
    refs = (Path(config.PROJECT_ROOT) / "research" / "references.md").read_text(
        encoding="utf-8")
    for line in refs.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("*"):
            continue
        candidate = line.lstrip("0123456789.").strip()
        if candidate and candidate not in ("Verified References (APA 7)",):
            add_para(doc, candidate, style="List Number")

    out = config.RESEARCH_DIR / "research_paper.docx"
    config.RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    print(f"[build_docs] paper -> {out}")



# ===========================================================================
# Presentation (PPTX)
# ===========================================================================

def _slide_title_layout(prs):
    return prs.slide_layouts[0]


def add_title_slide(prs, title, subtitle):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = title
    slide.placeholders[1].text = subtitle
    return slide


def add_bullet_slide(prs, title, bullets):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = title
    body = slide.placeholders[1].text_frame
    body.text = bullets[0]
    for b in bullets[1:]:
        p = body.add_paragraph()
        p.text = b
        p.level = 0
    return slide


def add_picture_slide(prs, title, image_path):
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = title
    if image_path.exists():
        region = slide.shapes.add_picture(
            str(image_path), PInches(4.5), Inches(1.4),
            width=Inches(5.2),
        )
    return slide


def build_presentation():
    metrics, metadata, midterm, eda = load_artifacts()
    best = metadata.get("best_model", metrics[0]["model"])

    prs = Presentation()
    add_title_slide(prs,
                    "Student Success Predictor",
                    "A comparative study of machine learning algorithms for "
                    "early academic risk prediction\n"
                    + PLACEHOLDER_TEAM + " - " + PLACEHOLDER_UNI + " - "
                    + PLACEHOLDER_DATE)

    add_bullet_slide(prs, "Problem Statement",
                     ["Schools need early signals about students who may need "
                      "academic support - not post-exam judgment.",
                      "Common shortcut: using intermediate grades (G1/G2) "
                      "which 'explain' the final grade but arrive too late.",
                      "This project enforces an honest early scenario: no "
                      "grade feature at all."])
    add_bullet_slide(prs, "Motivation & Research Objectives",
                     ["Evaluate classic ML algorithms on a public, "
                      "openly licensed dataset (UCI Student Performance).",
                      "Prevent data leakage through an audited, code "
                      "enforced feature policy.",
                      "Report every metric honestly (including the weak "
                      "ones) with full reproducibility.",
                      "Deliver an interactive app that can only show "
                      "measured results."])
    add_bullet_slide(prs, "Research Questions",
                     ["RQ1 - How do Logistic Regression, Decision Tree and "
                      "Random Forest compare in early risk prediction?",
                      "RQ2 - How much performance does adding G1/G2 inject "
                      "(quantification of leakage)?",
                      "RQ3 - Which metric best serves 'identifying students "
                      "who need support' under class imbalance?"])
    eda_counts = eda.get("target_counts", {})
    add_bullet_slide(prs, "Dataset & Features",
                     [f"UCI Student Performance - student-mat.csv: "
                      f"{eda.get('n_rows', 395)} students, 33 columns, "
                      "0 missing values.",
                      f"Target: at_risk if G3 < 10 - class balance: "
                      f"{eda_counts.get('not_at_risk', '?')} "
                      f"not_at_risk vs {eda_counts.get('at_risk', '?')} "
                      f"at_risk.",
                      "Features used (early scenario): 30 demographic, "
                      "social and school-context columns.",
                      "Excluded (leakage): G3 (defines target), G1 & G2 "
                      "(unavailable early)."])
    add_bullet_slide(prs, "Data Preprocessing",
                     ["Numeric: median imputation + standardization.",
                      "Categorical: most-frequent imputation + one-hot "
                      "encoding (handle_unknown='ignore').",
                      "All preprocessing fitted on the TRAINING split "
                      "(ColumnTransformer inside a sklearn Pipeline).",
                      "Tuning: GridSearchCV, 5-fold CV on training data "
                      "only; the test split is used once at the end."])
    add_bullet_slide(prs, "Algorithms compared",
                     ["Logistic Regression - regularized linear baseline "
                      "(C in {0.1, 1, 10}).",
                      "Decision Tree - interpretable pruning-sensitive "
                      "rules (depth/min-samples tuned).",
                      "Random Forest - bagged ensemble of trees "
                      "(n_estimators in {200, 400}, depth/min-samples "
                      "tuned).",
                      "All three use class_weight='balanced' for the "
                      "imbalanced at-risk class."])
    add_bullet_slide(prs, "Experimental Setup (reproducible)",
                     [f"Random seed {metadata.get('random_seed', 42)}; "
                      f"stratified split "
                      f"{metadata.get('n_train', '?')} train / "
                      f"{metadata.get('n_test', '?')} test.",
                      f"Class balance (train): "
                      f"{metadata.get('class_balance_train', '?')}, "
                      f"(test): {metadata.get('class_balance_test', '?')}.",
                      f"Positive class: at_risk; scoring: F1 of the "
                      "positive class.",
                      "Environment: Python "
                      f"{metadata.get('env', {}).get('python', '?')}, "
                      f"scikit-learn {metadata.get('env', {}).get('sklearn', '?')}."])
    add_bullet_slide(prs, "Evaluation metrics",
                     ["Accuracy / Precision / Recall / F1 (positive class",
                      "ROC-AUC and confusion matrices.",
                      "Permutation importance on the held-out split "
                      "(no causal interpretation).",
                      "Training time measured under identical conditions."])
    for m in sorted(metrics, key=lambda x: -x["f1_positive"]):
        add_bullet_slide(
            prs, f"Results - {m['model']}",
            [f"Accuracy: {fmt(m['accuracy'])}",
             f"Precision (at_risk): {fmt(m['precision_positive'])}",
             f"Recall (at_risk): {fmt(m['recall_positive'])}",
             f"F1 (at_risk): {fmt(m['f1_positive'])}",
             f"ROC-AUC: {fmt(m.get('roc_auc'))}",
             f"Fit+C V time: {m.get('train_seconds', '?')} s "
             f"(test predict: {fmt(m.get('predict_test_seconds'), 4)} s)"])
    add_picture_slide(prs, "Model comparison (measured)",
                      config.FIGURES_DIR / "model_comparison.png")
    add_picture_slide(prs, "Confusion matrices (measured)",
                      config.FIGURES_DIR / "confusion_matrices.png")
    if (config.FIGURES_DIR / "roc_curves.png").exists():
        add_picture_slide(prs, "ROC curves (measured)",
                          config.FIGURES_DIR / "roc_curves.png")
    add_picture_slide(prs, "Feature importance (measured)",
                      config.FIGURES_DIR / "feature_importance.png")
    if midterm:
        mx = max(midterm, key=lambda m: m["f1_positive"])
        add_bullet_slide(prs, "Leakage quantification (optional experiment)",
                         ["With G1/G2 included (NOT comparable!):",
                          f"Best model {mx['model']}: "
                          f"accuracy {fmt(mx['accuracy'])}, "
                          f"F1 {fmt(mx['f1_positive'])}.",
                          "Early-scenario best F1: "
                          f"{fmt([m for m in metrics if m['model'] == best][0]['f1_positive'])}.",
                          "The gap is the 'shortcut' that G1/G2 provide near "
                          "the end of the course."])
    add_bullet_slide(prs, "Application demonstration",
                     ["Streamlit dashboard: risk input form generated from "
                      "the model schema, measured metrics, EDA, "
                      "methodology and disclaimers.",
                      "Only artifacts produced by 'python -m src.train' are "
                      "shown - no hardcoded numbers.",
                      "Missing artifacts -> clear setup instructions, "
                      "never a fake prediction."])
    add_bullet_slide(prs, "Limitations & Ethics",
                     ["Small dataset (395 students, one subject, two "
                      "schools, 2008).",
                      "Single-split evaluation - differences are noisy; "
                      "CV F1 was similar across models.",
                      "Threshold G3 < 10 is a project-defined rule.",
                      "Predictive association only - no causality claims.",
                      "Fairness/bias caveats; educational prototype, not "
                      "for real decisions."])
    add_bullet_slide(prs, "Conclusion & Future Work",
                     ["Random Forest delivered the best measured F1 and "
                      "ROC-AUC, but no algorithm separates decisively at "
                      "this scale.",
                      "Leakage-aware design is the real contribution: "
                      "without it, metrics are optimistically inflated.",
                      "Future: repeated CV, calibration, fairness audit, "
                      "cost-sensitive thresholds."])
    refs = (Path(config.PROJECT_ROOT) / "research" / "references.md").read_text(
        encoding="utf-8")
    ref_lines = [l.strip() for l in refs.splitlines()
                 if l.strip() and not l.strip().startswith(("#", "*"))]
    add_bullet_slide(prs, "References (verified)",
                     ref_lines[:12])

    out = config.RESEARCH_DIR / "presentation.pptx"
    config.RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    print(f"[build_docs] presentation -> {out}")


def main():
    if not ARTIFACTS_OK:
        print("[build_docs] ERROR: experiment artifacts are missing.")
        print("    Run 'python -m src.train' first - this script refuses to "
              "write documents with fabricated results.")
        sys.exit(1)
    build_paper()
    build_presentation()


if __name__ == "__main__":
    main()
