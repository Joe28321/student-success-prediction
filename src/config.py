"""Shared project configuration for the Student Success Predictor.

This module is the single source of truth for:
- dataset paths,
- the prediction scenario (which features are used, and which are excluded),
- the target definition (at-risk rule),
- the reproducible experiment settings (seeds, split, models).

Every other module imports from here so that the app, the experiments,
the tests and the documentation all agree on the same definitions.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to the project root, so the project can move around)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
RESEARCH_DIR = PROJECT_ROOT / "research"

# Primary dataset (Mathematics). We keep the Portuguese file available for a
# robustness check but document Mathematics as the primary subject dataset.
PRIMARY_SUBJECT = "mathematics"
DATASET_FILENAMES = {
    "mathematics": "student-mat.csv",
    "portuguese": "student-por.csv",
}

ARTIFACT_MODEL = MODELS_DIR / "best_model.joblib"
ARTIFACT_METADATA = METRICS_DIR / "model_metadata.json"
ARTIFACT_METRICS = METRICS_DIR / "model_metrics.json"
ARTIFACT_METRICS_CSV = METRICS_DIR / "model_metrics.csv"
ARTIFACT_PROCESSED_DATA = DATA_PROCESSED_DIR / "student_mat_processed.csv"
ARTIFACT_EDA_SUMMARY = METRICS_DIR / "eda_summary.json"

# ---------------------------------------------------------------------------
# Target definition
# ---------------------------------------------------------------------------
# Research question: compare Logistic Regression, Decision Tree and Random
# Forest when predicting an explicitly defined academic-risk category from
# attributes available BEFORE the prediction time.
#
# Target rule (project-defined operational rule, based on the dataset's
# 0-20 grading scale where pass is >= 10):
#     at_risk (positive class) : G3 < 10
#     not_at_risk              : G3 >= 10
TARGET_LABELS = ["not_at_risk", "at_risk"]
POSITIVE_CLASS = "at_risk"
G3_PASS_THRESHOLD = 10

# ---------------------------------------------------------------------------
# Prediction scenario (leakage policy)
# ---------------------------------------------------------------------------
# Two scenarios are implemented:
#
# 1. "early"  (PRIMARY): no grade information at all. In a real school
#    environment this corresponds to the start of the course / first term,
#    when G1 (1st period grade), G2 (2nd period grade) and G3 (final grade)
#    are not yet available. This is the honest early-warning scenario.
#
# 2. "midterm" (OPTIONAL, not directly comparable): includes G1 and G2 as
#    predictors. This is a separate experiment, NOT comparable to the early
#    scenario, because it uses information only available late in the course.
SCENARIOS = {
    # PRIMARY early-prediction scenario: NO grade information at all is used.
    # G3 is always excluded (it defines the target); G1/G2 are additionally
    # excluded here because they are only available later in the course.
    "early": {"additional_exclusions": ["G1", "G2"]},
    # OPTIONAL scenario: G1/G2 available mid-course (NOT comparable to the
    # primary early scenario).
    "midterm": {"additional_exclusions": []},
}
PRIMARY_SCENARIO = "early"

# Columns that are ALWAYS excluded from the feature space in every scenario:
GRADE_COLUMNS = ["G1", "G2", "G3"]

# "At risk" is the minority class in this dataset (~33%), so we use
# class_weight="balanced" in the models that support it, to avoid the model
# collapsing into the majority class ("not_at_risk").
USE_CLASS_WEIGHT_BALANCED = True

# ---------------------------------------------------------------------------
# Experiment settings (reproducibility)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.20  # 80% train / 20% test
CV_FOLDS = 5

MODEL_CONFIGS = {
    "Logistic Regression": {
        "estimator": "logistic_regression",
        "class_weight": "balanced" if USE_CLASS_WEIGHT_BALANCED else None,
        "tuning": "logistic_regression",  # key into PARAM_GRIDS below
    },
    "Decision Tree": {
        "estimator": "decision_tree",
        "class_weight": "balanced" if USE_CLASS_WEIGHT_BALANCED else None,
        "tuning": "decision_tree",
    },
    "Random Forest": {
        "estimator": "random_forest",
        "class_weight": "balanced" if USE_CLASS_WEIGHT_BALANCED else None,
        "tuning": "random_forest",
    },
}

# Modest tuning grids (GridSearchCV runs on TRAINING data only, using CV).
PARAM_GRIDS = {
    "logistic_regression": {
        "classifier__C": [0.1, 1.0, 10.0],
    },
    "decision_tree": {
        "classifier__max_depth": [3, 5, 7, None],
        "classifier__min_samples_leaf": [5, 10, 20],
    },
    "random_forest": {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [5, 8, None],
        "classifier__min_samples_leaf": [5, 10],
    },
}

# ---------------------------------------------------------------------------
# Dataset documentation metadata (verified against data/raw/student.txt)
# ---------------------------------------------------------------------------
DATASET_INFO = {
    "source": "UCI Machine Learning Repository - Student Performance",
    "source_url": "https://archive.ics.uci.edu/dataset/320/student+performance",
    "citation": (
        "Cortez, P., & Silva, A. M. G. (2008). Using data mining to predict "
        "secondary school student performance. In A. Brito & J. Teixeira (Eds.), "
        "Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) "
        "(pp. 5-12). EUROSIS-ETI."
    ),
    "n_samples_math": 395,
    "n_features_math": 33,
    "missing_values": 0,
    "target_column": "G3",
    "target_grading_scale": "0 to 20 (Portuguese secondary school scale)",
}

FEATURE_DICTIONARY = {
    # demographic
    "school": "Student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)",
    "sex": "Student's sex (binary: 'F' - female or 'M' - male)",
    "age": "Student's age (numeric: 15 to 22)",
    "address": "Home address type (binary: 'U' - urban or 'R' - rural)",
    "famsize": "Family size (binary: 'LE3' - <= 3 or 'GT3' - > 3)",
    "Pstatus": "Parent's cohabitation status (binary: 'T' - together or 'A' - apart)",
    "Medu": "Mother's education (0=none, 1=primary(4th grade), 2=5th-9th grade, 3=secondary, 4=higher)",
    "Fedu": "Father's education (same scale as Medu)",
    "Mjob": "Mother's job (teacher, health, services, at_home, other)",
    "Fjob": "Father's job (teacher, health, services, at_home, other)",
    "reason": "Reason to choose this school (home, reputation, course, other)",
    "guardian": "Student's guardian (mother, father, other)",
    # school-related
    "traveltime": "Home to school travel time (1=<15min, 2=15-30min, 3=30min-1h, 4=>1h)",
    "studytime": "Weekly study time (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)",
    "failures": "Number of past class failures (numeric: 0-3; n if 1<=n<3, else 4)",
    "schoolsup": "Extra educational support (binary: yes/no)",
    "famsup": "Family educational support (binary: yes/no)",
    "paid": "Extra paid classes within the course subject (binary: yes/no)",
    "activities": "Extra-curricular activities (binary: yes/no)",
    "nursery": "Attended nursery school (binary: yes/no)",
    "higher": "Wants to take higher education (binary: yes/no)",
    "internet": "Internet access at home (binary: yes/no)",
    "romantic": "With a romantic relationship (binary: yes/no)",
    # social
    "famrel": "Quality of family relationships (1=very bad to 5=excellent)",
    "freetime": "Free time after school (1=very low to 5=very high)",
    "goout": "Going out with friends (1=very low to 5=very high)",
    "Dalc": "Workday alcohol consumption (1=very low to 5=very high)",
    "Walc": "Weekend alcohol consumption (1=very low to 5=very high)",
    "health": "Current health status (1=very bad to 5=very good)",
    "absences": "Number of school absences (numeric: 0-93)",
    # grades (TARGET SOURCE ONLY - never used as features in the early scenario)
    "G1": "First period grade (0-20)",
    "G2": "Second period grade (0-20)",
    "G3": "Final grade (0-20) - defines the target, EXCLUDED from features",
}


def describe_target_rule() -> str:
    """Human-readable description of the at-risk rule (used in app + paper)."""
    return (
        f"A student is labelled at_risk when the final grade G3 < "
        f"{G3_PASS_THRESHOLD} (on the 0-20 scale), and not_at_risk when "
        f"G3 >= {G3_PASS_THRESHOLD}. This threshold is a project-defined "
        "operational rule based on the dataset's grading scale (10 is the "
        "pass mark in the Portuguese system), not a universal definition of "
        "academic risk."
    )
