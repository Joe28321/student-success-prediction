"""Preprocessing pipeline for the Student Success Predictor.

Builds a leakage-safe scikit-learn Pipeline + ColumnTransformer:
- one-hot encoding for categorical columns,
- standard scaling for numeric columns,
- fitted on TRAINING data only (handled by Pipeline semantics).

The transformer is deliberately separate from any estimator so that the
same preprocessing definition is shared by all three algorithms, giving
comparable evaluation conditions.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression as _LR  # noqa: F401
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import config


def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    """ColumnTransformer with imputation + encoding/scaling per dtype."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return transformer


def make_estimator(name: str) -> object:
    """Create an unfitted classifier from a human-readable model name."""
    est = name.lower()
    if est in ("logistic regression", "logistic_regression", "logreg"):
        cls = LogisticRegression
    elif est in ("decision tree", "decision_tree", "tree"):
        cls = DecisionTreeClassifier
    elif est in ("random forest", "random_forest", "rf"):
        cls = RandomForestClassifier
    else:
        raise ValueError(f"Unknown model name: {name}")
    return cls(random_state=config.RANDOM_SEED)


MODEL_CLASSES = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
}


def build_pipeline(name: str, numeric_cols, categorical_cols) -> Pipeline:
    """Full pipeline: preprocess -> classifier.

    Accuracy of class-imbalance handling: the 'at risk' class is the
    minority (~33%), so class_weight='balanced' is justified and used
    consistently for all three algorithms (documented in config).
    """
    classifier_cls = MODEL_CLASSES[name]
    cfg = config.MODEL_CONFIGS[name]
    classifier = classifier_cls(
        random_state=config.RANDOM_SEED,
        class_weight=cfg["class_weight"],
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
            ("classifier", classifier),
        ]
    )
    return pipeline
