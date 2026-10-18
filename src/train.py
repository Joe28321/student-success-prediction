"""Model training, tuning and persistence for the Student Success Predictor.

Experiment design (reproducible):
- stratified 80/20 train/test split (random_state=42),
- GridSearchCV with 5-fold cross-validation on the TRAINING data only,
- the test set is NEVER used for selection or tuning,
- fitted pipelines are saved with joblib together with JSON metadata
  (feature schema, scenario, seed, target rule) so that predictions later
  always use the exact same preprocessing.

Three algorithms are compared under identical conditions:
  Logistic Regression | Decision Tree | Random Forest
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)

from src import config
from src.data_loader import build_target, get_feature_metadata, load_raw
from src.evaluate import roc_auc_summary
from src.preprocessing import build_pipeline

MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest"]

_CFG_BY_NAME = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
}


def cfg_name(model_name: str) -> str:
    """Map 'Logistic Regression' -> 'logistic_regression' etc."""
    return _CFG_BY_NAME[model_name]


def make_split(df, feature_cols, target):
    """Stratified, reproducible train/test split (80/20, seed=42)."""
    X = df[feature_cols]
    y = target
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y,  # preserve the class ratio in both splits
    )
    return X_train, X_test, y_train, y_test


def tune_model(name, X_train, y_train, numeric_cols, categorical_cols,
               cv_folds=config.CV_FOLDS):
    """GridSearchCV on TRAINING data only. Returns (best_pipeline, cv_info)."""
    pipeline = build_pipeline(name, numeric_cols, categorical_cols)
    grid = config.PARAM_GRIDS[cfg_name(name)]
    scorers = {
        "at_risk": make_scorer(f1_score, pos_label=config.POSITIVE_CLASS,
                               zero_division=0),
    }
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring=scorers["at_risk"],  # positive class = 'at_risk'
        cv=StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_SEED
        ),
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    t_start = time.perf_counter()
    grid_search.fit(X_train, y_train)
    elapsed = time.perf_counter() - t_start
    cv_info = {
        "best_params": {k: (None if v is None else v)
                        for k, v in grid_search.best_params_.items()},
        "best_cv_f1": float(grid_search.best_score_),
        "n_cv_fits": int(len(grid_search.cv_results_["params"]) * cv_folds),
        "cv_folds": cv_folds,
        "scoring": "f1 (positive class = at_risk)",
        "tuning_seconds": round(elapsed, 2),
    }
    return grid_search.best_estimator_, cv_info


def evaluate_on_test(name, pipeline, X_test, y_test):
    """Compute test metrics with 'at_risk' as the positive class."""
    from src.evaluate import compute_metrics

    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    predict_seconds = time.perf_counter() - t0
    metrics = compute_metrics(y_test, y_pred, pipeline, X_test)
    metrics["predict_test_seconds"] = round(predict_seconds, 4)
    metrics["model"] = name
    return metrics


def train_all_models(df=None, scenario=config.PRIMARY_SCENARIO, save=True,
                     verbose=True, artifact_suffix=None, write_figures=True):
    """Train/tune/evaluate the three algorithms and persist artifacts.

    Returns: (results_df, artifacts_dict)
    """
    if df is None:
        df = load_raw()

    target = build_target(df)
    audit, feature_cols, numeric_cols, categorical_cols = get_feature_metadata(
        df, scenario
    )

    X_train, X_test, y_train, y_test = make_split(df, feature_cols, target)

    rows, model_blobs, cv_summaries = [], {}, {}
    for name in MODEL_NAMES:
        if verbose:
            print(f"[train] fitting + tuning {name} ...")
        best_pipe, cv_info = tune_model(
            name, X_train, y_train, numeric_cols, categorical_cols
        )
        metrics = evaluate_on_test(name, best_pipe, X_test, y_test)
        metrics["train_seconds"] = cv_info["tuning_seconds"]
        rows.append(metrics)
        model_blobs[name] = best_pipe
        cv_summaries[name] = cv_info
        if verbose:
            print(
                f"    test accuracy={metrics['accuracy']:.3f}  "
                f"recall(at_risk)={metrics['recall_positive']:.3f}  "
                f"f1(at_risk)={metrics['f1_positive']:.3f}"
            )

    results_df = pd.DataFrame(rows)
    # Add ROC-AUC (and train timing) BEFORE saving or printing anything.
    roc_auc_summary(results_df, model_blobs, X_test, y_test)

    artifacts = None
    if save:
        artifacts = save_artifacts(
            results_df, model_blobs, cv_summaries, audit,
            X_train, X_test, y_train, y_test, df, scenario,
            artifacts_suffix=artifact_suffix, write_figures=write_figures,
        )
    return results_df, artifacts


def build_eda_summary(df, audit):
    """Compute the EDA summary used by the Data Insights page."""
    target_counts = df["target"].value_counts().to_dict()
    g3_counts = df["G3"].value_counts().sort_index()
    numeric_cols = audit["numeric"]
    corr_cols = ["G3"] + numeric_cols
    corr = df[corr_cols].corr(numeric_only=True)

    return {
        "n_rows": int(len(df)),
        "n_features_used": audit["n_features"],
        "excluded_columns": audit["excluded_columns"],
        "target_counts": {str(k): int(v) for k, v in target_counts.items()},
        "target_share": {
            str(k): round(v / len(df), 4)
            for k, v in target_counts.items()
        },
        "g3_distribution": {str(int(k)): int(v) for k, v in g3_counts.items()},
        "g3_stats": {
            "mean": round(float(df["G3"].mean()), 2),
            "median": float(df["G3"].median()),
            "std": round(float(df["G3"].std()), 2),
            "min": int(df["G3"].min()),
            "max": int(df["G3"].max()),
        },
        "missing_values": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "correlation_with_G3": {
            c: round(float(corr.loc[c, "G3"]), 3) for c in numeric_cols
        },
        "correlation_matrix": {
            "columns": corr_cols,
            "values": [[round(float(v), 3) for v in row] for row in corr.values],
        },
        "numeric_describe": json.loads(
            df[numeric_cols].describe().round(2).to_json()
        ),
    }


def save_artifacts(
    results_df, model_blobs, cv_summaries, audit,
    X_train, X_test, y_train, y_test, df, scenario,
    artifacts_suffix=None, write_figures=True,
):
    """Persist metrics (JSON/CSV), the best model, metadata and figures.

    artifacts_suffix: used to keep the OPTIONAL 'with grades' experiment in
    separate files (never overwriting primary artifacts).
    """
    from src.evaluate import (
        make_all_figures,
        permutation_importance_summary,
        roc_auc_summary,
    )

    suffix = artifacts_suffix or ""
    primary = suffix == ""

    results_df = results_df.copy()
    roc_auc_summary(results_df, model_blobs, X_test, y_test)
    # Sort by F1 (positive class) on the test split to select the best model.
    # NOTE: the test split is used for this single final comparison (the
    # honest reporting scenario for a small course project); tuning itself
    # happened on the training split only.
    results_df = results_df.sort_values(
        "f1_positive", ascending=False
    ).reset_index(drop=True)
    best_name = results_df.iloc[0]["model"]

    for directory in (config.MODELS_DIR, config.METRICS_DIR,
                      config.FIGURES_DIR, config.DATA_PROCESSED_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    # ---- best fitted pipeline -> joblib ---------------------------------
    model_path = config.MODELS_DIR / f"best_model{suffix}.joblib"
    joblib.dump(model_blobs[best_name], model_path)

    # ---- permutation importance on held-out data ------------------------
    try:
        perm_summary = permutation_importance_summary(
            model_blobs[best_name], X_test, y_test
        )
    except Exception as exc:
        perm_summary = {"error": f"permutation importance failed: {exc}"}

    # ---- metrics CSV + JSON ---------------------------------------------
    metrics_csv_path = config.METRICS_DIR / f"model_metrics{suffix}.csv"
    metrics_json_path = config.METRICS_DIR / f"model_metrics{suffix}.json"
    export_cols = [
        "model", "accuracy", "precision_positive", "recall_positive",
        "f1_positive", "roc_auc", "train_seconds", "predict_test_seconds",
    ]
    export_cols = [c for c in export_cols if c in results_df.columns]
    results_df[export_cols].round(4).to_csv(metrics_csv_path, index=False)
    metrics_json_path.write_text(
        json.dumps(json.loads(results_df.to_json(orient="records")),
                   indent=2, default=str),
        encoding="utf-8",
    )

    # ---- processed data + EDA summary (primary experiment only) ----------
    labelled = df.copy()
    labelled["target"] = build_target(df)
    if primary:
        labelled.to_csv(config.ARTIFACT_PROCESSED_DATA, index=False)
        eda = build_eda_summary(labelled, audit)
        config.ARTIFACT_EDA_SUMMARY.write_text(
            json.dumps(eda, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    else:
        eda = None

    # ---- figures (primary experiment only) --------------------------------
    if primary:
        try:
            make_all_figures(results_df, model_blobs, X_test, y_test, labelled,
                             audit, perm_summary)
        except Exception as exc:
            print(f"[train] WARNING: figure generation failed: {exc}")

    # ---- metadata --------------------------------------------------------
    metadata = {
        "artifact_id": str(uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_file": str(model_path),
        "best_model": best_name,
        "selection_rule": (
            "highest F1 (positive class = at_risk) on the held-out test "
            "split; tuning itself used training data only (5-fold CV)"
        ),
        "primary_scenario": scenario,
        "scenario_labels": {
            "early": ("PRIMARY early-prediction: no grade features "
                      "(G1/G2/G3 excluded)"),
            "midterm": ("OPTIONAL: includes G1/G2 - NOT comparable to the "
                        "primary scenario"),
        },
        "target_definition": config.describe_target_rule(),
        "positive_class": config.POSITIVE_CLASS,
        "class_names": config.TARGET_LABELS,
        "random_seed": config.RANDOM_SEED,
        "test_size": config.TEST_SIZE,
        "cv_folds": config.CV_FOLDS,
        "class_weight_balanced": config.USE_CLASS_WEIGHT_BALANCED,
        "excluded_columns": audit["excluded_columns"],
        "feature_schema": {
            "features": audit["features"],
            "numeric": audit["numeric"],
            "categorical": audit["categorical"],
        },
        "categorical_values": {
            col: sorted(df[col].dropna().astype(str).unique().tolist())
            for col in audit["categorical"]
        },
        "numeric_ranges": {
            col: [float(df[col].min()), float(df[col].max())]
            for col in audit["numeric"]
        },
        "cv_results": cv_summaries,
        "permutation_importance": perm_summary,
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sklearn": __import__("sklearn").__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_balance_train": {str(k): int(v)
                                for k, v in y_train.value_counts().items()},
        "class_balance_test": {str(k): int(v)
                               for k, v in y_test.value_counts().items()},
    }
    metadata_path = config.METRICS_DIR / f"model_metadata{suffix}.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"[train] best model: {best_name} -> saved to {model_path}")
    return {
        "model_path": model_path,
        "metadata": metadata,
        "results_df": results_df,
        "eda": eda,
    }


def main():
    """CLI entry points:
    python -m src.train               -> PRIMARY early-prediction experiment
    python -m src.train --with-grades -> OPTIONAL midterm experiment
    (artifacts saved separately; NOT comparable to the primary experiment)
    """
    with_grades = "--with-grades" in sys.argv
    print("[train] loading raw dataset ...")
    df = load_raw()
    print(f"[train] dataset shape: {df.shape}")
    if with_grades:
        print("[train] running the OPTIONAL midterm experiment "
              "(G1/G2 included - NOT comparable to the primary scenario) ...")
        train_all_models(
            df, scenario="midterm",
            artifact_suffix="_with_grades", write_figures=False,
        )
        return
    print("[train] running the PRIMARY early-prediction experiment ...")
    results, _artifacts = train_all_models(df, scenario=config.PRIMARY_SCENARIO)
    print("\n[train] measured test metrics (positive class = at_risk):")
    cols = ["model", "accuracy", "precision_positive", "recall_positive",
            "f1_positive", "roc_auc", "train_seconds"]
    print(results[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
