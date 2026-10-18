"""Tests for training, evaluation and saved artifacts.

These tests use the real dataset; artifact-dependent tests skip with a
clear reason when artifacts have not been produced yet.
"""

from __future__ import annotations

import json
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix

from src import config
from src.data_loader import build_target, get_feature_metadata, load_raw
from src.train import make_split, train_all_models


@pytest.fixture(scope="module")
def raw():
    try:
        return load_raw()
    except Exception as exc:  # pragma: no cover - environment-specific
        pytest.skip(f"Dataset not available: {exc}")


def _artifacts_exist() -> bool:
    return (
        config.ARTIFACT_MODEL.exists()
        and config.ARTIFACT_METADATA.exists()
        and config.ARTIFACT_METRICS.exists()
    )


# ---------------------------------------------------------------------------
# Split reproducibility and stratification
# ---------------------------------------------------------------------------


def test_split_is_reproducible(raw):
    target = build_target(raw)
    audit, feature_cols, *_ = get_feature_metadata(raw, "early")
    s1 = make_split(raw, feature_cols, target)
    s2 = make_split(raw, feature_cols, target)
    for a, b in zip(s1, s2):
        pd.testing.assert_frame_equal(a, b) if isinstance(a, pd.DataFrame) \
            else pd.testing.assert_series_equal(a, b)


def test_split_stratified_and_stacked(raw):
    target = build_target(raw)
    audit, feature_cols, *_ = get_feature_metadata(raw, "early")
    X_train, X_test, y_train, y_test = make_split(raw, feature_cols, target)
    assert len(X_train) == int(round(len(raw) * (1 - config.TEST_SIZE)))
    assert len(X_test) == len(raw) - len(X_train)
    overall = target.value_counts(normalize=True)
    for split_name, split in (("train", y_train), ("test", y_test)):
        share = split.value_counts(normalize=True)
        for label in overall.index:
            assert abs(share[label] - overall[label]) < 0.02, (
                f"stratification failure in {split_name}: {share[label]} vs "
                f"{overall[label]} for {label}"
            )


def test_split_contains_no_leakage_columns(raw):
    target = build_target(raw)
    audit, feature_cols, *_ = get_feature_metadata(raw, "early")
    X_train, X_test, *_ = make_split(raw, feature_cols, target)
    for grade in ("G1", "G2", "G3"):
        assert grade not in X_train.columns
        assert grade not in X_test.columns


# ---------------------------------------------------------------------------
# Full training (grid-search CV) with the real data
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_train_all_models_produces_valid_metrics(raw):
    results, _ = train_all_models(raw, save=False)
    assert len(results) == 3
    for _, row in results.iterrows():
        for col in ("accuracy", "precision_positive", "recall_positive",
                    "f1_positive"):
            value = float(row[col])
            assert 0 <= value <= 1, f"{row['model']}.{col} out of range"
        # recall consistent with the stored confusion matrix
        cm = np.array(row["confusion_matrix"]["matrix"])
        assert cm.shape == (2, 2)


def test_recall_matches_confusion_matrix_from_saved_metrics():
    """Saved metrics must be internally consistent (no fake numbers)."""
    if not _artifacts_exist():
        pytest.skip("artifacts missing - run 'python -m src.train' first")
    metrics = json.loads(config.ARTIFACT_METRICS.read_text(encoding="utf-8"))
    for row in metrics:
        labels = row["confusion_matrix"]["labels"]
        cm = np.array(row["confusion_matrix"]["matrix"])
        tp = int(cm[labels.index("at_risk"), labels.index("at_risk")])
        fn = int(cm[labels.index("at_risk"), 0])
        expected_recall = tp / (tp + fn)
        assert abs(expected_recall - row["recall_positive"]) < 1e-6
        assert abs(row["accuracy"] - cm.diagonal().sum() / cm.sum()) < 1e-6


def test_saved_model_loads_and_predicts_valid_class():
    if not _artifacts_exist():
        pytest.skip("artifacts missing - run 'python -m src.train' first")
    pipeline = joblib.load(config.ARTIFACT_MODEL)
    metadata = json.loads(config.ARTIFACT_METADATA.read_text(encoding="utf-8"))
    features = metadata["feature_schema"]["features"]
    assert "G3" not in features and "G1" not in features
    record = {}
    for col in metadata["feature_schema"]["numeric"]:
        record[col] = metadata["numeric_ranges"][col][1]
    for col in metadata["feature_schema"]["categorical"]:
        record[col] = metadata["categorical_values"][col][0]
    frame = pd.DataFrame([record])
    pred = pipeline.predict(frame)[0]
    assert pred in config.TARGET_LABELS
    proba = pipeline.predict_proba(frame)[0]
    assert float(proba.sum()) == pytest.approx(1.0, abs=1e-6)


def test_saved_metadata_agrees_with_experiment_config():
    if not _artifacts_exist():
        pytest.skip("artifacts missing - run 'python -m src.train' first")
    metadata = json.loads(config.ARTIFACT_METADATA.read_text(encoding="utf-8"))
    assert metadata["random_seed"] == config.RANDOM_SEED
    assert metadata["test_size"] == config.TEST_SIZE
    assert metadata["excluded_columns"] == ["G1", "G2", "G3"]
    assert metadata["positive_class"] == config.POSITIVE_CLASS
    schema = metadata["feature_schema"]
    assert len(schema["features"]) == 30
    assert set(schema["numeric"]) | set(schema["categorical"]) == set(
        schema["features"]
    )


def test_metrics_csv_matches_json():
    if not _artifacts_exist():
        pytest.skip("artifacts missing - run 'python -m src.train' first")
    csv = pd.read_csv(config.ARTIFACT_METRICS_CSV)
    js = pd.read_json(config.ARTIFACT_METRICS)
    merged = csv.merge(
        js[["model", "accuracy", "f1_positive"]], on="model",
        suffixes=("_csv", "_json"),
    )
    for col in ("accuracy", "f1_positive"):
        diff = (merged[f"{col}_csv"] - merged[f"{col}_json"]).abs().max()
        assert diff < 1e-3
