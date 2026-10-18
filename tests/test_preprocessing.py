"""Tests for data loading, leakage policy, target rule and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loader import (
    DataNotFoundError,
    build_target,
    get_feature_metadata,
    load_raw,
    validate_raw,
)
from src.preprocessing import build_pipeline

SAMPLES = {
    "GP": {"GP", "MS"},
    "sex": {"F", "M"},
    "address": {"U", "R"},
    "famsize": {"LE3", "GT3"},
    "Pstatus": {"T", "A"},
    "schoolsup": {"yes", "no"},
}


@pytest.fixture(scope="module")
def raw():
    try:
        return load_raw()
    except DataNotFoundError as exc:
        pytest.skip(f"Dataset not available in this environment: {exc}")


# ---------------------------------------------------------------------------
# Missing-file handling must be clear and actionable (no fabricated data)
# ---------------------------------------------------------------------------


def test_missing_file_raises_clear_message(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path)
    monkeypatch.setitem(config.DATASET_FILENAMES, "mathematics", "nope.csv")
    monkeypatch.setattr("src.data_loader._download_zip", lambda d: False)
    with pytest.raises(DataNotFoundError) as excinfo:
        load_raw()
    message = str(excinfo.value)
    assert "not found" in message
    # instructions present
    assert "How to fix" in message
    assert "student-mat.csv" in message


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_validate_raw_schema(raw):
    report = validate_raw(raw)
    assert report["n_rows"] == 395
    assert report["n_columns"] == 33
    assert report["n_missing"] == 0
    assert report["target_within_scale"] is True


def test_validate_raw_broken_schema():
    bad = pd.DataFrame({"school": ["GP"], "sex": ["F"]})
    with pytest.raises(DataNotFoundError):
        validate_raw(bad)


# ---------------------------------------------------------------------------
# Target definition (G3 < 10 => at_risk)
# ---------------------------------------------------------------------------


def test_target_rule():
    df = pd.DataFrame({"G3": [0, 9, 10, 20]})
    target = build_target(df)
    assert list(target) == ["at_risk", "at_risk", "not_at_risk", "not_at_risk"]
    assert target.name == "target"


def test_target_matches_expected_counts(raw):
    target = build_target(raw)
    counts = target.value_counts()
    assert counts["at_risk"] == int((raw["G3"] < 10).sum())
    assert counts["not_at_risk"] == int((raw["G3"] >= 10).sum())
    assert set(target.unique()).issubset(set(config.TARGET_LABELS))


# ---------------------------------------------------------------------------
# Data-leakage audit (the core safeguard of this project)
# ---------------------------------------------------------------------------


def test_primary_scenario_excludes_all_grades(raw):
    audit, feature_cols, numeric_cols, categorical_cols = get_feature_metadata(
        raw, "early"
    )
    for grade in ("G1", "G2", "G3"):
        assert grade not in feature_cols, (
            f"LEAKAGE: {grade} must not be a feature in the early scenario"
        )
    assert audit["excluded_columns"] == ["G1", "G2", "G3"]
    assert len(feature_cols) == 30
    assert set(feature_cols) == set(numeric_cols) | set(categorical_cols)


def test_midterm_scenario_includes_g1_g2_only(raw):
    audit, feature_cols, _, _ = get_feature_metadata(raw, "midterm")
    assert "G3" not in feature_cols
    assert "G1" in feature_cols and "G2" in feature_cols
    assert set(audit["excluded_columns"]) == {"G3"}


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------


def test_pipeline_structure(raw):
    audit, feature_cols, numeric_cols, categorical_cols = get_feature_metadata(
        raw, "early"
    )
    X = raw[feature_cols]
    y = build_target(raw)
    pipeline = build_pipeline("Random Forest", numeric_cols, categorical_cols)
    fitted = pipeline.fit(X, y)
    assert "preprocess" in fitted.named_steps
    assert "classifier" in fitted.named_steps
    transformed_shape = fitted.named_steps["preprocess"].transform(X).shape
    assert transformed_shape[0] == len(X)
    # one-hot encoding expands categorical columns
    assert transformed_shape[1] > len(feature_cols)


def test_pretrained_pipeline_handles_unknown_category(raw):
    """OneHotEncoder(handle_unknown='ignore') makes saved models load-safe."""
    audit, feature_cols, numeric_cols, categorical_cols = get_feature_metadata(
        raw, "early"
    )
    X, y = raw[feature_cols], build_target(raw)
    pipeline = build_pipeline("Logistic Regression", numeric_cols, categorical_cols)
    pipeline.fit(X, y)
    corrupted = X.iloc[[0]].copy()
    corrupted["sex"] = "X"  # category never seen in training
    prediction = pipeline.predict(corrupted)
    assert prediction.shape == (1,)


def test_numeric_imputation(raw):
    audit, feature_cols, numeric_cols, categorical_cols = get_feature_metadata(
        raw, "early"
    )
    X, y = raw[feature_cols], build_target(raw)
    X_missing = X.copy()
    X_missing.loc[X_missing.index[0], "age"] = np.nan
    pipeline = build_pipeline("Decision Tree", numeric_cols, categorical_cols)
    pipeline.fit(X_missing, y)  # must not raise on missing numeric values
    preds = pipeline.predict(X_missing.head(3))
    assert set(preds).issubset(set(config.TARGET_LABELS))
