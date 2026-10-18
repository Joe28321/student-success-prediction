"""Tests for the prediction layer used by the Streamlit app."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import config
from src.predict import (
    ModelNotAvailableError,
    feature_schema,
    load_artifacts,
    predict_student,
    validate_input,
)


def _artifacts_exist() -> bool:
    return (
        config.ARTIFACT_MODEL.exists()
        and config.ARTIFACT_METADATA.exists()
    )


@pytest.fixture(scope="module")
def artifacts():
    if not _artifacts_exist():
        pytest.skip("artifacts missing - run 'python -m src.train' first")
    return load_artifacts()


@pytest.fixture(scope="module")
def metadata(artifacts):
    return artifacts[1]


def test_load_artifacts_missing_model_has_instructions(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ARTIFACT_MODEL", tmp_path / "none.joblib")
    with pytest.raises(ModelNotAvailableError) as excinfo:
        load_artifacts()
    message = str(excinfo.value)
    assert "How to fix" in message
    assert "python -m src.train" in message


def test_schema_matches_expected_features(metadata):
    features, numeric, categorical = feature_schema(metadata)
    assert "G3" not in features
    assert "G1" not in features and "G2" not in features
    assert set(metadata["excluded_columns"]) == {"G1", "G2", "G3"}
    assert set(numeric) | set(categorical) == set(features)


def _default_record(metadata) -> dict:
    features, numeric, categorical = feature_schema(metadata)
    record = {}
    for col in numeric:
        record[col] = metadata["numeric_ranges"][col][0]
    for col in categorical:
        record[col] = metadata["categorical_values"][col][0]
    return record


def test_valid_record_passes_validation(metadata):
    problems = validate_input(_default_record(metadata), metadata)
    assert problems == [], problems


def test_validation_rejects_bad_age(metadata):
    record = _default_record(metadata)
    if "age" in record:
        record["age"] = 200
        problems = validate_input(record, metadata)
        assert any("age" in p for p in problems)


def test_validation_rejects_unknown_category(metadata):
    record = _default_record(metadata)
    col = metadata["feature_schema"]["categorical"][0]
    record[col] = "!!!"
    problems = validate_input(record, metadata)
    assert any(col in p for p in problems)


def test_validation_rejects_missing_field(metadata):
    record = _default_record(metadata)
    if "age" in record:
        del record["age"]
    problems = validate_input(record, metadata)
    assert problems


def test_predict_student_output_shape(metadata, artifacts):
    pipeline, _ = artifacts
    result = predict_student(_default_record(metadata), pipeline, metadata)
    assert result["predicted_class"] in config.TARGET_LABELS
    if "probabilities" in result:
        probs = result["probabilities"]
        assert set(probs.keys()) == set(config.TARGET_LABELS)
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        # probabilities must come from the trained pipeline, never invented
        proba = pipeline.predict_proba(
            pd.DataFrame([_default_record(metadata)])
        )
        assert result["probabilities"]["at_risk"] \
            == pytest.approx(round(float(proba[0][0]), 4))
