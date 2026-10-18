"""Prediction module for the Streamlit application.

Loads the saved pipeline + metadata produced by `python -m src.train` and
turns one student's attributes into a risk category. It NEVER invents a
prediction: if the artifacts are missing we raise a clear, actionable
error instead of guessing.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src import config


class ModelNotAvailableError(FileNotFoundError):
    """Raised when the trained model / metadata artifacts are missing."""


def load_artifacts():
    """Load (pipeline, metadata). Fail with actionable instructions."""
    model_path = Path(config.ARTIFACT_MODEL)
    metadata_path = Path(config.ARTIFACT_METADATA)
    missing = [str(p) for p in (model_path, metadata_path) if not p.exists()]
    if missing:
        raise ModelNotAvailableError(
            "Trained model artifacts not found:\n  - "
            + "\n  - ".join(missing)
            + "\n\nHow to fix:\n"
              "1. Activate the virtual environment:  .\\.venv\\Scripts\\Activate.ps1\n"
              "2. Make sure the dataset is in data/raw/ (see README).\n"
              "3. Train the models:  python -m src.train\n"
              "4. Restart this application."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pipeline = joblib.load(model_path)
    return pipeline, metadata


def feature_schema(metadata):
    """Return (features, numeric, categorical) from saved metadata."""
    schema = metadata["feature_schema"]
    return schema["features"], schema["numeric"], schema["categorical"]


def validate_input(row: dict, metadata) -> list[str]:
    """Validate one student record against the saved schema.

    Returns a list of human-readable problems (empty list = valid).
    """
    features, numeric, categorical = feature_schema(metadata)
    problems = []
    for col in features:
        if col not in row or row[col] is None or (
            isinstance(row[col], float) and pd.isna(row[col])
        ):
            problems.append(f"Missing value for required field: {col}")
            continue
        if col in numeric:
            lo, hi = metadata["numeric_ranges"].get(col, [None, None])
            try:
                val = float(row[col])
            except (TypeError, ValueError):
                problems.append(f"{col}: expected a number, got {row[col]!r}")
                continue
            if lo is not None and val < lo:
                problems.append(f"{col}: {val} is below the expected range ({lo})")
            if hi is not None and val > hi:
                problems.append(f"{col}: {val} is above the expected range ({hi})")
        elif col in categorical:
            allowed = metadata["categorical_values"].get(col)
            val = str(row[col])
            if allowed and val not in allowed:
                problems.append(
                    f"{col}: '{val}' is not one of the expected values {allowed}"
                )
    return problems


def predict_student(record: dict, pipeline, metadata):
    """Predict the risk category for one student record.

    Returns a dict:
      predicted_class, probabilities (both classes), and the raw row used.
    The probabilities come from the fitted pipeline's predict_proba
    (supported by all three implemented algorithms); they are estimates,
    not certainties.
    """
    features, numeric, _categorical = feature_schema(metadata)
    row = {f: record[f] for f in features}
    frame = pd.DataFrame([row])
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col])

    pred = pipeline.predict(frame)[0]
    result = {"predicted_class": str(pred)}
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(frame)[0]
        classes = list(pipeline.classes_)
        result["probabilities"] = {str(c): round(float(p), 4)
                                   for c, p in zip(classes, proba)}
    return result
