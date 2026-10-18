"""Data loading and validation for the Student Success Predictor.

Responsibilities:
- Locate the raw UCI CSV file(s).
- Fail with a CLEAR, actionable message when the file is missing
  (never silently substitute fabricated data).
- Basic sanity validation of the schema (columns, dtypes, target).
- Build the model design matrix for a given prediction scenario
  (with an explicit leakage audit).
"""

from __future__ import annotations

import io
import urllib.request
import zipfile

import pandas as pd

from src import config

REQUIRED_URL = config.DATASET_INFO["source_url"]


class DataNotFoundError(FileNotFoundError):
    """Raised when the raw dataset file is missing or unreadable."""


def _download_zip(target_dir):
    """Download + extract the official UCI zip. Returns True on success."""
    target_dir = str(target_dir)
    zip_url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
    try:
        req = urllib.request.Request(
            zip_url, headers={"User-Agent": "Mozilla/5.0 (educational project)"}
        )
        import io as _io
        raw = urllib.request.urlopen(req, timeout=60).read()
        zf = zipfile.ZipFile(_io.BytesIO(raw))
        names = {n.split("/")[-1] for n in zf.namelist()}
        if "student-mat.csv" not in names:
            raise RuntimeError("Unexpected archive layout from UCI.")
        zf.extractall(target_dir)
        # The archive nests one zip inside another -> extract that too.
        inner = os.path.join(target_dir, "student.zip")
        if os.path.exists(inner):
            zipfile.ZipFile(inner).extractall(target_dir)
        return True
    except Exception as exc:  # network error, 4xx/5xx, bad archive...
        print(f"[data_loader] download failed: {exc}")
        return False


import os  # noqa: E402  (kept after helpers for readable grouping)


def get_raw_csv_path(subject: str = config.PRIMARY_SUBJECT):
    """Return the Path of the raw CSV, or None if it is not present.

    If the file is missing, try to download it from the official UCI source
    (internet permitting). We never fabricate data.
    """
    path = config.DATA_RAW_DIR / config.DATASET_FILENAMES[subject]
    if path.exists():
        return path
    print(f"[data_loader] {path} not found - attempting official download...")
    if _download_zip(config.DATA_RAW_DIR):
        return path if path.exists() else None
    return None


def validate_raw(df: pd.DataFrame) -> dict:
    """Validate the schema of a raw UCI student DataFrame.

    Returns a report dict; raises DataNotFoundError on a broken schema.
    """
    expected = set(config.FEATURE_DICTIONARY.keys())
    missing = sorted(expected - set(df.columns))
    if missing:
        raise DataNotFoundError(
            f"The dataset file does not contain the expected columns. "
            f"Missing: {missing}. Expected the official UCI "
            f"'student performance' schema with 33 columns."
        )
    report = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_missing": int(df.isna().sum().sum()),
        "g3_min": float(df["G3"].min()),
        "g3_max": float(df["G3"].max()),
        "duplicates": int(df.duplicated().sum()),
    }
    report["target_within_scale"] = (
        0 <= report["g3_min"] and report["g3_max"] <= 20
    )
    return report


def load_raw(subject: str = config.PRIMARY_SUBJECT) -> pd.DataFrame:
    """Load a raw subject dataset (semicolon-separated UCI CSV)."""
    path = get_raw_csv_path(subject)
    if path is None:
        raise DataNotFoundError(
            f"Dataset file '{config.DATASET_FILENAMES[subject]}' was not found "
            f"in {config.DATA_RAW_DIR} and the automatic download from UCI "
            "failed.\n\nHow to fix it manually:\n"
            f"1. Open {REQUIRED_URL}\n"
            "2. Download the dataset (ZIP) and extract it.\n"
            "3. Copy 'student-mat.csv' (and optionally 'student-por.csv') "
            f"into {config.DATA_RAW_DIR}\n"
            "4. Re-run the training script or the application."
        )
    df = pd.read_csv(path, sep=";")
    validate_raw(df)
    return df


def build_target(df: pd.DataFrame) -> pd.Series:
    """Create the binary target from G3 using the project rule.

    at_risk     : G3 < 10  (positive class)
    not_at_risk : G3 >= 10
    """
    return (df["G3"] < config.G3_PASS_THRESHOLD).map(
        {False: "not_at_risk", True: "at_risk"}
    ).rename("target")


def get_feature_metadata(df: pd.DataFrame, scenario: str = config.PRIMARY_SCENARIO):
    """Split the (leakage-safe) feature columns into numeric / categorical.

    Exclusions for the PRIMARY early-prediction scenario:
      - G3  : directly defines the target            -> direct leakage
      - G1, G2: intermediate grades not available at
        the intended early prediction time           -> temporal leakage

    Exclusions, by design (documented in config.py):
      - G3  : directly defines the target            -> direct leakage
      - G1, G2: intermediate grades not available at
        the intended early prediction time           -> temporal leakage

    Every excluded column is audited and reported, never silently reused.
    """
    # G3 always excluded; scenario adds further exclusions (early: also G1/G2)
    excluded = {"G3"} | set(config.SCENARIOS[scenario]["additional_exclusions"])
    excluded = sorted(excluded)

    excluded_present = [c for c in excluded if c in df.columns]
    feature_cols = [c for c in df.columns if c not in excluded_present]

    numeric_cols = [
        c for c in feature_cols
        if str(df[c].dtype).startswith(("int", "float"))
    ]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    audit = {
        "scenario": scenario,
        "excluded_columns": excluded_present,
        "exclusion_reasons": {
            "G3": "defines the target (at_risk = G3 < 10) - direct leakage",
            "G1": "intermediate grade, unavailable at early prediction time - temporal leakage (excluded in the primary scenario)",
            "G2": "intermediate grade, unavailable at early prediction time - temporal leakage (excluded in the primary scenario)",
        },
        "n_features": len(feature_cols),
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "features": feature_cols,
    }
    return audit, feature_cols, numeric_cols, categorical_cols


def load_processed(path=None) -> pd.DataFrame:
    """Load the processed (target-labelled) dataset written by training."""
    path = path or config.ARTIFACT_PROCESSED_DATA
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset '{path}' not found. Run 'python -m src.train' "
            "first - it writes data/processed/student_mat_processed.csv."
        )
    return pd.read_csv(path)


def download_data_for_setup(subject: str = config.PRIMARY_SUBJECT) -> str:
    """Setup helper used by tests / app: report dataset availability."""
    path = get_raw_csv_path(subject)
    return "available" if path is not None else "missing"
