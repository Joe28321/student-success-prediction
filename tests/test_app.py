"""Smoke tests for the Streamlit application (uses streamlit.testing.v1.AppTest)."""

from __future__ import annotations

from pathlib import Path

import pytest

APP_PAGE = str(Path(__file__).resolve().parents[1] / "app.py")

PAGES = [
    "Home",
    "Student Risk Assessment",
    "Algorithm Comparison",
    "Data Insights",
    "Research & Methodology",
]


def _artifacts_ready() -> bool:
    from pathlib import Path

    from src import config

    return config.ARTIFACT_MODEL.exists() and config.ARTIFACT_METADATA.exists()


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES)
def test_app_page_runs_without_exception(page):
    if not _artifacts_ready():
        pytest.skip("artifacts missing - run 'python -m src.train' first")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PAGE, default_timeout=90)
    at.run()
    assert not at.exception, f"{page}: {at.exception}"
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page}: {at.exception}"
