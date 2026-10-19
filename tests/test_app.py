"""Dashboard smoke tests: every page and every EDA tab must render without an exception.

Uses Streamlit's headless ``AppTest`` runner on the bundled sample, i.e. exactly
what a user without network access would see.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import config

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["overview", "data_sources", "cleaning_report", "eda_explorer", "correlations", "reliability_risk",
         "ranked_insights", "methodology"]
EDA_TABS = ["Distributions", "Relationships", "Outliers", "Trends", "Team groups", "Report"]

pytestmark = pytest.mark.skipif(not (config.SAMPLE_DIR / "results.parquet").exists(),
                                reason="bundled sample missing")


@pytest.fixture(scope="module")
def sample_mode(tmp_path_factory):
    """Point the app at an empty raw cache so it must fall back to data/sample."""
    tmp = tmp_path_factory.mktemp("f1sb_app")
    patch = pytest.MonkeyPatch()
    patch.setattr(config, "RAW_DIR", tmp / "raw")
    patch.setattr(config, "SYNTHETIC_DIR", tmp / "synthetic")
    patch.setattr(config, "PROCESSED_DIR", tmp / "processed")
    patch.setattr(config, "WATCHLIST_DIR", tmp / "watchlist")
    patch.setattr(config, "REPORTS_DIR", tmp / "reports")
    patch.chdir(ROOT)
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    patch.undo()
    st.cache_resource.clear()
    st.cache_data.clear()


def _open(page: str) -> AppTest:
    app = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=180)
    app.run()
    if page != "overview":
        app.switch_page(f"app_pages/{page}.py")
        app.run()
    return app


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_in_sample_mode(sample_mode, page: str) -> None:
    app = _open(page)
    assert not app.exception, f"{page}: {[e.value for e in app.exception]}"
    assert len(app.error) == 0, f"{page}: {[e.value for e in app.error]}"


@pytest.mark.parametrize("tab", EDA_TABS)
def test_eda_tab_renders_in_sample_mode(sample_mode, tab: str) -> None:
    app = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=240)
    app.query_params["eda_tab"] = tab
    app.run()
    app.switch_page("app_pages/eda_explorer.py")
    app.run()
    assert not app.exception, f"{tab}: {[e.value for e in app.exception]}"


@pytest.mark.parametrize("page", PAGES)
def test_page_survives_narrow_filters(sample_mode, page: str) -> None:
    """One team and one season leaves very little data; pages must degrade gracefully."""
    app = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=240)
    app.session_state["f_teams"] = ["Ferrari"]
    app.session_state["f_seasons"] = (2021, 2021)
    app.run()
    if page != "overview":
        app.switch_page(f"app_pages/{page}.py")
        app.run()
    assert not app.exception, f"{page}: {[e.value for e in app.exception]}"


@pytest.mark.parametrize("tab", EDA_TABS)
def test_eda_tab_survives_narrow_filters(sample_mode, tab: str) -> None:
    app = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=240)
    app.query_params["eda_tab"] = tab
    app.session_state["f_teams"] = ["Ferrari", "Mercedes"]
    app.session_state["f_seasons"] = (2020, 2021)
    app.run()
    app.switch_page("app_pages/eda_explorer.py")
    app.run()
    assert not app.exception, f"{tab}: {[e.value for e in app.exception]}"


def test_filters_narrow_the_view(sample_mode) -> None:
    app = _open("overview")
    app.sidebar.multiselect(key="f_teams").set_value(["Ferrari"]).run()
    assert not app.exception
    assert app.session_state["f_teams"] == ["Ferrari"]
    drivers = app.sidebar.multiselect(key="f_drivers").options
    assert 0 < len(drivers) < 20                                 # only Ferrari's drivers are offered
