"""F1 Strategy Board - dashboard entry point.

Run from the project root::

    streamlit run app/Home.py

This script owns everything shared by all pages: page config, styling, the
sidebar filters and the navigation. Each page lives in ``app/app_pages``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # make ``src`` and ``app`` importable however Streamlit is launched
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import ui  # noqa: E402

st.set_page_config(page_title="F1 Strategy Board", page_icon=":material/sports_score:", layout="wide",
                   initial_sidebar_state="auto")
ui.inject_css()

try:
    tables = ui.get_tables()
except FileNotFoundError as exc:
    st.error(f"{exc}", icon=":material/error:")
    st.stop()

pages = {
    "": [st.Page("app_pages/overview.py", title="Overview", icon=":material/home:", default=True)],
    "Data": [
        st.Page("app_pages/data_sources.py", title="Where the data comes from", icon=":material/database:"),
        st.Page("app_pages/cleaning_report.py", title="How it was cleaned", icon=":material/cleaning_services:"),
    ],
    "Explore": [
        st.Page("app_pages/eda_explorer.py", title="Explore the data", icon=":material/query_stats:"),
        st.Page("app_pages/correlations.py", title="What moves together", icon=":material/hub:"),
    ],
    "Results": [
        st.Page("app_pages/ranked_insights.py", title="What drives results", icon=":material/leaderboard:"),
        st.Page("app_pages/reliability_risk.py", title="Retirement risk", icon=":material/car_crash:"),
    ],
    "About": [st.Page("app_pages/methodology.py", title="How it works", icon=":material/menu_book:")],
}
page = st.navigation(pages, position="sidebar")
ui.sidebar_filters(tables)
st.session_state["_titled"] = False      # reset the heading/title pairing for this run
page.run()
