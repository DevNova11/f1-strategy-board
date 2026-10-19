"""Shared dashboard helpers: styling, cached data access, filters and chart cards."""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, feedback, insights, model, pipeline, sources
from src.eda import EDAResult, outliers, reduction

PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
SYNTHETIC_NOTE = ("Budget, logistics and simulation figures on this page are made up by a seeded generator "
                  "(no team publishes them). Links involving them were planted to test the method, so read them "
                  "as a demonstration, not as facts about real teams.")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_CAPS = re.compile(r"\b(?:REAL|SYNTHETIC)\b")


# ------------------------------------------------------------------- styling
def inject_css() -> None:
    """Load the custom stylesheet from ``assets/style.css``."""
    css = (config.ASSETS_DIR / "style.css").read_text(encoding="utf-8")
    st.html(f"<style>{css}</style>")


def header(title: str, subtitle: str) -> None:
    """Page title with a one-line explanation of what the page answers."""
    st.html(f'<div class="f1-header"><div class="f1-title">{html.escape(title)}</div>'
            f'<p class="f1-sub">{html.escape(subtitle)}</p></div>')


def section(title: str, note: str = "") -> None:
    """Section heading, with an optional plain-language line underneath."""
    extra = f"<p>{html.escape(note)}</p>" if note else ""
    st.html(f'<div class="f1-section"><h3>{html.escape(title)}</h3>{extra}</div>')
    st.session_state["_titled"] = True      # the next chart can drop its own, now redundant, title


def _plain(text: str) -> str:
    """Soften the all-caps data labels (REAL, SYNTHETIC) when they appear inside a sentence."""
    return _CAPS.sub(lambda m: m.group(0).lower(), text.replace(" * ", " "))


def takeaway(text: str) -> None:
    """The note under a chart: the finding first, supporting detail after it, quieter."""
    sentences = _SENTENCE.split(_plain(text).strip(), maxsplit=1)
    lead = html.escape(sentences[0])
    rest = f" {html.escape(sentences[1])}" if len(sentences) > 1 else ""
    st.html(f'<p class="f1-note"><strong>{lead}</strong>{rest}</p>')


def answer(text_html: str) -> None:
    """A short, prominent answer paragraph (used on the overview)."""
    st.html(f'<div class="f1-answer">{text_html}</div>')


def synthetic_banner(text: str = SYNTHETIC_NOTE) -> None:
    """One-line notice that a page uses synthetic data."""
    st.html(f'<div class="f1-notice"><span class="f1-pill synthetic">Synthetic data</span><span>{text}</span></div>')


def badge(origin: str) -> str:
    """Small pill for real or synthetic data."""
    return f'<span class="f1-pill {origin.lower()}">{"Real data" if origin == "REAL" else "Synthetic data"}</span>'


def show(result: EDAResult, key: str, table: bool = False) -> None:
    """Render an EDA result as a card: chart, the note under it, optional numbers."""
    fig = result.fig
    if st.session_state.pop("_titled", False):  # a section heading already says what this chart is
        fig = go.Figure(fig)
        fig.update_layout(title_text="", margin=dict(t=56 if fig.layout.showlegend is not False else 24))
    with st.container(border=True):
        st.plotly_chart(fig, theme=None, key=key, config=PLOTLY_CONFIG)
        takeaway(result.takeaway)
        if table and result.table is not None and not result.table.empty:
            with st.expander("Show the numbers"):
                st.dataframe(result.table, hide_index=True)


def kpi_row(items: list[dict]) -> None:
    """A row of KPI cards. Each item: label, value, optional delta, delta_color, help."""
    for col, item in zip(st.columns(len(items)), items):
        colour = item.get("delta_color", "normal")
        col.metric(item["label"], item["value"], item.get("delta"), delta_color=colour, help=item.get("help"),
                   border=True, delta_arrow="off" if colour == "off" else "auto")


# --------------------------------------------------------------- cached data
@st.cache_resource(show_spinner="Building analysis tables...")
def get_tables() -> pipeline.Tables:
    """Cleaned, joined tables shared by every session (treated as read-only)."""
    return pipeline.load()


@st.cache_resource(show_spinner=False)
def get_sources() -> sources.SourceBundle:
    """Raw sources, for the Data Sources page previews."""
    return sources.load_all()


@st.cache_resource(show_spinner="Training the reliability model...")
def get_reliability() -> model.ReliabilityModel:
    """Reliability model trained once on all seasons (filters only change what is shown)."""
    return model.train_reliability(get_tables().driver_race)


@dataclass
class View:
    """The tables after the sidebar filters, plus a hashable key for caching."""

    races: pd.DataFrame
    driver_race: pd.DataFrame
    team_race: pd.DataFrame
    team_season: pd.DataFrame
    component: pd.DataFrame
    laps: pd.DataFrame
    stints: pd.DataFrame
    pitstops: pd.DataFrame
    key: tuple
    seasons: tuple[int, int]
    meta: dict


def _apply(frame: pd.DataFrame, seasons: tuple[int, int], teams: tuple[str, ...], drivers: tuple[str, ...],
           exclude: frozenset[str]) -> pd.DataFrame:
    mask = frame["season"].between(*seasons)
    if teams and "team" in frame:
        mask &= frame["team"].isin(teams)
    if drivers and "driver_id" in frame:
        mask &= frame["driver_id"].isin(drivers)
    if exclude and "team_race_key" in frame:
        mask &= ~frame["team_race_key"].isin(exclude)
    return frame[mask]


def sidebar_filters(tables: pipeline.Tables) -> None:
    """Season, team and driver filters. State lives in ``st.session_state``."""
    dr = tables.driver_race
    lo, hi = int(dr["season"].min()), int(dr["season"].max())
    with st.sidebar:
        st.html('<div class="f1-brand"><b>F1 Strategy Board</b><span>Seasons 2019 to 2025</span></div>')
        head, reset = st.columns([3, 2], vertical_alignment="center")
        head.markdown("**Filters**")
        if reset.button("Reset", key="f_reset", type="tertiary", icon=":material/restart_alt:"):
            for name in ("f_seasons", "f_teams", "f_drivers", "f_exclude_watch"):
                st.session_state.pop(name, None)
            st.rerun()
        if lo < hi:
            seasons = st.slider("Seasons", lo, hi, (lo, hi), key="f_seasons")
        else:
            seasons = (lo, hi)
            st.session_state["f_seasons"] = seasons
        in_range = dr[dr["season"].between(*seasons)]
        team_options = sorted(in_range["team"].dropna().unique())
        st.session_state["f_teams"] = [t for t in st.session_state.get("f_teams", []) if t in team_options]
        teams = st.multiselect("Teams", team_options, key="f_teams", placeholder="All teams")
        pool = in_range[in_range["team"].isin(teams)] if teams else in_range
        names = pool.drop_duplicates("driver_id").set_index("driver_id")["driver_name"].sort_values()
        st.session_state["f_drivers"] = [d for d in st.session_state.get("f_drivers", []) if d in names.index]
        st.multiselect("Drivers", list(names.index), key="f_drivers", placeholder="All drivers",
                       format_func=lambda d: names.get(d, d),
                       help="Applies to driver-level views: laps, stints, pit stops and reliability risk.")
        st.toggle("Hide watchlisted rows", key="f_exclude_watch",
                  help="Rows you flagged on the Outliers tab are left out of every chart while this is on.")
        st.divider()
        mode = tables.meta.get("mode", "sample")
        st.caption(f"{'Full download' if mode == 'raw' else 'Bundled sample'}: {tables.meta.get('races', 0)} races, "
                   f"{tables.meta.get('races_with_laps', 0)} with lap-by-lap data.")
        st.caption("Budget, logistics and simulation data are synthetic.")


def view() -> View:
    """Apply the sidebar filters to the cached tables."""
    tables = get_tables()
    dr = tables.driver_race
    seasons = tuple(st.session_state.get("f_seasons", (int(dr["season"].min()), int(dr["season"].max()))))
    teams = tuple(st.session_state.get("f_teams", []))
    drivers = tuple(st.session_state.get("f_drivers", []))
    exclude = frozenset(feedback.load_watchlist()["key"].astype(str)) if st.session_state.get("f_exclude_watch") else frozenset()
    frames = {name: _apply(frame, seasons, teams, drivers, exclude) for name, frame in tables.frames().items()}
    return View(**frames, key=(seasons, teams, drivers, tuple(sorted(exclude))), seasons=seasons, meta=tables.meta)


def require(frame: pd.DataFrame, what: str = "rows") -> None:
    """Stop the page with a friendly message when the filters leave nothing to show."""
    if frame.empty:
        st.warning(f"No {what} match the current filters. Widen the season range or clear the team and driver filters.",
                   icon=":material/filter_alt_off:")
        st.stop()


# ----------------------------------------------------- cached heavy analyses
def _state_key() -> tuple:
    """Everything a chart can depend on: the sidebar filters, the page's own
    widgets and, when watchlisted rows are excluded, the watchlist itself."""
    simple = (str, int, float, bool, tuple, list, type(None))
    state = tuple(sorted((str(k), repr(val)) for k, val in st.session_state.items()
                         if isinstance(val, simple) and not str(k).startswith("_")))
    path = feedback.watchlist_path()
    return state + (("watchlist", path.stat().st_mtime_ns if path.exists() else 0),)


@st.cache_data(ttl=3600, max_entries=400, show_spinner=False)
def _memo(name: str, state: tuple, _fn: Callable[[], Any]) -> Any:
    return _fn()


def memo(name: str, fn: Callable[[], Any]) -> Any:
    """Compute ``fn()`` once per combination of filters and widget values.

    Charts are recomputed only when something they could depend on changes, so
    switching tabs or revisiting a page is instant.
    """
    return _memo(name, _state_key(), fn)



@st.cache_data(ttl=3600, max_entries=24, show_spinner="Ranking factors...")
def cached_rankings(key: tuple, off_track_only: bool) -> dict[str, pd.DataFrame]:
    """Factor rankings for the three targets under the current filters."""
    v = view()
    return insights.build_all(v.team_race, v.team_season, off_track_only)


@st.cache_data(ttl=3600, max_entries=24, show_spinner="Scoring anomalies...")
def cached_anomalies(key: tuple, cols: tuple[str, ...], contamination: float) -> pd.DataFrame:
    """Team x race rows scored by the three anomaly detectors."""
    return outliers.detect_anomalies(view().team_race, list(cols), by=["season"], contamination=contamination)


@st.cache_data(ttl=3600, max_entries=24, show_spinner="Computing embedding...")
def cached_embedding(key: tuple, cols: tuple[str, ...], method: str) -> tuple[pd.DataFrame, str]:
    """t-SNE or UMAP coordinates for the filtered team-seasons."""
    return reduction.embed(view().team_season, list(cols), method)
