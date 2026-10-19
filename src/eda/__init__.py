"""Layer 4 - the EDA engine.

Six modules, one per EDA step:

* :mod:`src.eda.univariate`  - distributions, skew and kurtosis
* :mod:`src.eda.bivariate`   - correlations, pair plots, cross-domain scatters
* :mod:`src.eda.outliers`    - IQR, z-score, Isolation Forest, DNF analysis
* :mod:`src.eda.timeseries`  - season trends and tyre-degradation curves
* :mod:`src.eda.reduction`   - PCA, t-SNE / UMAP and k-means team profiles
* :mod:`src.eda.report`      - automated HTML report written to ``reports/``

Every chart-producing function returns an :class:`EDAResult`, so the dashboard,
the notebook and the report all render the same figure with the same
plain-English takeaway.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go

from src import theme  # noqa: F401  (registers the Plotly template on import)


@dataclass
class EDAResult:
    """A figure, the sentence that explains it, and optionally the numbers behind it."""

    fig: go.Figure
    takeaway: str
    table: pd.DataFrame | None = None


def describe_strength(r: float) -> str:
    """Translate a correlation coefficient into a plain-English strength."""
    size = abs(r)
    if size < 0.1:
        return "no meaningful"
    if size < 0.3:
        return "a weak"
    if size < 0.5:
        return "a moderate"
    if size < 0.7:
        return "a strong"
    return "a very strong"


# Human-readable labels for every analysis column, with its domain and origin.
# ``origin`` is REAL or SYNTHETIC; ``role`` marks on-track mediators and targets
# so the insight layer can separate off-track drivers from race outcomes.
FEATURES: dict[str, dict[str, str]] = {
    # --- financial (SYNTHETIC)
    "spend_total_musd": {"label": "Total spend per round (USD M)", "domain": "Financial", "origin": "SYNTHETIC", "role": "off_track"},
    "aero_share": {"label": "Aero share of spend", "domain": "Financial", "origin": "SYNTHETIC", "role": "off_track"},
    "power_unit_share": {"label": "Power-unit share of spend", "domain": "Financial", "origin": "SYNTHETIC", "role": "off_track"},
    # --- logistics (SYNTHETIC, travel distance real-derived)
    "travel_km": {"label": "Travel distance from HQ (km)", "domain": "Logistics", "origin": "REAL", "role": "off_track"},
    "freight_tonnes": {"label": "Freight shipped (t)", "domain": "Logistics", "origin": "SYNTHETIC", "role": "off_track"},
    "customs_delay_h": {"label": "Customs delay (h)", "domain": "Logistics", "origin": "SYNTHETIC", "role": "off_track"},
    "late_parts": {"label": "Late parts (count)", "domain": "Logistics", "origin": "SYNTHETIC", "role": "off_track"},
    "spares_coverage_pct": {"label": "Spares coverage (%)", "domain": "Logistics", "origin": "SYNTHETIC", "role": "off_track"},
    "crew_fatigue_idx": {"label": "Crew fatigue index", "domain": "Logistics", "origin": "SYNTHETIC", "role": "off_track"},
    "setup_hours": {"label": "Garage set-up time (h)", "domain": "Logistics", "origin": "SYNTHETIC", "role": "off_track"},
    # --- simulation / CFD (SYNTHETIC)
    "atr_allocation_pct": {"label": "Aero testing allocation (%)", "domain": "Simulation / CFD", "origin": "SYNTHETIC", "role": "off_track"},
    "wind_tunnel_runs": {"label": "Wind-tunnel runs", "domain": "Simulation / CFD", "origin": "SYNTHETIC", "role": "off_track"},
    "cfd_items": {"label": "CFD items", "domain": "Simulation / CFD", "origin": "SYNTHETIC", "role": "off_track"},
    "sim_hours": {"label": "Simulator hours", "domain": "Simulation / CFD", "origin": "SYNTHETIC", "role": "off_track"},
    "sim_track_corr_pct": {"label": "Sim-to-track correlation (%)", "domain": "Simulation / CFD", "origin": "SYNTHETIC", "role": "off_track"},
    "upgrade_count": {"label": "Upgrades brought", "domain": "Simulation / CFD", "origin": "SYNTHETIC", "role": "off_track"},
    "setup_iterations": {"label": "Set-up iterations", "domain": "Simulation / CFD", "origin": "SYNTHETIC", "role": "off_track"},
    "cfd_items_raw": {"label": "CFD items, as logged", "domain": "Simulation / CFD", "origin": "SYNTHETIC", "role": "raw"},
    # --- pit stops (REAL)
    "pit_median_s": {"label": "Median pit-lane time (s)", "domain": "Pit stops", "origin": "REAL", "role": "operations"},
    "pit_delta_s": {"label": "Pit time vs field (s)", "domain": "Pit stops", "origin": "REAL", "role": "operations"},
    "n_stops": {"label": "Pit stops made", "domain": "Pit stops", "origin": "REAL", "role": "operations"},
    # --- weather (REAL)
    "track_temp_c": {"label": "Track temperature (C)", "domain": "Weather", "origin": "REAL", "role": "context"},
    "air_temp_c": {"label": "Air temperature (C)", "domain": "Weather", "origin": "REAL", "role": "context"},
    "humidity_pct": {"label": "Humidity (%)", "domain": "Weather", "origin": "REAL", "role": "context"},
    "wind_speed_ms": {"label": "Wind speed (m/s)", "domain": "Weather", "origin": "REAL", "role": "context"},
    "rain_share": {"label": "Share of race with rain", "domain": "Weather", "origin": "REAL", "role": "context"},
    # --- telemetry / timing (REAL, on-track)
    "pace_delta_pct": {"label": "Race pace vs field (%)", "domain": "Telemetry", "origin": "REAL", "role": "on_track"},
    "lap_std_s": {"label": "Lap-time spread (s)", "domain": "Telemetry", "origin": "REAL", "role": "on_track"},
    "deg_slope_s_per_lap": {"label": "Tyre degradation (s/lap)", "domain": "Telemetry", "origin": "REAL", "role": "on_track"},
    "mean_grid": {"label": "Mean grid slot", "domain": "Telemetry", "origin": "REAL", "role": "on_track"},
    # --- outcomes (REAL)
    "points": {"label": "Points scored", "domain": "Outcome", "origin": "REAL", "role": "target"},
    "win": {"label": "Race win", "domain": "Outcome", "origin": "REAL", "role": "target"},
    "dnf": {"label": "Retirements", "domain": "Outcome", "origin": "REAL", "role": "target"},
    "mech_dnf": {"label": "Mechanical retirements", "domain": "Outcome", "origin": "REAL", "role": "target"},
}


def label(col: str) -> str:
    """Readable label for a column (falls back to the column name)."""
    return FEATURES.get(col, {}).get("label", col.replace("_", " ").capitalize())


def short(col: str) -> str:
    """Label without the unit, for chart titles that must stay on one line."""
    return label(col).split(" (")[0]


def sentence(text: str) -> str:
    """Prepare a label for use mid-sentence: drop the unit, lower the first
    letter, but leave acronyms such as CFD alone."""
    text = str(text).split(" (")[0].replace(" *", "")
    first = text.split(" ")[0] if text else ""
    return text if first.isupper() and len(first) > 1 else text[:1].lower() + text[1:]


def plain(col: str) -> str:
    """Label of a column, ready for use inside a sentence."""
    return sentence(label(col))


def origin(col: str) -> str:
    """``REAL`` or ``SYNTHETIC`` for a column (``REAL`` when unknown)."""
    return FEATURES.get(col, {}).get("origin", "REAL")


def tagged(col: str) -> str:
    """Label with a visible synthetic marker where it applies.

    ``Total spend per round (USD M)`` becomes ``Total spend per round (USD M, synthetic)``.
    """
    text = label(col)
    if origin(col) != "SYNTHETIC":
        return text
    return text[:-1] + ", synthetic)" if text.endswith(")") else text + " (synthetic)"


def factor_columns(frame: pd.DataFrame, roles: tuple[str, ...] = ("off_track", "operations", "context", "on_track")) -> list[str]:
    """Columns of ``frame`` registered as explanatory factors with one of ``roles``."""
    return [c for c, meta in FEATURES.items() if meta["role"] in roles and c in frame.columns]
