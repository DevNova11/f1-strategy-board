"""Methodology: architecture, data provenance, methods and limitations."""

import pandas as pd
import streamlit as st

from app import ui
from src import config
from src.eda import FEATURES, label
from src.synthetic import COST_CAP_MUSD, FX_TO_USD, PLANTED_EFFECTS

v = ui.view()
ui.header("How it works", "How the board is built, what is real and what is not, and how far the results can be trusted.")

ui.section("Seven steps from raw data to this page", "Each step is one module in src/.")
layers = [
    ("01", "Sources", "sources.py", "Five loaders: track timing and weather (real); finance, logistics, simulation (synthetic)"),
    ("02", "Ingest", "ingest.py", "Align schemas, sync weather to laps on session time, join on season, round, team, driver, component"),
    ("03", "Clean", "clean.py", "Fill and flag gaps, harmonise units, flag outliers, add feature keys"),
    ("04", "EDA engine", "eda/", "Univariate, bivariate, outliers, time series, clustering, automated report"),
    ("05", "Model", "model.py", "Correlation engine, regression, retirement-risk scoring with SHAP"),
    ("06", "Insights", "insights.py", "Factors ranked for wins and points by four methods"),
    ("07", "Dashboard", "app/", "Eight pages, sidebar filters, anomaly watchlist"),
]
st.html('<div class="f1-steps">' + "".join(
    f'<div class="f1-step"><i>{n}</i><b>{name}</b><span><code>{module}</code><br>{text}</span></div>'
    for n, name, module, text in layers) + "</div>")
ui.takeaway("Data flows from layer 1 to layer 7. One arrow runs backwards, the feedback loop: rows an analyst flags on "
            "the EDA explorer's Outliers tab are written to a watchlist that the EDA layer (04) reads on its next pass. "
            "The README has the same architecture as a diagram.")

ui.section("Real and synthetic data")
left, right = st.columns(2)
with left.container(border=True):
    st.html(ui.badge("REAL"))
    st.markdown(
        f"""
**Race and sprint results, pit stops, standings, calendar** come from the Jolpica API, the maintained successor to
Ergast. **Lap times, tyre stints and weather** come from FastF1, which reads the sport's live-timing feed.

Loaded now: **{v.meta['races']} races** across {min(v.meta['seasons'])}-{max(v.meta['seasons'])},
**{v.meta['races_with_laps']}** of them with lap-level data, in **{'full download' if v.meta['mode'] == 'raw' else 'bundled sample'}** mode.

Everything is cached in `data/raw/`. With no cache and no network, the app falls back to `data/sample/`.
"""
    )
with right.container(border=True):
    st.html(ui.badge("SYNTHETIC"))
    st.markdown(
        f"""
**Financial (cost cap), logistics and simulation / CFD** data are not public. `src/synthetic.py` fabricates them
with a seeded generator (seed {config.SEED}), conditioned on the real results so the numbers are plausible.

Two public rules are respected: the cost-cap level per season and the sliding aero-testing allocation by previous
championship position. Travel distance is computed from real circuit and factory coordinates.

Every synthetic row carries `data_origin = "SYNTHETIC"` and every synthetic factor is starred or badged in the UI.
"""
    )

ui.section("What was built into the synthetic data", "The known answers the ranking is tested against.")
planted = pd.DataFrame(PLANTED_EFFECTS)
planted["factor"] = planted["factor"].map(lambda c: label(c) if c in FEATURES else c.capitalize())
planted["target"] = planted["target"].map({"points": "Points scored", "dnf": "Retirements", "grid": "Grid slot",
                                           "pit_delta_s": "Pit time vs field"}).fillna(planted["target"].str.capitalize())
st.dataframe(planted, hide_index=True,
             column_config={"factor": "Synthetic factor", "target": "Linked to (real)", "direction": "Direction",
                            "mechanism": st.column_config.TextColumn("Mechanism in the generator", width="large")})
caps = ", ".join(f"{season}: {'no cap' if cap is None else f'USD {cap:.0f}M'}" for season, cap in COST_CAP_MUSD.items())
fx = ", ".join(f"1 {cur} = {rate:.2f} USD" for cur, rate in FX_TO_USD.items() if cur != "USD")
st.caption(f"Cost cap used by the generator: {caps}. Fixed conversion rates: {fx}.")

ui.section("Methods")
st.markdown(
    """
| Step | What is done | Why |
|---|---|---|
| Ingest | Team names normalised to one id; laps joined to results on season, round and driver code; weather attached to each lap with a backward as-of join on session time | Five sources, five naming schemes and two clocks |
| Clean | Units harmonised, gaps filled with group medians and flagged, outliers flagged not deleted | Keeps the raw signal available to the EDA |
| Clean laps | Green flag, accurate timing, not an in/out lap, within 107% of the fastest lap | Race pace, not traffic or safety cars |
| Tyre degradation | Slope of fuel-corrected lap time on tyre age per stint | Separates wear from the car getting lighter |
| Outliers | IQR fences, z-score and Isolation Forest, combined by vote | Each has a different blind spot |
| Correlation | Pearson, Spearman and rank-based partial correlation | Skewed data needs rank methods; partial correlation removes the prior-strength confounder |
| Regression | Standardised OLS (points) and logit (wins), team-clustered errors, VIF reported | Effect of each factor with the others held fixed |
| Reliability | Gradient boosting on pre-race features, trained on earlier seasons, tested on later ones, explained with SHAP | An honest, time-ordered test |
| Ranking | Mean rank across Spearman, partial correlation, regression and boosting importance | No single statistic deserves to be called importance |
| Clustering | PCA, t-SNE or UMAP, k-means with k chosen by silhouette score | Finds operating profiles without labels |
"""
)

ui.section("Limitations")
st.markdown(
    """
1. **Three of five sources are synthetic.** Rankings that involve budget, logistics or simulation factors reflect
   what the generator planted. They show the platform works; they say nothing about real teams.
2. **Association, not causation.** Points fund budgets as much as budgets buy points. Partial correlation and
   regression reduce confounding but cannot remove it.
3. **Retirement causes disappear from 2023.** The public feed logs most recent retirements as plain "Retired", so
   component-level failure analysis covers 2019 to 2022 and the reliability target is any retirement.
4. **Weather cannot explain between-team gaps.** Every team shares the same weather in a race, so it ranks low for
   points and wins by construction. Its real role is on field-wide pace, tyre wear and retirement risk.
5. **Small samples at season level.** There are only ten teams a year, so team-season results rest on about 70 rows.
6. **Approximations.** Fuel correction uses a fixed 0.055 s per lap; pit times are pit-lane times, not stationary
   times; travel distance is factory to circuit, not circuit to circuit.
7. **Retirements are mostly chance.** Expect the risk model to rank a little better than random, not to predict
   individual failures.
"""
)

ui.section("Reproducibility")
st.markdown(
    f"""
- Random seed **{config.SEED}** for the synthetic generator, Isolation Forest, k-means, t-SNE and the models.
- All paths are relative to the project root; nothing is hard-coded.
- `python -m src.fetch` refreshes the real data, `python -m src.pipeline` rebuilds the tables, `pytest` runs the tests.
"""
)
