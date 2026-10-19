"""Correlations: the cross-domain correlation engine."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app import ui
from src import config, insights, model, theme
from src.eda import FEATURES, EDAResult, factor_columns, label, plain, tagged, sentence

v = ui.view()
ui.header("What moves together", "Pick an outcome to see which factors rise or fall with it, and which links "
                                 "survive once you allow for how strong each team already was.")
ui.require(v.team_race, "team-race rows")
ui.synthetic_banner()

frame = insights.add_prior_strength(v.team_race, v.team_season)
frame = frame.assign(win=frame["win"].astype(float))
factors = factor_columns(frame)
targets = {"Points scored": "points", "Race win": "win", "Retirements": "dnf",
           "Pit time vs field": "pit_delta_s", "Mean grid slot": "mean_grid"}
controls = {"Last season's points share": "prior_points_share", "Mean grid slot": "mean_grid",
            "Total spend (synthetic)": "spend_total_musd"}

c1, c2 = st.columns([2, 3])
target_name = c1.selectbox("Outcome", list(targets), key="co_target")
control_names = c2.pills("Control for (partial correlation)", list(controls), selection_mode="multi",
                         default=["Last season's points share"], key="co_controls")
target = targets[target_name]
covars = [controls[c] for c in (control_names or []) if controls[c] != target]
usable = [f for f in factors if f != target and f not in covars]
table = ui.memo("co_table", lambda: model.cross_domain_table(frame, usable, target, covars or None))

ui.section("Links with the outcome", "Left: the raw link. Right: what is left after the controls you pick.")
left, right = st.columns(2)
with left:
    ui.show(ui.memo("co_raw", lambda: model.cross_domain_chart(table, target)), "co_raw")
with right:
    if covars:
        ui.show(ui.memo("co_partial", lambda: model.cross_domain_chart(table, target, controlled=True)), "co_partial")
    else:
        st.info("Pick at least one control variable to see partial correlations.", icon=":material/info:")

if covars and not table.empty:
    shrink = table.dropna(subset=["spearman_r", "partial_r"]).copy()
    shrink["lost"] = shrink["spearman_r"].abs() - shrink["partial_r"].abs()
    big = shrink[shrink["spearman_r"].abs() > 0.2].sort_values("lost", ascending=False)
    if not big.empty:
        row = big.iloc[0]
        ui.takeaway(f"Controlling for {', '.join(n.lower() for n in control_names)} matters most for "
                    f"{sentence(row['label'])}: its correlation with {plain(target)} goes from "
                    f"{row['spearman_r']:+.2f} to {row['partial_r']:+.2f}. A link that collapses like this was mostly "
                    f"standing in for how strong the team already was.")

st.dataframe(
    table.drop(columns=["factor"]), hide_index=True,
    column_config={
        "label": st.column_config.TextColumn("Factor", width="medium"), "domain": "Domain", "origin": "Origin",
        "pearson_r": st.column_config.NumberColumn("Pearson r", format="%.2f"),
        "spearman_r": st.column_config.NumberColumn("Spearman r", format="%.2f"),
        "spearman_p": st.column_config.NumberColumn("p", format="%.2g"),
        "partial_r": st.column_config.NumberColumn("Partial r", format="%.2f"),
        "partial_p": st.column_config.NumberColumn("Partial p", format="%.2g"),
        "n": st.column_config.NumberColumn("n", format="%d"),
    },
)

ui.section("Every factor against every outcome", "Rank correlation, one row per factor.")
matrix = ui.memo("co_matrix_data", lambda: pd.DataFrame(
    {name: [model.correlation(frame, f, col, "spearman")["r"] if f != col else np.nan for f in factors]
     for name, col in targets.items()},
    index=[label(f) + (" *" if FEATURES[f]["origin"] == "SYNTHETIC" else "") for f in factors]))
fig = px.imshow(matrix.values, x=list(matrix.columns), y=list(matrix.index), zmin=-1, zmax=1,
                color_continuous_scale=config.DIVERGING, aspect="auto", text_auto=".2f",
                labels=dict(color="Spearman r"))
fig.update_xaxes(side="top", showgrid=False)
fig.update_yaxes(showgrid=False)
stacked = matrix.abs().stack()
(best_factor, best_target) = stacked.idxmax()
by_target = matrix.abs().mean().sort_values(ascending=False)
ui.show(EDAResult(
    theme.finish(fig, "Cross-domain correlation matrix", height=max(480, 26 * len(matrix) + 170), legend=False),
    f"Rows are factors (* = synthetic), columns are outcomes. The single strongest cell is "
    f"{sentence(best_factor)} against {best_target.lower()} (r = {matrix.loc[best_factor, best_target]:+.2f}). "
    f"{by_target.index[0]} is the outcome the factors explain best on average and {by_target.index[-1].lower()} the one "
    f"they explain least."), "co_matrix")

ui.section("All factors at once", "A regression that holds the other factors fixed.")
kind = "Logit" if target == "win" else "OLS"
reg_factors = [f for f in usable if FEATURES[f]["role"] != "context"]
st.caption(f"{kind} of {target_name.lower()} on {len(reg_factors)} factors. Weather is left out: it is identical for "
           f"every team in a race, so it cannot separate them. Standard errors are clustered by team.")
try:
    ui.show(ui.memo("co_reg", lambda: model.coefficient_chart(
        model.fit_logit(frame, target, reg_factors) if kind == "Logit"
        else model.fit_ols(frame, target, reg_factors, cluster="lineage"), target)), "co_reg", table=True)
except Exception as exc:  # too few complete rows under tight filters
    st.info(f"Not enough complete rows to fit the regression under the current filters ({exc}).",
            icon=":material/info:")

ui.section("Check any two measures", "With and without the controls above.")
c1, c2 = st.columns(2)
everything = factors + list(targets.values())
x = c1.selectbox("First variable", everything, format_func=tagged, key="co_x",
                 index=everything.index("late_parts") if "late_parts" in everything else 0)
y = c2.selectbox("Second variable", everything, format_func=tagged, key="co_y", index=everything.index("dnf"))
raw_s = model.correlation(frame, x, y, "spearman")
raw_p = model.correlation(frame, x, y, "pearson")
part = model.partial_correlation(frame, x, y, covars) if covars else {"r": float("nan"), "p": float("nan"), "n": 0}
ui.kpi_row([
    {"label": "Spearman r", "value": f"{raw_s['r']:+.2f}", "delta": f"p = {raw_s['p']:.2g}", "delta_color": "off"},
    {"label": "Pearson r", "value": f"{raw_p['r']:+.2f}", "delta": f"p = {raw_p['p']:.2g}", "delta_color": "off"},
    {"label": "Partial r", "value": f"{part['r']:+.2f}" if np.isfinite(part["r"]) else "n/a",
     "delta": f"p = {part['p']:.2g}" if np.isfinite(part["p"]) else "no controls", "delta_color": "off"},
    {"label": "Rows used", "value": f"{raw_s['n']:,}"},
])
