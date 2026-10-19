"""Cleaning report: what the clean layer changed, and what it only flagged."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app import ui
from src import config, theme
from src.eda import EDAResult

v = ui.view()
meta = v.meta
steps = pd.DataFrame(meta["cleaning_steps"])
before = pd.DataFrame(meta["missing_before"])
after = pd.DataFrame(meta["missing_after"])

ui.header("How the data was cleaned", "Every fix applied before analysis, in order. Suspicious values are "
                                     "marked rather than deleted, and filled-in gaps stay labelled.")

rows_in = sum(meta["rows_before"].values())
imputed = int(steps.loc[steps["step"] == "impute missing", "rows_affected"].sum())
flagged = int(steps.loc[steps["step"].str.contains("flag"), "rows_affected"].sum())
harmonised = int(steps.loc[steps["step"].str.contains("harmonise"), "rows_affected"].sum())
dropped = int(steps.loc[steps["step"].isin(["deduplicate", "drop untimed laps"]), "rows_affected"].sum())
ui.kpi_row([
    {"label": "Rows checked", "value": f"{rows_in:,}", "help": "Rows across all aligned source tables."},
    {"label": "Rows dropped", "value": f"{dropped:,}", "delta": f"{dropped / rows_in:.2%} of input", "delta_color": "off",
     "help": "Only exact duplicates and laps with no recorded time are removed."},
    {"label": "Gaps filled", "value": f"{imputed:,}", "delta": "with typical values", "delta_color": "off"},
    {"label": "Units converted", "value": f"{harmonised:,}", "delta": "currency, scale, km, tonnes", "delta_color": "off"},
    {"label": "Rows marked suspicious", "value": f"{flagged:,}", "delta": "kept, not deleted", "delta_color": "off"},
])

ui.section("Missing values", "Share of empty cells per column, before and after cleaning.")
merged = before.merge(after, on=["table", "column"], how="left", suffixes=("_before", "_after"))
merged["missing_pct_after"] = merged["missing_pct_after"].fillna(0.0)
merged["field"] = merged["table"] + "." + merged["column"]
focus = merged[merged["table"].isin(["financial", "logistics", "simulation", "pitstops", "laps"])]
focus = focus[focus["missing_pct_before"] < 60].sort_values("missing_pct_before", ascending=True).tail(18)
fig = go.Figure()
fig.add_trace(go.Bar(y=focus["field"], x=focus["missing_pct_before"], orientation="h", name="Before cleaning",
                     marker_color=config.RED, hovertemplate="%{y}<br>%{x:.2f}% missing<extra>Before</extra>"))
fig.add_trace(go.Bar(y=focus["field"], x=focus["missing_pct_after"], orientation="h", name="After cleaning",
                     marker_color=config.CYAN, hovertemplate="%{y}<br>%{x:.2f}% missing<extra>After</extra>"))
fig.update_layout(barmode="group", xaxis_title="Cells missing (%)", yaxis_title="")
fixed = int(((focus["missing_pct_before"] > 0) & (focus["missing_pct_after"] == 0)).sum())
worst = focus.iloc[-1]
ui.show(EDAResult(
    theme.finish(fig, "Missing values before and after", height=max(380, 26 * len(focus) + 150)),
    f"{fixed} of the {len(focus)} columns shown are fully repaired. The largest gap was {worst['field']} at "
    f"{worst['missing_pct_before']:.1f}% missing. Synthetic operations columns are filled with the median of the same "
    f"team and season; real timing gaps (sector times on lap 1, laps without a tyre compound) are left as they are "
    f"because inventing a lap time would be worse than admitting it is unknown."), "cl_missing")

left, right = st.columns(2)
with left:
    ui.section("Units made consistent")
    units = steps[steps["step"].str.contains("harmonise")].copy()
    units["Conversion"] = units["detail"]
    fig = px.bar(units, x="rows_affected", y="Conversion", orientation="h", text="rows_affected",
                 color_discrete_sequence=[config.CYAN], labels={"rows_affected": "Rows converted"})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis=dict(autorange="reversed", title=""))
    ui.show(EDAResult(theme.finish(fig, "Rows converted to a common unit", height=340, legend=False),
                      f"{harmonised:,} values arrived in a different unit from the project standard (USD millions, "
                      f"kilometres, tonnes). Summing spend before this step would have mixed pounds, euros and "
                      f"dollars, and thousands with millions."), "cl_units")
with right:
    ui.section("Values marked suspicious")
    flags = steps[steps["step"].str.contains("flag")].copy()
    flags["Rule"] = flags["table"] + ": " + flags["detail"].str.slice(0, 46)
    flags["share_pct"] = flags["share"] * 100
    fig = px.bar(flags, x="share_pct", y="Rule", orientation="h", text="rows_affected",
                 color_discrete_sequence=[config.RED], labels={"share_pct": "Rows flagged (%)"})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis=dict(autorange="reversed", title=""))
    top = flags.sort_values("share", ascending=False).iloc[0]
    ui.show(EDAResult(theme.finish(fig, "Share of rows flagged by each rule", height=340, legend=False),
                      f"The busiest rule is on {top['table']} ({top['share']:.0%} of rows): {top['detail']}. "
                      f"Flags mark rows for the EDA to treat with care; the values themselves are kept."), "cl_flags")

ui.section("Every step", "In the order it ran.")
st.dataframe(
    steps, hide_index=True,
    column_config={
        "table": "Table", "step": "Step", "detail": st.column_config.TextColumn("What was done", width="large"),
        "rows_affected": st.column_config.NumberColumn("Rows affected", format="%d"),
        "rows_total": st.column_config.NumberColumn("Rows in table", format="%d"),
        "share": st.column_config.ProgressColumn("Share", min_value=0, max_value=1, format="percent"),
    },
)

ui.section("The rules used")
st.markdown(
    """
- **Results**: status text is classified into finished, mechanical, incident, unspecified or other. A grid slot of 0
  (pit-lane start) is recoded to the back of the field.
- **Laps**: a lap is *clean* when timing is accurate, the track is green, it is not an in or out lap and it is within
  107% of the race's fastest lap. Lap times are fuel-corrected at 0.055 s per lap of fuel.
- **Pit stops**: pit-lane times outside 12 to 60 seconds are flagged as red-flag waits, repairs or penalties.
- **Financial (synthetic)**: duplicates removed, thousands converted to millions, EUR and GBP converted to USD at fixed
  rates, gaps filled with the team-season-component median, values with a robust z-score above 6 flagged.
- **Logistics and simulation (synthetic)**: kilograms to tonnes, miles to kilometres, gaps filled with the team-season
  median, extreme customs delays and CFD counts flagged.
"""
)
