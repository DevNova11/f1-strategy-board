"""Data sources: provenance of the five sources, coverage and join diagnostics."""

import pandas as pd
import plotly.express as px
import streamlit as st

from app import ui
from src import config, theme
from src.eda import EDAResult

v = ui.view()
bundle = ui.get_sources()
ui.header("Where the data comes from", "Two sources are real and downloaded from public services. Three are "
                                       "synthetic, because teams keep that information private.")

cards = st.columns(len(bundle.info))
for col, info in zip(cards, bundle.info):
    col.html(
        f'<div class="f1-card">{ui.badge(info.origin)}<h4>{info.name}</h4>'
        f'<div class="rows">{info.rows:,}</div><div class="meta">rows | {info.grain}</div>'
        f'<p><b>{info.provider}</b></p><p>{info.description}</p></div>'
    )

ui.section("How much real data there is", "Races per season, and how many have lap-by-lap timing.")
coverage = v.races.assign(has_laps=v.races["track_temp_c"].notna())
coverage = coverage.groupby(["season", "has_laps"]).size().rename("races").reset_index()
coverage["Lap and weather data"] = coverage["has_laps"].map({True: "Results + laps + weather", False: "Results and pit stops only"})
fig = px.bar(coverage, x="season", y="races", color="Lap and weather data",
             color_discrete_map={"Results + laps + weather": config.RED, "Results and pit stops only": "#4A4A55"},
             labels={"season": "Season", "races": "Races"})
fig.update_layout(barmode="stack", xaxis=dict(dtick=1))
with_laps = int(coverage.loc[coverage["has_laps"], "races"].sum())
total = int(coverage["races"].sum())
ui.show(EDAResult(theme.finish(fig, "Races covered per season", height=360),
                  f"{total} races have full results, pit stops and standings; {with_laps} of them "
                  f"({with_laps / max(total, 1):.0%}) also have lap-by-lap timing, tyre stints and weather. "
                  + ("Lap-level features are blank for the rest. Run python -m src.fetch --laps-per-season 0 "
                     "to download every race." if with_laps < total else "Coverage is complete.")), "ds_coverage")

ui.section("Do the sources line up?", "Share of rows matched when the sources are joined together.")
checks = pd.DataFrame(v.meta["join_checks"])
st.dataframe(
    checks, hide_index=True,
    column_config={
        "name": st.column_config.TextColumn("Alignment or join", width="large"),
        "left_rows": st.column_config.NumberColumn("Rows", format="%d"),
        "matched_rows": st.column_config.NumberColumn("Matched", format="%d"),
        "unmatched_rows": st.column_config.NumberColumn("Unmatched", format="%d"),
        "match_rate": st.column_config.ProgressColumn("Match rate", min_value=0, max_value=1, format="percent"),
    },
)
key_joins = checks[~checks["name"].str.contains("lap features|weather|pit stops")]
ui.takeaway(f"Every table is joined on season, round, constructor, driver and component. "
            f"The {len(key_joins)} key-alignment checks match "
            + ("every row" if key_joins["match_rate"].min() >= 1 else f"at least {key_joins['match_rate'].min():.0%} of rows")
            + ": team spellings from all sources resolve to one canonical id. Lower rates on lap, weather and pit-stop "
            f"joins reflect coverage (races without lap data, cars that never pitted), not key mismatches.")

ui.section("Peek at the raw data", "Exactly what each source looks like before cleaning.")
previews = {
    "Results (REAL)": bundle.track["results"], "Laps (REAL)": bundle.track["laps"],
    "Pit stops (REAL)": bundle.track["pitstops"], "Tyre stints (REAL)": bundle.track["stints"],
    "Weather (REAL)": bundle.weather, "Financial (SYNTHETIC)": bundle.financial,
    "Logistics (SYNTHETIC)": bundle.logistics, "Simulation / CFD (SYNTHETIC)": bundle.simulation,
}
choice = st.segmented_control("Source table", list(previews), default="Financial (SYNTHETIC)", key="ds_preview")
frame = previews[choice or "Financial (SYNTHETIC)"]
if "SYNTHETIC" in (choice or "SYNTHETIC"):
    ui.synthetic_banner("Synthetic table, deliberately messy so the cleaning step has real work to do: "
                        "team names spelled differently per source, mixed currencies and scales, kilograms next to "
                        "tonnes, miles next to kilometres, gaps and keying errors.")
st.dataframe(frame.head(300), hide_index=True, height=340)
st.caption(f"{len(frame):,} rows x {frame.shape[1]} columns. Showing the first 300.")
