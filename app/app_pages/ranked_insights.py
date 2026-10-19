"""What drives results: every factor ranked by how well it explains wins and points."""

import streamlit as st

from app import ui
from src import insights

v = ui.view()
ui.header("What drives results", "Every factor ranked by how closely it lines up with winning races or scoring "
                                 "points. A factor ranks high only if four different methods agree it matters.")
ui.require(v.team_race, "team-race rows")

c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
labels = {meta["label"]: target for target, meta in insights.TARGETS.items()}
choice = c1.segmented_control("Explain", list(labels), default="Points per race", key="ri_target")
off_track = c2.toggle("Only things decided away from the track", key="ri_off",
                      help="Hides race pace, grid position, tyre wear and lap consistency. Those are results in "
                           "their own right, so they crowd out everything else when included.")
target = labels[choice or "Points per race"]
name = insights.TARGETS[target]["label"]
ranking = ui.cached_rankings(v.key, bool(off_track))[target]
if ranking.empty:
    st.warning("Not enough data to rank factors for this under the current filters. Widen the season range or "
               "add more teams.", icon=":material/filter_alt_off:")
    st.stop()

real_off = ranking[(ranking["origin"] == "REAL") & ranking["role"].isin(["operations", "off_track"])]
best = ranking.iloc[0]
best_real = real_off.iloc[0] if not real_off.empty else ranking.iloc[0]
ui.kpi_row([
    {"label": "Ranked first", "value": best["label"].split(" (")[0],
     "delta": f"{best['domain']}, {'real' if best['origin'] == 'REAL' else 'synthetic'} data", "delta_color": "off"},
    {"label": "Best real factor decided away from the track", "value": best_real["label"].split(" (")[0],
     "delta": f"rank {best_real['consensus_rank']} of {len(ranking)}", "delta_color": "off"},
    {"label": "Factors compared", "value": f"{len(ranking)}",
     "delta": f"{int((ranking['spearman_p'] < 0.05).sum())} with a clear link", "delta_color": "off",
     "help": "A clear link means the rank correlation is statistically significant at the 5% level."},
])

left, right = st.columns([3, 2])
with left:
    ui.show(ui.memo("ri_rank", lambda: insights.ranking_chart(ranking, name, top=12)), "ri_rank")
with right:
    ui.section("In plain words")
    for statement in insights.insight_statements(ranking, name, 4):
        st.markdown(statement.replace(" SYNTHETIC", " synthetic").replace(", SYNTHETIC)", ", synthetic)")
                    .replace(", REAL)", ", real)"))
ui.synthetic_banner()

lap_share = v.team_race["pace_delta_pct"].notna().mean()
if lap_share < 0.999 and not off_track:
    st.caption(f"Lap-based factors (race pace, lap spread, tyre wear) are only available for the {lap_share:.0%} of "
               f"races whose lap data has been downloaded, which slightly understates them.")

ui.section("Which kind of factor matters most", "Each source of data, judged by its best factor.")
ui.show(ui.memo("ri_domain", lambda: insights.domain_chart(ranking, name)), "ri_domain", table=True)

with st.expander("How the ranking is made, and whether the methods agree"):
    st.markdown(
        "Four methods each rank every factor: **rank correlation** (does it rise or fall with the outcome?), "
        "**partial correlation** (does that survive once last season's strength is allowed for?), "
        "**regression** (does it matter with all other factors held fixed?) and **boosting importance** (does a "
        "prediction model get worse without it?). The final rank is the average of the four.")
    ui.show(ui.memo("ri_methods", lambda: insights.method_comparison(ranking, top=10)), "ri_methods")

with st.expander("Full ranking table"):
    st.dataframe(
        ranking[["consensus_rank", "label", "domain", "origin", "direction", "spearman", "partial", "regression",
                 "boosting", "score", "n"]], hide_index=True,
        column_config={
            "consensus_rank": st.column_config.NumberColumn("Rank", format="%d"),
            "label": st.column_config.TextColumn("Factor", width="medium"), "domain": "Source", "origin": "Data",
            "direction": "Direction",
            "spearman": st.column_config.NumberColumn("Rank corr.", format="%.2f"),
            "partial": st.column_config.NumberColumn("Partial", format="%.2f",
                                                     help="After allowing for last season's points share."),
            "regression": st.column_config.NumberColumn("Regression", format="%.2f"),
            "boosting": st.column_config.NumberColumn("Boosting", format="%.3f"),
            "score": st.column_config.ProgressColumn("Overall", min_value=0, max_value=1, format="%.2f"),
            "n": st.column_config.NumberColumn("Rows", format="%d"),
        },
    )
    st.download_button("Download as CSV", ranking.to_csv(index=False), f"ranked_factors_{target}.csv", "text/csv",
                       key="ri_dl", icon=":material/download:")

with st.expander("Check: does the ranking find what was built into the synthetic data?"):
    recovery = insights.recovery_check(v.team_race)
    st.dataframe(
        recovery, hide_index=True,
        column_config={
            "factor": "Synthetic factor", "target": "Real outcome", "planted": "Built-in direction",
            "observed_r": st.column_config.NumberColumn("Found", format="%.2f"),
            "p": st.column_config.NumberColumn("p", format="%.2g"),
            "recovered": st.column_config.CheckboxColumn("Found it"),
            "mechanism": st.column_config.TextColumn("How it was built in", width="large"),
        },
    )
    found = int(recovery["recovered"].sum())
    missed = recovery[~recovery["recovered"]]
    text = (f"{found} of {len(recovery)} built-in links are found with the right direction. Because we know exactly "
            f"what the synthetic data contains, this is a direct test that the ranking method works.")
    if not missed.empty:
        text += (f" Missed: {', '.join(missed['factor'].str.split(' (', regex=False).str[0].str.lower())}, which was "
                 f"built in as a change in points around an upgrade rather than a level, so a correlation of levels "
                 f"is the wrong tool to see it.")
    ui.takeaway(text)
