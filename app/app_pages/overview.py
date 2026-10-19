"""Overview: the short answer, a few headline numbers and where to go next."""

import streamlit as st

from app import ui
from src import insights
from src.eda import sentence, timeseries

v = ui.view()
ui.header("Overview", "Which things Formula 1 teams do away from the track line up with winning races and "
                      "scoring points? Seven seasons of real results, 2019 to 2025.")
ui.require(v.team_race, "team-race rows")

latest = int(v.team_race["season"].max())
previous = latest - 1
rankings = ui.cached_rankings(v.key, False)
ranking = rankings["points"]

# ------------------------------------------------------------------ the answer
if not ranking.empty:
    real_on = ranking[(ranking["origin"] == "REAL") & (ranking["role"] == "on_track")]
    real_off = ranking[(ranking["origin"] == "REAL") & ranking["role"].isin(["operations", "off_track"])]
    synth = ranking[ranking["origin"] == "SYNTHETIC"]
    parts = []
    if not real_on.empty:
        top = real_on.iloc[0]
        parts.append(f"<b>Points follow pace.</b> The strongest real link is {sentence(top['label'])} "
                     f"(rank correlation {top['spearman']:+.2f}): the quicker a team is on Sunday, the more it scores.")
    if not real_off.empty:
        top = real_off.iloc[0]
        parts.append(f"Of the real factors a team controls away from the circuit, <b>{sentence(top['label'])}</b> "
                     f"matters most ({top['spearman']:+.2f}).")
    if not synth.empty:
        parts.append(f"Budget, logistics and simulation factors such as {sentence(synth.iloc[0]['label'])} also rank "
                     f"high, but that data is synthetic, so their place shows the method working rather than how "
                     f"real teams operate.")
    ui.answer(" ".join(parts))

# ---------------------------------------------------------------- key numbers
now, before = v.driver_race[v.driver_race["season"] == latest], v.driver_race[v.driver_race["season"] == previous]
pits = v.pitstops[v.pitstops["pit_valid"]]
pit_now = pits.loc[pits["season"] == latest, "duration_s"].median()
pit_before = pits.loc[pits["season"] == previous, "duration_s"].median()
season_pts = v.team_season[v.team_season["season"] == latest].sort_values("total_points", ascending=False)
has_prev = not before.empty
ui.kpi_row([
    {"label": "Races in view", "value": f"{v.races.shape[0]}",
     "delta": f"{v.races['season'].nunique()} seasons", "delta_color": "off"},
    {"label": f"Cars that retired in {latest}", "value": f"{now['dnf'].mean():.1%}",
     "delta": f"{(now['dnf'].mean() - before['dnf'].mean()) * 100:+.1f} pts on {previous}" if has_prev else None,
     "delta_color": "inverse", "help": "Share of race starts that did not finish."},
    {"label": f"Typical pit stop in {latest}", "value": f"{pit_now:.1f} s",
     "delta": f"{pit_now - pit_before:+.2f} s on {previous}" if has_prev else None, "delta_color": "inverse",
     "help": "Median time from pit-lane entry to exit."},
    {"label": f"Most points in {latest}", "value": season_pts["team"].iloc[0].split(" /")[0] if len(season_pts) else "-",
     "delta": f"{season_pts['total_points'].iloc[0]:.0f} points" if len(season_pts) else None, "delta_color": "off"},
])

# --------------------------------------------------------------- where to go
ui.section("Where to go next")
cards = [
    ("app_pages/eda_explorer.py", "Explore the data", ":material/query_stats:",
     "Distributions, relationships, outliers, trends and team groupings, one tab each."),
    ("app_pages/ranked_insights.py", "What drives results", ":material/leaderboard:",
     "Every factor ranked by how well it explains wins and points."),
    ("app_pages/reliability_risk.py", "Retirement risk", ":material/car_crash:",
     "Which cars were most likely to retire, and why the model thinks so."),
]
for col, (path, title, icon, text) in zip(st.columns(3), cards):
    with col.container(border=True):
        st.page_link(path, label=title, icon=icon)
        st.caption(text)

# --------------------------------------------------------------------- charts
left, right = st.columns([3, 2])
with left:
    ui.section(f"The {latest} title fight", "Points collected race by race.")
    ui.show(ui.memo("ov_progress", lambda: timeseries.season_progress(v.team_race, latest, "cum_points")), "ov_progress")
with right:
    ui.section("What explains points", "Longer bar = stronger link. Red = real data, grey = synthetic.")
    ui.show(ui.memo("ov_rank", lambda: insights.ranking_chart(ranking, "Points per race", top=10)), "ov_rank")

ui.synthetic_banner("Results, laps, pit stops, tyres, standings and weather are real (Jolpica and FastF1). "
                    "Budget, logistics and simulation data are generated, because no team publishes them.")

with st.expander("Terms used on this board"):
    st.markdown(
        """
- **Retirement (DNF)**: a car that started but did not finish the race.
- **Pit delta**: how much longer (or shorter) a pit stop took than the median stop in the same race.
- **Race pace vs field**: a team's typical clean lap compared with the field, in percent. Negative is faster.
- **Rank correlation (Spearman)**: from -1 to +1. Near +1 means the two things rise together; near -1 means one
  falls as the other rises; near 0 means no consistent link.
- **Partial correlation**: the same, after removing what last season's strength already explains.
- **Synthetic**: made-up data from a seeded generator, used where real data is not public.
"""
    )
