"""EDA step 4 - time series and trends.

Two time axes matter in this project: the *season* (round by round, and season
by season) for spend and performance, and the *stint* (lap by lap of tyre age)
for degradation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from src import config, theme
from src.eda import EDAResult, describe_strength, label, tagged, plain, short
from src.eda.univariate import COMPOUND_COLOURS, COMPOUND_ORDER

TEAM_COLOUR_MAP = {theme.team_label(k): v for k, v in config.TEAM_COLOR.items()}
SHORT_NAMES = {theme.team_label(k): theme.team_short(theme.team_label(k)) for k in config.TEAM_COLOR}


def season_progress(team_race: pd.DataFrame, season: int, metric: str = "cum_points") -> EDAResult:
    """Round-by-round line per team for one season."""
    data = team_race[team_race["season"] == season].sort_values("round")
    names = {"cum_points": "Cumulative points", "cum_spend_musd": "Cumulative spend (USD M, synthetic)",
             "points_roll3": "Points, 3-race rolling mean"}
    fig = px.line(data, x="round", y=metric, color="team", color_discrete_map=TEAM_COLOUR_MAP, markers=True,
                  hover_data={"race_name": True},
                  labels={"round": "Round", metric: names.get(metric, tagged(metric)), "team": "Team",
                          "race_name": "Race"})
    fig.update_traces(marker=dict(size=5))
    fig.update_xaxes(dtick=2)
    if data.empty:
        return EDAResult(theme.finish(fig, f"{season} season progress"), "No data for this season.")
    theme.label_line_ends(fig, data, "round", metric, "team", names=SHORT_NAMES, colours=TEAM_COLOUR_MAP)
    final = data.groupby("team")[metric].last().sort_values(ascending=False)
    gap = final.iloc[0] - final.iloc[1] if len(final) > 1 else 0
    text = (f"In {season}, {final.index[0]} finishes on top at {final.iloc[0]:,.0f}, {gap:,.0f} clear of "
            f"{final.index[1] if len(final) > 1 else 'the field'}. Lines that steepen mid-season mark a team "
            f"finding form; flat stretches mark a slump.")
    return EDAResult(theme.finish(fig, f"{season}: {names.get(metric, label(metric)).split(' [')[0]}", height=480), text)


def spend_vs_performance(team_race: pd.DataFrame, season: int, teams: list[str] | None = None) -> EDAResult:
    """Season-long spend against points, both as a share of the team's season total.

    Indexing both measures to 0-100% puts them on one honest axis: a team that
    spends evenly and scores evenly follows the diagonal, and the gap between
    the two lines shows whether money is going out ahead of results.
    """
    data = team_race[team_race["season"] == season].sort_values("round").copy()
    if teams:
        data = data[data["team"].isin(teams)]
    grp = data.groupby("team")
    data["Spend (synthetic)"] = grp["spend_total_musd"].cumsum() / grp["spend_total_musd"].transform("sum") * 100
    data["Points"] = grp["points"].cumsum() / grp["points"].transform("sum").replace(0, np.nan) * 100
    long = data.melt(id_vars=["round", "team", "race_name"], value_vars=["Spend (synthetic)", "Points"],
                     var_name="Measure", value_name="share")
    fig = px.line(long, x="round", y="share", color="Measure", facet_col="team", facet_col_wrap=5,
                  color_discrete_map={"Spend (synthetic)": config.RED, "Points": config.CYAN},
                  labels={"round": "Round", "share": "% of season total"}, facet_row_spacing=0.14)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].split(" /")[0],
                                               font=dict(size=12, color=config.MUTED)))
    fig.update_yaxes(range=[0, 105])
    lead = (data["Spend (synthetic)"] - data["Points"]).groupby(data["team"]).mean().dropna()
    n_teams = data["team"].nunique()
    if lead.empty:
        return EDAResult(theme.finish(fig, f"{season}: spend vs points through the season"), "No data.")
    text = (f"On average spending runs {lead.mean():+.0f} percentage points ahead of points through the season "
            f"(positive means money goes out before results come in); the widest gap is "
            f"{lead.idxmax().split(' /')[0]} ({lead.max():+.0f}). Both lines are shares of the team's own season "
            f"total. Spend is synthetic.")
    return EDAResult(theme.finish(fig, f"{season}: spend vs points through the season",
                                  height=300 * (1 + (n_teams - 1) // 5) + 120), text)


def multi_season_trend(team_season: pd.DataFrame, metric: str = "total_points") -> EDAResult:
    """Season-by-season trend per team lineage, so rebrands stay on one line."""
    names = {"total_points": "Points (race + sprint)", "spend_total_musd": "Season spend (USD M, synthetic)",
             "points_per_musd": "Points per USD M (synthetic spend)", "dnf_rate": "Retirement rate",
             "pit_median_s": "Median pit-lane time (s)", "championship_position": "Championship position"}
    data = team_season.sort_values("season")
    fig = px.line(data, x="season", y=metric, color="team", color_discrete_map=TEAM_COLOUR_MAP, markers=True,
                  hover_data={"constructor_name": True},
                  labels={"season": "Season", metric: names.get(metric, tagged(metric)), "team": "Team",
                          "constructor_name": "Entered as"})
    fig.update_xaxes(dtick=1)
    if metric == "championship_position":
        fig.update_yaxes(autorange="reversed", dtick=1)
    if metric == "dnf_rate":
        fig.update_yaxes(tickformat=".0%")
    if metric == "spend_total_musd" and data["season"].min() <= 2020 < data["season"].max():
        fig.add_vline(x=2020.5, line_color=config.MUTED, line_dash="dot")
        fig.add_annotation(x=2020.55, y=1, yref="paper", text="Cost cap begins", showarrow=False, xanchor="left",
                           font=dict(color=config.MUTED, size=12))
    if not data.empty:
        theme.label_line_ends(fig, data, "season", metric, "team", names=SHORT_NAMES, colours=TEAM_COLOUR_MAP)
    change = data.groupby("team")[metric].agg(["first", "last"])
    change["delta"] = change["last"] - change["first"]
    if change.empty:
        return EDAResult(theme.finish(fig, f"{names.get(metric, label(metric))} by season"), "No data.")
    up, down = change["delta"].idxmax(), change["delta"].idxmin()
    first, last = int(data["season"].min()), int(data["season"].max())
    def fmt(value: float) -> str:
        return f"{value:+,.0f}" if abs(value) >= 10 else f"{value:+,.2f}"

    what = names.get(metric, label(metric)).split(" [")[0].split(" (")[0].lower()
    if change["delta"].max() <= 0:
        text = (f"From {first} to {last} {what} fell for every team: most for {down} "
                f"({fmt(change.loc[down, 'delta'])}) and least for {up} ({fmt(change.loc[up, 'delta'])}).")
    elif change["delta"].min() >= 0:
        text = (f"From {first} to {last} {what} rose for every team: most for {up} "
                f"({fmt(change.loc[up, 'delta'])}) and least for {down} ({fmt(change.loc[down, 'delta'])}).")
    else:
        text = (f"From {first} to {last} the biggest rise in {what} is {up} ({fmt(change.loc[up, 'delta'])}) "
                f"and the biggest fall is {down} ({fmt(change.loc[down, 'delta'])}).")
    if metric == "spend_total_musd":
        text += " SYNTHETIC: the generator applies the published cap, so the spread between teams collapses from 2021."
    return EDAResult(theme.finish(fig, f"{names.get(metric, label(metric)).split(' [')[0]} by season", height=480), text)


def spend_points_lag(team_season: pd.DataFrame) -> EDAResult:
    """Does this season's spend relate more to this year's points or next year's?"""
    data = team_season.sort_values(["lineage", "season"]).copy()
    data["points_next"] = data.groupby("lineage")["total_points"].shift(-1)
    data["points_prev"] = data.groupby("lineage")["total_points"].shift(1)
    rows = []
    for name, col in (("Previous season's points", "points_prev"), ("Same season's points", "total_points"),
                      ("Next season's points", "points_next")):
        pair = data.dropna(subset=["spend_total_musd", col])
        if len(pair) > 5:
            rho, p_value = stats.spearmanr(pair["spend_total_musd"], pair[col])
            rows.append({"Compared with": name, "Spearman r": rho, "p": p_value, "n": len(pair)})
    table = pd.DataFrame(rows)
    if table.empty:
        return EDAResult(theme.finish(go.Figure(), "Spend and points: lead or lag?", legend=False),
                         "Not enough team-seasons in the current selection to compare leads and lags.")
    fig = px.bar(table, x="Compared with", y="Spearman r", text="Spearman r", color_discrete_sequence=[config.RED])
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False, width=0.45)
    fig.update_layout(yaxis=dict(range=[min(0, table["Spearman r"].min() - 0.1) if len(table) else 0, 1.05]),
                      xaxis_title="Season spend (synthetic) correlated with")
    best = table.loc[table["Spearman r"].idxmax()]
    text = (f"Season spend lines up most closely with {best['Compared with'].lower()} "
            f"(Spearman {best['Spearman r']:.2f}). Spend is synthetic, and that is the generator's design: budgets "
            f"follow prize money "
            f"from the year before. It is also the textbook reverse-causality trap: points fund spend as much as "
            f"spend buys points.")
    return EDAResult(theme.finish(fig, "Spend and points: lead or lag?", height=380, legend=False), text, table)


def tyre_degradation_curves(laps: pd.DataFrame, min_laps: int = 20) -> EDAResult:
    """Median lap-time loss against tyre age, one curve per compound.

    Each stint is first expressed relative to its own early pace (the median of
    its first three clean laps), so fast and slow cars and circuits can be
    pooled. Lap times are fuel-corrected before the comparison.
    """
    data = laps[laps["is_clean"] & laps["compound"].isin(COMPOUND_ORDER)].dropna(subset=["tyre_life", "stint"]).copy()
    if data.empty:
        return EDAResult(theme.finish(go.Figure(), "Tyre degradation curves"), "No clean laps in the current selection.")
    keys = ["season", "round", "driver_code", "stint"]
    data["stint_lap"] = data.groupby(keys)["lap_number"].rank(method="first")
    base = data[data["stint_lap"] <= 3].groupby(keys)["fuel_corrected_s"].median().rename("base")
    data = data.merge(base, left_on=keys, right_index=True)
    data["delta_s"] = data["fuel_corrected_s"] - data["base"]
    data["tyre_age"] = data["tyre_life"].round().astype(int)
    curve = data.groupby(["compound", "tyre_age"])["delta_s"].agg(["median", "size"]).reset_index()
    curve = curve[curve["size"] >= min_laps]

    fig = go.Figure()
    slopes: dict[str, float] = {}
    for compound in COMPOUND_ORDER:
        part = curve[curve["compound"] == compound].sort_values("tyre_age")
        if len(part) < 4:
            continue
        slopes[compound] = float(np.polyfit(part["tyre_age"], part["median"], 1, w=np.sqrt(part["size"]))[0])
        fig.add_trace(go.Scatter(x=part["tyre_age"], y=part["median"], mode="lines+markers",
                                 name=compound.capitalize(), customdata=part["size"],
                                 line=dict(color=COMPOUND_COLOURS[compound], width=2), marker=dict(size=5),
                                 hovertemplate="Tyre age %{x} laps<br>%{y:+.2f}s vs early stint"
                                               "<br>%{customdata} laps<extra>" + compound.capitalize() + "</extra>"))
    fig.add_hline(y=0, line_color=theme.AXIS, line_dash="dot")
    fig.update_layout(xaxis_title="Tyre age (laps)", yaxis_title="Lap time vs start of stint (s, fuel-corrected)")
    if not slopes:
        return EDAResult(theme.finish(fig, "Tyre degradation curves"), "Too few laps per compound to fit a curve.")
    parts = ", ".join(f"{c.lower()} {v:+.3f}s/lap" for c, v in slopes.items())
    fastest = max(slopes, key=slopes.get)
    text = (f"Fitted degradation: {parts}. The {fastest.lower()} compound fades fastest. Curves that dip below "
            f"zero early show tyres coming up to temperature and the track rubbering in before wear takes over. "
            f"Fuel correction assumes {config.FUEL_EFFECT_S_PER_LAP}s per lap of fuel.")
    return EDAResult(theme.finish(fig, "Tyre degradation curves"), text, curve)


def degradation_by_season(stints: pd.DataFrame) -> EDAResult:
    """Median degradation rate per compound per season."""
    data = stints.dropna(subset=["deg_slope_s_per_lap"])
    data = data[data["compound"].isin(["SOFT", "MEDIUM", "HARD"])]
    trend = data.groupby(["season", "compound"])["deg_slope_s_per_lap"].agg(["median", "size"]).reset_index()
    trend = trend[trend["size"] >= 10]
    fig = px.line(trend, x="season", y="median", color="compound", markers=True,
                  color_discrete_map=COMPOUND_COLOURS, category_orders={"compound": COMPOUND_ORDER},
                  labels={"season": "Season", "median": "Median degradation (s/lap)", "compound": "Compound"})
    fig.update_xaxes(dtick=1)
    if trend.empty:
        return EDAResult(theme.finish(fig, "Tyre degradation by season"), "No stint data in the current selection.")
    by_season = trend.groupby("season")["median"].mean()
    text = (f"Average dry-compound degradation runs from {by_season.min():.3f}s/lap ({by_season.idxmin()}) to "
            f"{by_season.max():.3f}s/lap ({by_season.idxmax()}). Car regulations (18-inch tyres and ground-effect "
            f"cars from 2022) and the compounds nominated each year both move this line.")
    return EDAResult(theme.finish(fig, "Tyre degradation by season"), text, trend)


def pit_trend(pitstops: pd.DataFrame) -> EDAResult:
    """Which teams gain or lose time in the pit lane, compared with the field in the same race."""
    data = pitstops[pitstops["pit_valid"]].dropna(subset=["pit_delta_s"])
    trend = data.groupby(["season", "team"], as_index=False)["pit_delta_s"].median()
    if data.empty:
        return EDAResult(theme.finish(go.Figure(), "Pit stops compared with the field", legend=False),
                         "No pit-stop data in the current selection.")
    overall = data.groupby("team")["pit_delta_s"].median().sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=overall.to_numpy(), y=[theme.team_short(t) for t in overall.index], orientation="h",
        marker_color=[TEAM_COLOUR_MAP.get(t, config.MUTED) for t in overall.index],
        text=[f"{v:+.2f} s" for v in overall], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:+.2f} s against the race median<extra></extra>"))
    fig.add_vline(x=0, line_color=theme.AXIS)
    fig.update_layout(xaxis_title="Typical pit stop vs the race median (seconds; left is quicker)", yaxis_title="")
    text = (f"{theme.team_short(overall.index[-1])} has the quickest typical stop ({overall.iloc[-1]:+.2f} s against "
            f"the race median) and {theme.team_short(overall.index[0])} the slowest ({overall.iloc[0]:+.2f} s). "
            f"The whole spread is {overall.iloc[0] - overall.iloc[-1]:.2f} s, so pit crews are closely matched.")
    return EDAResult(theme.finish(fig, "Pit stops compared with the field", height=420, legend=False), text, trend)


def rolling_relationship(team_race: pd.DataFrame, x: str, y: str = "points") -> EDAResult:
    """How the correlation between a factor and an outcome changes season by season."""
    rows = []
    for season, grp in team_race.groupby("season"):
        pair = grp.dropna(subset=[x, y])
        if len(pair) > 20 and pair[x].nunique() > 3:
            rho, p_value = stats.spearmanr(pair[x], pair[y])
            rows.append({"season": season, "r": rho, "p": p_value, "n": len(pair)})
    table = pd.DataFrame(rows)
    fig = go.Figure()
    if not table.empty:
        fig.add_trace(go.Bar(x=table["season"], y=table["r"], marker_color=np.where(table["r"] >= 0, config.RED, config.CYAN),
                             text=table["r"].round(2), textposition="outside", cliponaxis=False,
                             hovertemplate="%{x}<br>Spearman %{y:.2f}<extra></extra>"))
    fig.add_hline(y=0, line_color=theme.AXIS)
    fig.update_layout(xaxis=dict(dtick=1, title="Season"), yaxis=dict(title=f"Spearman r with {plain(y)}",
                                                                      range=[-1, 1]))
    if table.empty:
        return EDAResult(theme.finish(fig, f"{short(x)} vs {short(y)} by season", legend=False),
                         "Not enough data per season.")
    stable = table["r"].std()
    text = (f"The link between {plain(x)} and {plain(y)} averages {table['r'].mean():+.2f} and is "
            f"{'stable' if stable < 0.1 else 'unstable'} across seasons (range {table['r'].min():+.2f} to "
            f"{table['r'].max():+.2f}). Overall strength: {describe_strength(table['r'].mean())} relationship.")
    return EDAResult(theme.finish(fig, f"{short(x)} vs {short(y)} by season", height=380, legend=False), text, table)
