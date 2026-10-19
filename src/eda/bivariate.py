"""EDA step 2 - bivariate and multivariate analysis.

How do variables move together? Correlation heatmaps, pair plots and the
project's two headline cross-domain questions: does spend buy reliability, and
does weather change pace?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from src import config, theme
from src.eda import EDAResult, describe_strength, label, tagged, plain, short, origin


def correlation_matrix(frame: pd.DataFrame, cols: list[str], method: str = "spearman") -> pd.DataFrame:
    """Pairwise correlation of numeric columns (pairwise-complete observations)."""
    present = [c for c in cols if c in frame.columns]
    data = frame[present].apply(pd.to_numeric, errors="coerce")
    return data.corr(method=method, min_periods=10)


def strongest_pairs(corr: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    """The off-diagonal pairs with the largest absolute correlation."""
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(mask).stack().rename("r").reset_index()
    pairs.columns = ["a", "b", "r"]
    return pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)


# A readable default: one or two variables from each domain plus the outcomes.
KEY_VARIABLES: list[str] = [
    "spend_total_musd", "aero_share", "late_parts", "spares_coverage_pct", "crew_fatigue_idx", "sim_hours",
    "sim_track_corr_pct", "pit_delta_s", "track_temp_c", "pace_delta_pct", "deg_slope_s_per_lap", "mean_grid",
    "points", "dnf",
]


def correlation_heatmap(frame: pd.DataFrame, cols: list[str], method: str = "spearman") -> EDAResult:
    """Correlation heatmap, lower triangle only, on a cyan-to-red scale centred on zero.

    Only the lower triangle is drawn because the upper half repeats it; with the
    diagonal removed every cell on screen is a distinct pair.
    """
    corr = correlation_matrix(frame, cols, method)
    mask = np.tril(np.ones(corr.shape, dtype=bool), k=-1)
    shown = corr.where(mask).iloc[1:, :-1]
    rows = [short(c) + (" *" if origin(c) == "SYNTHETIC" else "") for c in shown.index]
    columns = [short(c) + (" *" if origin(c) == "SYNTHETIC" else "") for c in shown.columns]
    few = len(corr) <= 16
    fig = go.Figure(go.Heatmap(
        z=shown.to_numpy(), x=columns, y=rows, zmin=-1, zmax=1, colorscale=config.DIVERGING,
        text=np.where(np.isnan(shown.to_numpy()), "", np.round(shown.to_numpy(), 2).astype(str)) if few else None,
        texttemplate="%{text}" if few else None, textfont=dict(size=12.5, color="#FFFFFF"),
        hovertemplate="%{y} and %{x}<br>r = %{z:.2f}<extra></extra>", hoverongaps=False,
        colorbar=dict(title=dict(text="r", side="top"), thickness=10, len=0.6)))
    fig.update_xaxes(tickangle=-40, showgrid=False, ticks="")
    fig.update_yaxes(showgrid=False, autorange="reversed", ticks="")
    pairs = strongest_pairs(corr, 30)
    if pairs.empty:
        return EDAResult(theme.finish(fig, "How the variables move together", legend=False),
                         "Not enough data to correlate.", corr)
    pos = pairs[pairs["r"] > 0].head(1)
    neg = pairs[pairs["r"] < 0].head(1)
    text = ""
    if not pos.empty:
        text += (f"{short(pos['a'].iloc[0])} and {plain(pos['b'].iloc[0])} rise together most strongly "
                 f"(r = {pos['r'].iloc[0]:.2f}). ")
    if not neg.empty:
        text += (f"The strongest opposite pair is {plain(neg['a'].iloc[0])} and {plain(neg['b'].iloc[0])} "
                 f"(r = {neg['r'].iloc[0]:.2f}). ")
    text += ("Red cells rise together, cyan cells move in opposite directions and dark cells are unrelated. "
             "A star marks synthetic data.")
    size = max(460, 30 * len(rows) + 200)
    return EDAResult(theme.finish(fig, "How the variables move together", height=size, legend=False), text, corr)


def scatter_trend(frame: pd.DataFrame, x: str, y: str, color: str | None = "team",
                  hover: str | None = None) -> EDAResult:
    """Scatter of ``y`` against ``x`` with a least-squares trend line and correlation."""
    data = frame.dropna(subset=[x, y]).copy()
    colour_map = {theme.team_label(k): v for k, v in config.TEAM_COLOR.items()} if color == "team" else None
    fig = px.scatter(data, x=x, y=y, color=color if color in data else None, color_discrete_map=colour_map,
                     hover_name=hover if hover in data else None, opacity=0.75,
                     labels={x: tagged(x), y: tagged(y), "team": "Team"})
    if len(data) < 5 or data[x].nunique() < 3:
        return EDAResult(theme.finish(fig, f"{short(y)} vs {short(x)}"),
                         "Not enough data in the current selection to estimate a relationship.")
    slope, intercept = np.polyfit(data[x].astype(float), data[y].astype(float), 1)
    xs = np.linspace(data[x].min(), data[x].max(), 50)
    fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Linear trend",
                             line=dict(color=config.TEXT, width=2, dash="dash"), hoverinfo="skip"))
    rho, p_value = stats.spearmanr(data[x], data[y])
    pearson = stats.pearsonr(data[x].astype(float), data[y].astype(float))[0]
    direction = "rises" if rho > 0 else "falls"
    sig = "statistically clear" if p_value < 0.01 else ("suggestive" if p_value < 0.05 else "not distinguishable from noise")
    text = (f"There is {describe_strength(rho)} relationship: {plain(y)} {direction} as {plain(x)} "
            f"increases (Spearman {rho:.2f}, Pearson {pearson:.2f}, n = {len(data)}), which is {sig} "
            f"(p = {p_value:.3g}).")
    return EDAResult(theme.finish(fig, f"{short(y)} vs {short(x)}"), text)


def pair_plot(frame: pd.DataFrame, cols: list[str], color: str | None = "team") -> EDAResult:
    """Scatter-plot matrix of up to six variables."""
    cols = [c for c in cols if c in frame.columns][:6]
    data = frame.dropna(subset=cols)
    colour_map = {theme.team_label(k): v for k, v in config.TEAM_COLOR.items()} if color == "team" else None
    fig = px.scatter_matrix(data, dimensions=cols, color=color if color in data else None,
                            color_discrete_map=colour_map, opacity=0.6,
                            labels={c: label(c).replace(" (", "<br>(") for c in cols} | {"team": "Team"})
    fig.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(size=4, line=dict(width=0)))
    fig.update_layout(font=dict(size=10))
    pairs = strongest_pairs(correlation_matrix(data, cols), 1)
    text = "Each panel plots one pair of variables; tight diagonal bands mean a strong relationship."
    if not pairs.empty:
        text += (f" The tightest pair here is {plain(pairs['a'].iloc[0])} and "
                 f"{plain(pairs['b'].iloc[0])} (Spearman {pairs['r'].iloc[0]:.2f}).")
    return EDAResult(theme.finish(fig, "Pair plot", height=720), text)


def spend_vs_reliability(team_season: pd.DataFrame) -> EDAResult:
    """Season spend (SYNTHETIC) against finish rate (REAL), one bubble per team-season."""
    data = team_season.dropna(subset=["spend_total_musd", "finish_rate"]).copy()
    data["Season"] = data["season"].astype(str)
    data["size"] = data["total_points"].clip(lower=5)
    fig = px.scatter(data, x="spend_total_musd", y="finish_rate", color="team", size="size", size_max=28,
                     color_discrete_map={theme.team_label(k): v for k, v in config.TEAM_COLOR.items()},
                     hover_name="constructor_name",
                     hover_data={"Season": True, "total_points": True, "size": False, "team": False},
                     labels={"spend_total_musd": "Season spend (USD M, synthetic)",
                             "finish_rate": "Share of entries that finished", "team": "Team",
                             "total_points": "Points"})
    fig.update_layout(yaxis_tickformat=".0%")
    if len(data) < 5:
        return EDAResult(theme.finish(fig, "Spend vs reliability"), "Not enough team-seasons selected.")
    rho, p_value = stats.spearmanr(data["spend_total_musd"], data["finish_rate"])
    capped = data[data["season"] >= 2021]
    rho_cap = stats.spearmanr(capped["spend_total_musd"], capped["finish_rate"])[0] if len(capped) > 5 else np.nan
    text = (f"Across {len(data)} team-seasons, bigger budgets show {describe_strength(rho)} link with finishing "
            f"races (Spearman {rho:.2f}, p = {p_value:.2g}). "
            + (f"Inside the cost-cap era (2021 on), where budgets are compressed, it is {rho_cap:.2f}. "
               if np.isfinite(rho_cap) else "")
            + f"Bubble size is points; spend against points gives Spearman "
              f"{stats.spearmanr(data['spend_total_musd'], data['total_points'])[0]:.2f}. Spend is synthetic.")
    return EDAResult(theme.finish(fig, "Spend vs reliability", height=480), text)


def weather_vs_pace(races: pd.DataFrame, team_race: pd.DataFrame) -> EDAResult:
    """Track temperature against tyre degradation, one point per race."""
    deg = team_race.groupby(["season", "round"], as_index=False)["deg_slope_s_per_lap"].median()
    data = races.merge(deg, on=["season", "round"]).dropna(subset=["track_temp_c", "deg_slope_s_per_lap"])
    data["Conditions"] = np.where(data["wet_race"].fillna(False).astype(bool), "Rain-affected", "Dry")
    fig = px.scatter(data, x="track_temp_c", y="deg_slope_s_per_lap", color="Conditions",
                     color_discrete_map={"Dry": config.RED, "Rain-affected": config.CYAN},
                     hover_name="race_name", hover_data={"season": True},
                     labels={"track_temp_c": "Mean track temperature (C)",
                             "deg_slope_s_per_lap": "Median tyre degradation (s/lap)", "season": "Season"})
    dry = data[data["Conditions"] == "Dry"]
    if len(dry) < 8:
        return EDAResult(theme.finish(fig, "Weather vs pace: track temperature and tyre wear"),
                         "Too few races with lap and weather data in the current selection.")
    slope, intercept = np.polyfit(dry["track_temp_c"], dry["deg_slope_s_per_lap"], 1)
    xs = np.linspace(dry["track_temp_c"].min(), dry["track_temp_c"].max(), 40)
    fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Dry-race trend",
                             line=dict(color=config.TEXT, dash="dash", width=2), hoverinfo="skip"))
    rho, p_value = stats.spearmanr(dry["track_temp_c"], dry["deg_slope_s_per_lap"])
    if p_value >= 0.05:
        text = (f"Across {len(dry)} dry races there is no statistically clear link between track temperature and "
                f"tyre degradation (Spearman {rho:.2f}, p = {p_value:.2g}). Circuit layout, compound choice and tyre "
                f"management vary alongside temperature and swamp any direct effect at race level.")
    else:
        text = (f"Across {len(dry)} dry races, hotter tracks show {describe_strength(rho)} link with tyre degradation "
                f"(Spearman {rho:.2f}, p = {p_value:.2g}); each extra 10 C of track temperature goes with "
                f"{slope * 10:+.3f}s per lap of wear. Circuit layout and compound choice vary alongside temperature, "
                f"so this is an association, not a clean causal effect.")
    return EDAResult(theme.finish(fig, "Weather vs pace: track temperature and tyre wear"), text)


def wet_vs_dry(races: pd.DataFrame) -> EDAResult:
    """Retirement rate in rain-affected races compared with dry ones."""
    data = races.dropna(subset=["wet_race"]).copy()
    data["Conditions"] = np.where(data["wet_race"].astype(bool), "Rain-affected", "Dry")
    fig = px.box(data, x="Conditions", y="dnf_rate", color="Conditions", points="all",
                 color_discrete_map={"Dry": config.RED, "Rain-affected": config.CYAN},
                 hover_name="race_name", labels={"dnf_rate": "Share of starters that retired"})
    fig.update_layout(yaxis_tickformat=".0%")
    means = data.groupby("Conditions")["dnf_rate"].agg(["mean", "size"])
    if len(means) < 2:
        return EDAResult(theme.finish(fig, "Retirements: wet vs dry races", legend=False),
                         "The current selection does not contain both wet and dry races.")
    test = stats.mannwhitneyu(data.loc[data["Conditions"] == "Dry", "dnf_rate"],
                              data.loc[data["Conditions"] == "Rain-affected", "dnf_rate"])
    text = (f"Rain-affected races average {means.loc['Rain-affected', 'mean']:.1%} retirements "
            f"({int(means.loc['Rain-affected', 'size'])} races) against {means.loc['Dry', 'mean']:.1%} in the dry "
            f"({int(means.loc['Dry', 'size'])} races); Mann-Whitney p = {test.pvalue:.2g}, "
            f"{'a statistically clear difference' if test.pvalue < 0.05 else 'so with this many wet races the gap could still be chance'}. "
            f"Weather shifts the risk for the whole field at once rather than for one team.")
    return EDAResult(theme.finish(fig, "Retirements: wet vs dry races", legend=False), text, means.reset_index())


def parallel_profile(team_season: pd.DataFrame, cols: list[str]) -> EDAResult:
    """Parallel-coordinates view of team-seasons across many dimensions at once."""
    cols = [c for c in cols if c in team_season.columns]
    data = team_season.dropna(subset=cols + ["total_points"])
    fig = px.parallel_coordinates(data, dimensions=cols + ["total_points"], color="total_points",
                                  color_continuous_scale=[[0, "#4A4A55"], [1, config.RED]],
                                  labels={c: label(c).split(" (")[0] for c in cols} | {"total_points": "Points"})
    fig.update_layout(margin=dict(l=80, r=60, t=110, b=40))
    text = ("Each line is one team-season, coloured by points (grey low, red high). Where red lines bunch on an "
            "axis, that factor separates front-runners from the rest; where colours mix, it does not.")
    return EDAResult(theme.finish(fig, "Multivariate profile of team-seasons", height=480, legend=False), text)
