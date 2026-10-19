"""EDA step 1 - univariate profiling.

What does each variable look like on its own: where is the middle, how wide is
the spread, is it lopsided (skew) and how heavy are the tails (kurtosis)?
Covers the four headline variables of the project - lap time, spend, pit delta
and tyre wear - plus a generic profiler for any numeric column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from src import config, theme
from src.eda import EDAResult, label, origin, tagged, short, sentence

COMPOUND_COLOURS: dict[str, str] = {
    "SOFT": config.RED, "MEDIUM": "#FFB800", "HARD": "#D5D8DF",
    "INTERMEDIATE": "#3DDC97", "WET": "#4D8DFF",
}
COMPOUND_ORDER: list[str] = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


def profile(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Summary statistics, shape measures and a normality test per column."""
    rows = []
    for col in cols:
        if col not in frame:
            continue
        series = pd.to_numeric(frame[col], errors="coerce")
        clean = series.dropna()
        if clean.empty:
            continue
        normal_p = stats.normaltest(clean).pvalue if len(clean) >= 20 and clean.std() > 0 else np.nan
        rows.append({
            "column": col, "label": label(col), "origin": origin(col), "n": len(clean),
            "missing_pct": series.isna().mean() * 100, "mean": clean.mean(), "median": clean.median(),
            "std": clean.std(), "min": clean.min(), "q1": clean.quantile(0.25), "q3": clean.quantile(0.75),
            "max": clean.max(), "skew": clean.skew(), "kurtosis": clean.kurt(), "normality_p": normal_p,
        })
    return pd.DataFrame(rows)


def shape_sentence(series: pd.Series, unit: str = "") -> str:
    """Describe a distribution's centre, skew and tails in plain English."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 3:
        return "Not enough data in the current selection to describe this distribution."
    skew, kurt = clean.skew(), clean.kurt()
    if abs(skew) < 0.5:
        lean = "roughly symmetric"
    else:
        side = "right" if skew > 0 else "left"
        lean = f"{'strongly' if abs(skew) > 1 else 'moderately'} {side}-skewed"
    tails = "heavy tails with frequent extremes" if kurt > 1 else ("light tails" if kurt < -1 else "ordinary tails")
    text = (f"The median is {clean.median():,.2f}{unit} against a mean of {clean.mean():,.2f}{unit}. "
            f"The distribution is {lean} (skew {skew:.2f}) with {tails} (excess kurtosis {kurt:.2f}).")
    if abs(skew) >= 1:
        text += " Use the median, not the mean, as the typical value."
    return text


def distribution(frame: pd.DataFrame, col: str, by: str | None = None, bins: int = 40) -> EDAResult:
    """Histogram with a box plot on top, optionally split by a category."""
    data = frame.dropna(subset=[col])
    fig = px.histogram(data, x=col, color=by, nbins=bins, marginal="box", opacity=0.85,
                       labels={col: tagged(col)},
                       color_discrete_sequence=[config.RED] if by is None else config.COLORWAY)
    fig.update_layout(barmode="overlay" if by else "relative", yaxis_title="Count")
    return EDAResult(theme.finish(fig, f"Distribution of {short(col)}", legend=by is not None),
                     shape_sentence(data[col]), profile(data, [col]))


def _box_stats(values: pd.Series) -> dict[str, list[float]]:
    """Five-number summary for a pre-computed Plotly box (Tukey whiskers)."""
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    spread = q3 - q1
    inside = values[(values >= q1 - 1.5 * spread) & (values <= q3 + 1.5 * spread)]
    return {"q1": [q1], "median": [median], "q3": [q3], "lowerfence": [inside.min()], "upperfence": [inside.max()]}


def lap_time_distribution(laps: pd.DataFrame) -> EDAResult:
    """Lap time relative to the race median, so different circuits can be pooled.

    Laps are binned here rather than in the browser: a season of racing is over
    20,000 laps, and sending counts instead of raw points keeps the chart instant.
    """
    if laps.empty:
        return EDAResult(theme.finish(go.Figure(), "Lap time distribution"), "No lap data in the current selection.")
    data = laps[["season", "round", "lap_time_s", "is_clean"]].copy()
    reference = data[data["is_clean"]].groupby(["season", "round"])["lap_time_s"].median().rename("ref")
    data = data.merge(reference, left_on=["season", "round"], right_index=True)
    data["lap_delta_pct"] = (data["lap_time_s"] / data["ref"] - 1) * 100
    clean, other = data.loc[data["is_clean"], "lap_delta_pct"], data.loc[~data["is_clean"], "lap_delta_pct"]
    allv = data["lap_delta_pct"]

    edges = np.arange(-5.0, 30.01, 0.4)
    centres = (edges[:-1] + edges[1:]) / 2
    groups = [("Clean racing lap", clean, config.CYAN), ("Pit, safety car or slow lap", other, config.RED)]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.16, 0.84], vertical_spacing=0.03)
    for name, values, colour in groups:
        if values.empty:
            continue
        counts, _ = np.histogram(values.clip(upper=edges[-1] - 1e-9), bins=edges)
        fig.add_trace(go.Bar(x=centres, y=counts, name=name, marker_color=colour, opacity=0.8, width=0.4,
                             legendgroup=name, hovertemplate="%{x:.1f}%: %{y:,} laps<extra>" + name + "</extra>"),
                      row=2, col=1)
        fig.add_trace(go.Box(y=[name], orientation="h", name=name, marker_color=colour, line_color=colour,
                             legendgroup=name, showlegend=False, hoverinfo="skip", **_box_stats(values)), row=1, col=1)
    fig.update_layout(barmode="overlay", bargap=0)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=1, col=1)
    fig.update_yaxes(title_text="Laps", row=2, col=1)
    fig.update_xaxes(title_text="Lap time vs race median clean lap (%); slower laps are grouped at +30%",
                     range=[-5, 30], row=2, col=1)
    text = (f"Across {len(data):,} laps the raw distribution is extremely right-skewed (skew {allv.skew():.1f}): "
            f"pit, safety-car and incident laps form a long slow tail. Keeping only the {len(clean):,} clean laps "
            f"({len(clean) / len(data):.0%}) leaves a tight band with a spread of {clean.std():.2f}% around the median.")
    table = profile(data.assign(clean_delta=clean), ["lap_delta_pct", "clean_delta"])
    return EDAResult(theme.finish(fig, "Lap time distribution"), text, table)


def spend_distribution(component: pd.DataFrame) -> EDAResult:
    """SYNTHETIC spend per round by car component."""
    data = component.copy()
    data["Component"] = data["component"].str.replace("_", " ").str.capitalize()
    order = data.groupby("Component")["spend_musd"].median().sort_values(ascending=False).index.tolist()
    fig = px.box(data, x="Component", y="spend_musd", category_orders={"Component": order}, points=False,
                 color_discrete_sequence=[config.RED],
                 labels={"spend_musd": "Spend per round (USD M, synthetic)", "Component": ""})
    fig.update_traces(fillcolor="rgba(225,6,0,0.18)")
    top = data.groupby("Component")["spend_musd"].sum().sort_values(ascending=False)
    text = (f"{top.index[0]} takes the largest share of spend ({top.iloc[0] / top.sum():.0%}), "
            f"followed by {top.index[1].lower()} ({top.iloc[1] / top.sum():.0%}). Most rounds are routine and a few "
            f"carry upgrade bursts, so every component is right-skewed (skew {data['spend_musd'].skew():.1f}). "
            f"Spend is synthetic.")
    return EDAResult(theme.finish(fig, "Spend by component", legend=False), text,
                     data.groupby("Component")["spend_musd"].describe().reset_index())


def pit_distribution(pitstops: pd.DataFrame) -> EDAResult:
    """Pit-lane time per season, plausible stops only."""
    data = pitstops[pitstops["pit_valid"]].copy()
    data["Season"] = data["season"].astype(str)
    fig = px.violin(data, x="Season", y="duration_s", box=True, points=False,
                    color_discrete_sequence=[config.CYAN],
                    labels={"duration_s": "Pit-lane time (s)"})
    fig.update_traces(meanline_visible=True, fillcolor="rgba(0,229,255,0.12)", line_color=config.CYAN)
    medians = data.groupby("season")["duration_s"].median()
    dropped = len(pitstops) - len(data)
    text = (f"The typical pit-lane time is {data['duration_s'].median():.1f}s, ranging from {medians.min():.1f}s "
            f"({medians.idxmin()}) to {medians.max():.1f}s ({medians.idxmax()}) by season; pit-lane length differs by "
            f"circuit, which is why the analysis uses the gap to the field in the same race. "
            f"{dropped} stops ({dropped / max(len(pitstops), 1):.0%}) outside 12-60s are excluded as red-flag waits or repairs.")
    return EDAResult(theme.finish(fig, "Pit-lane time by season", legend=False), text,
                     data.groupby("season")["duration_s"].describe().reset_index())


def pit_delta_distribution(pitstops: pd.DataFrame) -> EDAResult:
    """Pit delta: each stop's time minus the field median for that race."""
    data = pitstops[pitstops["pit_valid"]].dropna(subset=["pit_delta_s"])
    fig = px.histogram(data[data["pit_delta_s"].between(-5, 15)], x="pit_delta_s", nbins=80, marginal="box",
                       color_discrete_sequence=[config.RED],
                       labels={"pit_delta_s": "Pit time vs field median in the same race (s)"})
    fig.add_vline(x=0, line_color=config.CYAN, line_dash="dot")
    fig.update_layout(yaxis_title="Pit stops")
    slow = (data["pit_delta_s"] > 3).mean()
    text = (shape_sentence(data["pit_delta_s"], "s") +
            f" {slow:.0%} of stops lose more than 3s to the field: slow stops are rare but costly.")
    return EDAResult(theme.finish(fig, "Pit delta distribution", legend=False), text, profile(data, ["pit_delta_s"]))


def wear_distribution(stints: pd.DataFrame) -> EDAResult:
    """Tyre degradation rate per stint, by compound."""
    data = stints.dropna(subset=["deg_slope_s_per_lap", "compound"])
    data = data[data["compound"].isin(COMPOUND_ORDER)]
    if data.empty:
        return EDAResult(theme.finish(go.Figure(), "Tyre wear by compound"), "No stint data in the current selection.")
    fig = px.box(data, x="compound", y="deg_slope_s_per_lap", color="compound", points=False,
                 category_orders={"compound": COMPOUND_ORDER}, color_discrete_map=COMPOUND_COLOURS,
                 labels={"deg_slope_s_per_lap": "Degradation (s lost per lap of tyre age)", "compound": "Compound"})
    fig.add_hline(y=0, line_color=theme.AXIS, line_dash="dot")
    medians = data.groupby("compound")["deg_slope_s_per_lap"].median().reindex(COMPOUND_ORDER).dropna()
    parts = ", ".join(f"{c.lower()} {v:+.3f}" for c, v in medians.items())
    text = (f"Median fuel-corrected degradation in seconds per lap: {parts}. "
            f"Values are noisy stint to stint (spread {data['deg_slope_s_per_lap'].std():.3f}s/lap) because traffic, "
            f"tyre management and track evolution all leak into the slope.")
    return EDAResult(theme.finish(fig, "Tyre wear by compound", legend=False), text,
                     data.groupby("compound")["deg_slope_s_per_lap"].describe().reset_index())


def shape_overview(frame: pd.DataFrame, cols: list[str]) -> EDAResult:
    """Skew of every column in one chart, to spot the lopsided ones at a glance."""
    table = profile(frame, cols).sort_values("skew")
    colours = [config.RED if abs(s) >= 1 else config.CYAN for s in table["skew"]]
    fig = go.Figure(go.Bar(x=table["skew"], y=table["label"], orientation="h", marker_color=colours,
                           customdata=table[["kurtosis", "origin"]],
                           hovertemplate="%{y}<br>Skew %{x:.2f}<br>Excess kurtosis %{customdata[0]:.2f}"
                                         "<br>%{customdata[1]}<extra></extra>"))
    fig.add_vline(x=1, line_color=theme.AXIS, line_dash="dot")
    fig.add_vline(x=-1, line_color=theme.AXIS, line_dash="dot")
    fig.update_layout(xaxis_title="Skewness (red = beyond +/-1)", yaxis_title="")
    lopsided = table[table["skew"].abs() >= 1]
    worst = table.iloc[table["skew"].abs().argmax()]
    text = (f"{len(lopsided)} of {len(table)} variables are strongly skewed (|skew| >= 1); the most lopsided is "
            f"{sentence(worst['label'])} at {worst['skew']:.1f}. Those are the ones where rank-based statistics "
            f"(median, Spearman) are safer than means and Pearson correlations.")
    return EDAResult(theme.finish(fig, "Skewness across variables", height=max(360, 24 * len(table) + 120),
                                  legend=False), text, table)
