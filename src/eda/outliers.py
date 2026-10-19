"""EDA step 3 - outlier and anomaly detection.

Three detectors with different blind spots are combined by vote:

* **IQR fences** - robust, univariate, no distribution assumption;
* **z-score** - classic, sensitive to the very outliers it hunts;
* **Isolation Forest** - multivariate, finds odd *combinations* of values.

Retirements (DNFs) and component failures are the domain's own anomalies and
get their own summaries. Rows on the analyst watchlist (see ``src.feedback``)
are highlighted so dashboard flags feed back into the next EDA pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src import config, theme
from src.eda import EDAResult, label, tagged, short


# ------------------------------------------------------------------ primitives
def iqr_bounds(values: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Tukey fences ``(Q1 - k*IQR, Q3 + k*IQR)``; NaNs are ignored."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan"), float("nan")
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    spread = q3 - q1
    return float(q1 - k * spread), float(q3 + k * spread)


def iqr_flags(values: pd.Series, k: float = 1.5) -> pd.Series:
    """True where a value lies outside the Tukey fences. NaN is never flagged."""
    numeric = pd.to_numeric(values, errors="coerce")
    low, high = iqr_bounds(numeric, k)
    return ((numeric < low) | (numeric > high)).fillna(False).astype(bool)


def zscore_flags(values: pd.Series, threshold: float = 3.0) -> pd.Series:
    """True where ``|z| > threshold``. A constant series flags nothing."""
    numeric = pd.to_numeric(values, errors="coerce")
    std = numeric.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(False, index=values.index)
    return (((numeric - numeric.mean()) / std).abs() > threshold).fillna(False).astype(bool)


def robust_zscore(values: pd.Series) -> pd.Series:
    """Median/MAD z-score, which a single huge value cannot mask.

    When more than half the values are identical the MAD is zero; the mean
    absolute deviation is used instead (the usual fallback), so a lone extreme
    among otherwise constant values is still scored.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    deviation = (numeric - median).abs()
    mad = deviation.median()
    if np.isfinite(mad) and mad > 0:
        return 0.6745 * (numeric - median) / mad
    mean_ad = deviation.mean()
    if not np.isfinite(mean_ad) or mean_ad == 0:
        return pd.Series(0.0, index=values.index).where(numeric.notna())
    return (numeric - median) / (1.253314 * mean_ad)


def mad_flags(values: pd.Series, threshold: float = 3.5) -> pd.Series:
    """True where the robust z-score exceeds ``threshold`` in absolute value."""
    return (robust_zscore(values).abs() > threshold).fillna(False).astype(bool)


def grouped_flags(frame: pd.DataFrame, col: str, by: list[str] | None = None,
                  method: str = "iqr", **kwargs: float) -> pd.Series:
    """Apply a detector within groups, e.g. per race, so fast and slow tracks do not mix."""
    func = {"iqr": iqr_flags, "zscore": zscore_flags, "mad": mad_flags}[method]
    if not by:
        return func(frame[col], **kwargs)
    return frame.groupby(by, group_keys=False)[col].apply(lambda s: func(s, **kwargs)).reindex(frame.index)


def isolation_forest_flags(frame: pd.DataFrame, cols: list[str], contamination: float = 0.03,
                           seed: int = config.SEED) -> pd.DataFrame:
    """Multivariate anomaly score per row.

    Missing values are median-filled and features standardised first. Returns
    ``iso_score`` (higher = more anomalous) and the boolean ``iso_flag``.
    """
    data = frame[cols].apply(pd.to_numeric, errors="coerce")
    data = data.fillna(data.median()).fillna(0.0)
    if len(data) < 10:
        return pd.DataFrame({"iso_score": 0.0, "iso_flag": False}, index=frame.index)
    scaled = StandardScaler().fit_transform(data)
    forest = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
    flags = forest.fit_predict(scaled) == -1
    return pd.DataFrame({"iso_score": -forest.score_samples(scaled), "iso_flag": flags}, index=frame.index)


def detect_anomalies(frame: pd.DataFrame, cols: list[str], by: list[str] | None = None,
                     contamination: float = 0.03, seed: int = config.SEED) -> pd.DataFrame:
    """Run all three detectors and combine them by vote.

    Adds ``iqr_hits`` and ``z_hits`` (how many columns each rule flagged),
    ``iso_flag``/``iso_score``, ``anomaly_votes`` (0-3 detectors agreeing),
    ``anomaly`` (two or more agree) and ``anomaly_reason`` (the offending columns).
    """
    out = frame.copy()
    iqr = pd.DataFrame({c: grouped_flags(out, c, by, "iqr") for c in cols})
    z = pd.DataFrame({c: grouped_flags(out, c, by, "zscore") for c in cols})
    iso = isolation_forest_flags(out, cols, contamination, seed)
    out["iqr_hits"] = iqr.sum(axis=1)
    out["z_hits"] = z.sum(axis=1)
    out["iso_flag"], out["iso_score"] = iso["iso_flag"], iso["iso_score"]
    out["anomaly_votes"] = (out["iqr_hits"] > 0).astype(int) + (out["z_hits"] > 0).astype(int) + out["iso_flag"].astype(int)
    out["anomaly"] = out["anomaly_votes"] >= 2
    either = (iqr | z)
    out["anomaly_reason"] = [", ".join(label(c) for c in cols if row[c]) for _, row in either.iterrows()]
    out.loc[(out["anomaly_reason"] == "") & out["iso_flag"], "anomaly_reason"] = "unusual combination"
    return out


def annotate_watchlist(frame: pd.DataFrame, watchlist: pd.DataFrame, key: str = "team_race_key") -> pd.DataFrame:
    """Mark rows whose key is on the analyst watchlist (the feedback loop)."""
    out = frame.copy()
    keys = set(watchlist["key"]) if watchlist is not None and not watchlist.empty else set()
    out["on_watchlist"] = out[key].isin(keys) if key in out else False
    return out


# --------------------------------------------------------------------- figures
def outlier_box(frame: pd.DataFrame, col: str, by: str = "season") -> EDAResult:
    """Box plot per group with the IQR-flagged points drawn in red."""
    data = frame.dropna(subset=[col]).copy()
    data["flag"] = grouped_flags(data, col, [by], "iqr")
    fig = go.Figure()
    fig.add_trace(go.Box(x=data[by].astype(str), y=data[col], name="Distribution", boxpoints=False,
                         marker_color=config.CYAN, line_color=config.CYAN, fillcolor="rgba(0,229,255,0.12)"))
    flagged = data[data["flag"]]
    hover = flagged["team_race_key"] if "team_race_key" in flagged else flagged.index.astype(str)
    fig.add_trace(go.Scatter(x=flagged[by].astype(str), y=flagged[col], mode="markers", name="Outside IQR fences",
                             marker=dict(color=config.RED, size=8), text=hover,
                             hovertemplate="%{text}<br>%{y:.2f}<extra></extra>"))
    fig.update_layout(xaxis_title=by.replace("_", " ").capitalize(), yaxis_title=tagged(col))
    share = data["flag"].mean() * 100 if len(data) else 0.0
    worst = data.groupby(by)["flag"].mean().idxmax() if len(data) else "n/a"
    text = (f"{len(flagged)} of {len(data)} values ({share:.1f}%) sit outside the IQR fences for their {by}. "
            f"{worst} has the highest share of flagged values.")
    return EDAResult(theme.finish(fig, f"Outliers in {short(col)} by {by}"), text, flagged)


def anomaly_scatter(scored: pd.DataFrame, x: str, y: str) -> EDAResult:
    """Scatter of two features with rows coloured by how many detectors flagged them."""
    data = scored.dropna(subset=[x, y]).copy()
    names = {0: "Normal", 1: "1 test", 2: "2 tests", 3: "All 3 tests"}
    colours = {"Normal": "#4A4A55", "1 test": config.CYAN, "2 tests": "#FFB800", "All 3 tests": config.RED}
    data["Flagged by"] = data["anomaly_votes"].map(names)
    key = data["team_race_key"] if "team_race_key" in data else data.index.astype(str)
    fig = px.scatter(data, x=x, y=y, color="Flagged by", color_discrete_map=colours,
                     category_orders={"Flagged by": list(names.values())},
                     hover_name=key, hover_data={"anomaly_reason": True, "Flagged by": False},
                     labels={x: tagged(x), y: tagged(y), "anomaly_reason": "Why"})
    if "on_watchlist" in data and data["on_watchlist"].any():
        watch = data[data["on_watchlist"]]
        fig.add_trace(go.Scatter(x=watch[x], y=watch[y], mode="markers", name="On watchlist",
                                 marker=dict(symbol="circle-open", size=16, color=config.TEXT, line=dict(width=2)),
                                 hoverinfo="skip"))
    n_anom = int(data["anomaly"].sum())
    text = (f"{n_anom} of {len(data)} team-race rows ({n_anom / max(len(data), 1) * 100:.1f}%) are flagged by at "
            f"least two of the three detectors. Agreement between methods matters more than any single flag.")
    return EDAResult(theme.finish(fig, f"Anomalies: {short(y)} vs {short(x)}"), text)


def detector_agreement(scored: pd.DataFrame) -> EDAResult:
    """How many rows each detector flags, and how often they agree."""
    counts = pd.DataFrame({
        "Detector": ["IQR fences", "Z-score > 3", "Isolation Forest", "Two or more agree", "All three agree"],
        "Rows flagged": [int((scored["iqr_hits"] > 0).sum()), int((scored["z_hits"] > 0).sum()),
                         int(scored["iso_flag"].sum()), int((scored["anomaly_votes"] >= 2).sum()),
                         int((scored["anomaly_votes"] == 3).sum())],
    })
    fig = px.bar(counts, x="Rows flagged", y="Detector", orientation="h", text="Rows flagged",
                 color_discrete_sequence=[config.CYAN])
    fig.update_traces(marker_color=[config.CYAN] * 3 + [config.RED] * 2, textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis=dict(autorange="reversed", title=""), xaxis_title="Rows flagged")
    iqr_n, z_n = counts["Rows flagged"].iloc[0], counts["Rows flagged"].iloc[1]
    text = (f"IQR fences flag {iqr_n} rows and the z-score rule {z_n}: the z-score is the stricter test because "
            f"extreme values inflate the very standard deviation it divides by. "
            f"{counts['Rows flagged'].iloc[3]} rows are flagged by at least two detectors.")
    return EDAResult(theme.finish(fig, "Detector agreement", height=340, legend=False), text, counts)


def dnf_summary(driver_race: pd.DataFrame) -> pd.DataFrame:
    """Retirement counts and rates per season and outcome class."""
    out = driver_race.groupby(["season", "outcome"]).size().rename("entries").reset_index()
    out["share"] = out["entries"] / out.groupby("season")["entries"].transform("sum")
    return out


def dnf_by_cause(driver_race: pd.DataFrame) -> EDAResult:
    """Stacked bars of non-finishes per season, split by cause."""
    summary = dnf_summary(driver_race)
    summary = summary[summary["outcome"] != "finished"]
    names = {"mechanical": "Mechanical failure", "incident": "Accident / collision",
             "unspecified": "Retired (cause not published)", "other": "DNS / DSQ / withdrew"}
    colours = {"Mechanical failure": config.RED, "Accident / collision": config.CYAN,
               "Retired (cause not published)": "#FFB800", "DNS / DSQ / withdrew": "#8A8D93"}
    summary["Cause"] = summary["outcome"].map(names)
    fig = px.bar(summary, x="season", y="share", color="Cause", color_discrete_map=colours,
                 category_orders={"Cause": list(names.values())},
                 labels={"season": "Season", "share": "Share of race entries"},
                 hover_data={"entries": True, "share": ":.1%"})
    fig.update_layout(barmode="stack", yaxis_tickformat=".0%", xaxis=dict(dtick=1))
    total = driver_race.groupby("season")["dnf"].mean()
    mech = (driver_race["outcome"] == "mechanical").sum()
    unspecified = (driver_race["outcome"] == "unspecified").sum()
    text = (f"Retirement rates run from {total.min():.1%} ({total.idxmin()}) to {total.max():.1%} "
            f"({total.idxmax()}) of entries. {mech} retirements have a published mechanical cause")
    text += (f"; {unspecified} are logged only as 'Retired', so the mechanical count is a lower bound."
             if unspecified else ".")
    return EDAResult(theme.finish(fig, "Non-finishes by cause and season"), text, summary)


def failures_by_component(driver_race: pd.DataFrame) -> EDAResult:
    """Which car systems cause the mechanical retirements."""
    mech = driver_race[driver_race["outcome"] == "mechanical"]
    counts = mech["component"].value_counts().rename_axis("component").reset_index(name="failures")
    counts["Component"] = counts["component"].str.replace("_", " ").str.capitalize()
    fig = px.bar(counts, x="failures", y="Component", orientation="h", text="failures",
                 color_discrete_sequence=[config.RED])
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis=dict(autorange="reversed", title=""), xaxis_title="Mechanical retirements")
    if counts.empty:
        return EDAResult(theme.finish(fig, "Failures by component", legend=False),
                         "No retirements with a published mechanical cause in the current selection.", counts)
    top = counts.iloc[0]
    text = (f"{top['Component']} is the most common failed system with {top['failures']} of {len(mech)} "
            f"mechanical retirements ({top['failures'] / len(mech):.0%}).")
    return EDAResult(theme.finish(fig, "Failures by component", height=380, legend=False), text, counts)


def dnf_heatmap(driver_race: pd.DataFrame) -> EDAResult:
    """Team x season retirement rate heatmap."""
    rate = driver_race.groupby(["lineage", "season"])["dnf"].mean().unstack("season")
    rate = rate.loc[rate.mean(axis=1).sort_values(ascending=False).index]
    fig = px.imshow(rate.values, x=[str(c) for c in rate.columns], y=[theme.team_label(i) for i in rate.index],
                    color_continuous_scale=config.SEQUENTIAL, aspect="auto", text_auto=".0%",
                    labels=dict(color="DNF rate", x="Season", y=""))
    fig.update_layout(coloraxis_colorbar=dict(tickformat=".0%"))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    worst_team = rate.mean(axis=1).idxmax()
    stacked = rate.stack()
    (cell_team, cell_season) = stacked.idxmax()
    text = (f"{theme.team_label(worst_team)} has the highest average retirement rate "
            f"({rate.mean(axis=1).max():.0%}). The single worst team-season is {theme.team_label(cell_team)} in "
            f"{cell_season} at {stacked.max():.0%} of entries.")
    return EDAResult(theme.finish(fig, "Retirement rate by team and season", height=460, legend=False), text,
                     rate.reset_index())
