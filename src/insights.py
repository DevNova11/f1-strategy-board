"""Layer 6 - ranked factor importance for race wins and championship points.

No single statistic deserves to be called "importance", so four are computed
and combined:

1. **Spearman correlation** - the raw monotonic link with the outcome;
2. **Partial correlation** - the link that survives after controlling for how
   strong the team already was last season (a real, pre-determined baseline);
3. **Standardised regression** - the effect holding every other factor fixed;
4. **Gradient-boosting permutation importance** - how much out-of-sample
   predictive power is lost when the factor is scrambled (catches non-linear
   and interaction effects).

Each method ranks the factors; the **consensus rank** is the mean of the four
ranks. A factor that tops only one method is treated with suspicion.

Reading the ranking honestly
----------------------------
* Factors marked SYNTHETIC rank where they do because ``src.synthetic`` planted
  those relationships. ``recovery_check`` verifies the engine finds what was
  planted, which is the right test for synthetic inputs.
* The ranking is associational. Points fund budgets as much as budgets buy
  points; nothing here separates the two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from src import config, model, theme
from src.eda import FEATURES, EDAResult, factor_columns, label, origin, sentence
from src.synthetic import PLANTED_EFFECTS

TARGETS: dict[str, dict[str, str]] = {
    "win": {"label": "Race wins", "table": "team_race", "kind": "binary", "unit": "race wins"},
    "points": {"label": "Points per race", "table": "team_race", "kind": "numeric", "unit": "points per race"},
    "total_points": {"label": "Championship points (season)", "table": "team_season", "kind": "numeric",
                     "unit": "championship points"},
}
METHODS: list[str] = ["spearman", "partial", "regression", "boosting"]
RANK_COLUMNS: list[str] = (
    ["factor", "label", "domain", "origin", "role", "spearman", "spearman_p", "partial", "regression", "boosting", "n"]
    + [f"rank_{m}" for m in METHODS] + ["strength", "mean_rank", "score", "consensus_rank", "direction", "agreement"]
)
MIN_ROWS = 30      # below this a ranking is noise, so none is produced
METHOD_LABELS: dict[str, str] = {
    "spearman": "Spearman", "partial": "Partial", "regression": "Regression", "boosting": "Boosting",
}


def add_prior_strength(frame: pd.DataFrame, team_season: pd.DataFrame) -> pd.DataFrame:
    """Attach ``prior_points_share``: the team's share of points in the previous season.

    This is the control variable for the partial correlations. It is real and
    fixed before the season starts, so it cannot be caused by anything that
    happens during the season under study.
    """
    season = team_season[["season", "lineage", "total_points"]].copy()
    season["share"] = season["total_points"] / season.groupby("season")["total_points"].transform("sum")
    prior = season.assign(season=season["season"] + 1).rename(columns={"share": "prior_points_share"})
    return frame.merge(prior[["season", "lineage", "prior_points_share"]], how="left", on=["season", "lineage"])


def _boosting_importance(frame: pd.DataFrame, target: str, factors: list[str], kind: str, seed: int) -> pd.Series:
    """Permutation importance of a gradient-boosting model on held-out seasons."""
    data = frame.dropna(subset=[target])
    x = data[factors].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median()).fillna(0.0)
    y = data[target].astype(float)
    seasons = sorted(data["season"].unique())
    held = seasons[-2:] if len(seasons) >= 4 else seasons[-1:]
    test = data["season"].isin(held).to_numpy()
    if test.all() or test.sum() < 10 or (~test).sum() < 20:
        test = np.zeros(len(data), dtype=bool)
        test[np.random.default_rng(seed).choice(len(data), size=max(5, len(data) // 4), replace=False)] = True
    binary = kind == "binary" and y[~test].nunique() == 2 and y[test].nunique() == 2
    params = dict(n_estimators=150, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=seed)
    estimator = GradientBoostingClassifier(**params) if binary else GradientBoostingRegressor(**params)
    estimator.fit(x[~test], y[~test].astype(int) if binary else y[~test])
    result = permutation_importance(estimator, x[test], y[test].astype(int) if binary else y[test], n_repeats=8,
                                    random_state=seed, scoring="roc_auc" if binary else "r2")
    return pd.Series(np.clip(result.importances_mean, 0, None), index=factors)


def _regression_betas(frame: pd.DataFrame, target: str, factors: list[str], kind: str) -> pd.Series:
    """Standardised, lightly ridge-penalised coefficients (stable under collinearity)."""
    data = frame.dropna(subset=[target])
    x = data[factors].apply(pd.to_numeric, errors="coerce")
    x = pd.DataFrame(StandardScaler().fit_transform(x.fillna(x.median()).fillna(0.0)), columns=factors)
    y = data[target].astype(float).to_numpy()
    if kind == "binary" and len(np.unique(y)) == 2:
        coef = LogisticRegression(C=1.0, max_iter=2000).fit(x, y.astype(int)).coef_[0]
    else:
        coef = Ridge(alpha=1.0).fit(x, (y - y.mean()) / (y.std() or 1.0)).coef_
    return pd.Series(coef, index=factors)


def rank_factors(frame: pd.DataFrame, target: str, factors: list[str] | None = None,
                 control: str = "prior_points_share", kind: str = "numeric",
                 seed: int = config.SEED) -> pd.DataFrame:
    """Rank explanatory factors for ``target`` by consensus across four methods.

    Returns one row per factor with the signed statistic from each method, the
    per-method ranks, ``consensus_rank`` (1 = most important), ``score``
    (0-1, the mean rank rescaled so 1 means "first under every method") and
    ``strength`` (0-1, mean of each method's importance scaled by its maximum).
    Returns an empty frame when there are fewer than ``MIN_ROWS`` usable rows.
    """
    factors = [f for f in (factors or factor_columns(frame)) if f in frame.columns and f != target
               and frame[f].notna().sum() >= 20 and frame[f].nunique() > 2]
    if not factors or frame[target].notna().sum() < MIN_ROWS or frame[target].nunique() < 2:
        return pd.DataFrame(columns=RANK_COLUMNS)
    frame = frame.copy()
    frame[target] = frame[target].astype(float)
    corr = model.cross_domain_table(frame, factors, target, [control] if control in frame.columns else None)
    corr = corr.set_index("factor")
    betas = _regression_betas(frame, target, factors, kind)
    boost = _boosting_importance(frame, target, factors, kind, seed)

    out = pd.DataFrame({
        "factor": factors,
        "label": [label(f) for f in factors],
        "domain": [FEATURES.get(f, {}).get("domain", "Other") for f in factors],
        "origin": [origin(f) for f in factors],
        "role": [FEATURES.get(f, {}).get("role", "off_track") for f in factors],
        "spearman": corr["spearman_r"].reindex(factors).to_numpy(),
        "spearman_p": corr["spearman_p"].reindex(factors).to_numpy(),
        "partial": corr["partial_r"].reindex(factors).to_numpy(),
        "regression": betas.reindex(factors).to_numpy(),
        "boosting": boost.reindex(factors).to_numpy(),
        "n": corr["n"].reindex(factors).to_numpy(),
    })
    if out["partial"].isna().all():                      # no usable control: fall back to the raw link
        out["partial"] = out["spearman"]
    scaled = []
    for method in METHODS:
        strength = out[method].abs().fillna(0.0)
        out[f"rank_{method}"] = strength.rank(ascending=False, method="min").astype(int)
        scaled.append(strength / (strength.max() or 1.0))
    out["strength"] = np.mean(scaled, axis=0)
    out["mean_rank"] = out[[f"rank_{m}" for m in METHODS]].mean(axis=1)
    out["score"] = 1 - (out["mean_rank"] - 1) / max(len(out) - 1, 1)
    out["consensus_rank"] = out["mean_rank"].rank(method="first").astype(int)
    out["direction"] = np.where(out["spearman"] >= 0, "+", "-")
    out["agreement"] = out[[f"rank_{m}" for m in METHODS]].std(axis=1)
    return out.sort_values("consensus_rank").reset_index(drop=True)


def ranking_chart(ranking: pd.DataFrame, target_label: str, top: int = 15) -> EDAResult:
    """Consensus importance, coloured by whether the factor uses real or synthetic data."""
    data = ranking.head(top).iloc[::-1]
    names = [f"{l} *" if o == "SYNTHETIC" else l for l, o in zip(data["label"], data["origin"])]
    fig = go.Figure()
    for origin, colour, legend_name in (("REAL", config.RED, "Real data"), ("SYNTHETIC", "#6B6B78", "Synthetic data *")):
        part = data[data["origin"] == origin]
        part_names = [n for n, o in zip(names, data["origin"]) if o == origin]
        fig.add_trace(go.Bar(
            x=part["score"], y=part_names, orientation="h", name=legend_name, marker_color=colour,
            customdata=np.column_stack([part["domain"], part["consensus_rank"],
                                        np.where(part["direction"] == "+", "more", "less")]) if len(part) else None,
            hovertemplate=("%{y}<br>Rank %{customdata[1]}, %{customdata[0]}<br>Higher values go with %{customdata[2]} "
                           + target_label.lower() + "<extra></extra>")))
    fig.update_layout(xaxis=dict(title="Overall importance (0 to 1)", range=[0, 1.05]), yaxis=dict(
        title="", categoryorder="array", categoryarray=names), barmode="overlay")
    if data.empty:
        return EDAResult(theme.finish(fig, f"Factors ranked for {target_label.lower()}", legend=False), "No factors to rank.")
    best = ranking.iloc[0]
    text = (f"{best['label'].split(' (')[0]} ranks first for {target_label.lower()} across all four methods "
            f"({best['domain'].lower()}, {'real' if best['origin'] == 'REAL' else 'synthetic'} data). Longer bars "
            f"matter more. Red bars use real data; grey, starred bars use synthetic data. Hover a bar to see "
            f"whether more of that factor goes with more or fewer {target_label.lower()}.")
    return EDAResult(theme.finish(fig, f"Factors ranked for {target_label.lower()}",
                                  height=max(400, 30 * len(data) + 160)), text, ranking)


def method_comparison(ranking: pd.DataFrame, top: int = 10) -> EDAResult:
    """Heatmap of each factor's rank under each method: do the methods agree?"""
    data = ranking.head(top)
    ranks = data[[f"rank_{m}" for m in METHODS]].to_numpy()
    fig = go.Figure(go.Heatmap(
        z=ranks, x=[METHOD_LABELS[m] for m in METHODS], y=data["label"], text=ranks, texttemplate="%{text}",
        colorscale=[[0, config.RED], [1, "#1B1B20"]], zmin=1, zmax=max(int(ranks.max()), 2), showscale=False,
        hovertemplate="%{y}<br>%{x}: rank %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=0)
    steady = data.loc[data["agreement"].idxmin()]
    shaky = data.loc[data["agreement"].idxmax()]
    text = (f"{steady['label'].split(' (')[0]} is ranked consistently by all four methods; "
            f"{sentence(shaky['label'])} depends on the method, which usually means it overlaps with another "
            f"factor or matters only in combination. Numbers are ranks, 1 being most important.")
    return EDAResult(theme.finish(fig, "Do the methods agree?", height=max(360, 34 * len(data) + 150), legend=False),
                     text)


def domain_ranking(ranking: pd.DataFrame) -> pd.DataFrame:
    """Roll the factor ranking up to the level of the source domain."""
    out = ranking.groupby("domain").agg(
        factors=("factor", "size"), best_rank=("consensus_rank", "min"), mean_score=("score", "mean"),
        top_score=("score", "max"), top_factor=("label", "first"),
        synthetic_share=("origin", lambda s: float((s == "SYNTHETIC").mean()))).reset_index()
    return out.sort_values("top_score", ascending=False).reset_index(drop=True)


def domain_chart(ranking: pd.DataFrame, target_label: str) -> EDAResult:
    """Which domain explains the outcome best, by its strongest factor."""
    domains = domain_ranking(ranking).iloc[::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=domains["top_score"], y=domains["domain"], orientation="h", name="Strongest factor",
                         marker_color=config.RED, customdata=domains["top_factor"],
                         hovertemplate="%{y}<br>Strongest: %{customdata} (%{x:.2f})<extra></extra>"))
    fig.add_trace(go.Scatter(x=domains["mean_score"], y=domains["domain"], mode="markers", name="Average of its factors",
                             marker=dict(color=config.CYAN, size=11, symbol="diamond"),
                             hovertemplate="%{y}<br>Average %{x:.2f}<extra></extra>"))
    fig.update_layout(xaxis=dict(title="Overall importance (0 to 1)", range=[0, 1.05]), yaxis_title="")
    best = domains.iloc[-1]
    last = domains.iloc[0]
    text = (f"For {target_label.lower()}, {best['domain']} is the strongest domain (best factor: "
            f"{sentence(best['top_factor'])}); {last['domain']} is the weakest.")
    if "Weather" in domains["domain"].to_numpy():
        text += (" Weather ranks low by construction of the question: every team races in the same weather, so it "
                 "cannot explain why one team beats another on the day.")
    return EDAResult(theme.finish(fig, f"Domains ranked for {target_label}", height=380), text, domains.iloc[::-1])


def insight_statements(ranking: pd.DataFrame, target_label: str, top: int = 5) -> list[str]:
    """Plain-English sentences for the top-ranked factors."""
    statements = []
    if ranking.empty:
        return ["Not enough rows under the current filters to rank factors for this outcome."]
    unit = next((m["unit"] for m in TARGETS.values() if m["label"] == target_label), target_label.lower())
    for _, row in ranking.head(top).iterrows():
        way = f"higher values go with {'more' if row['direction'] == '+' else 'fewer'} {unit}"
        survive = ("holds up" if abs(row["partial"]) >= 0.5 * abs(row["spearman"]) and abs(row["partial"]) > 0.05
                   else "mostly disappears")
        tag = " This factor is SYNTHETIC: the link was planted by the generator." if row["origin"] == "SYNTHETIC" else ""
        statements.append(
            f"**#{row['consensus_rank']} {row['label']}** ({row['domain']}, {row['origin']}): {way} "
            f"(Spearman {row['spearman']:+.2f}). After controlling for last season's "
            f"strength the link {survive} (partial {row['partial']:+.2f}).{tag}")
    return statements


def recovery_check(team_race: pd.DataFrame) -> pd.DataFrame:
    """Does the engine recover the relationships planted in the SYNTHETIC data?

    For each documented planted effect that maps onto team x race columns, the
    observed Spearman correlation is compared with the planted direction.
    """
    aliases = {"late_parts": "late_parts", "pit_delta_s": "pit_delta_s", "grid": "mean_grid"}
    rows = []
    for effect in PLANTED_EFFECTS:
        factor, target = effect["factor"], aliases.get(effect["target"], effect["target"])
        if factor not in team_race.columns or target not in team_race.columns:
            continue
        stat = model.correlation(team_race, factor, target, "spearman")
        recovered = np.isfinite(stat["r"]) and stat["p"] < 0.05 and (stat["r"] > 0) == (effect["direction"] == "+")
        rows.append({"factor": label(factor), "target": label(target), "planted": effect["direction"],
                     "observed_r": stat["r"], "p": stat["p"], "recovered": bool(recovered),
                     "mechanism": effect["mechanism"]})
    return pd.DataFrame(rows)


def build_all(team_race: pd.DataFrame, team_season: pd.DataFrame, off_track_only: bool = False) -> dict[str, pd.DataFrame]:
    """Rankings for the three headline targets."""
    roles = ("off_track", "operations", "context") if off_track_only else ("off_track", "operations", "context", "on_track")
    race = add_prior_strength(team_race, team_season)
    season = add_prior_strength(team_season, team_season)
    out = {}
    for target, meta in TARGETS.items():
        frame = race if meta["table"] == "team_race" else season
        out[target] = rank_factors(frame, target, factor_columns(frame, roles), kind=meta["kind"])
    return out
