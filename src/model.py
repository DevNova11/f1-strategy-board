"""Layer 5 - cross-domain correlation engine and predictive reliability scoring.

Part A: correlation engine
    Pearson and Spearman correlations with p-values, partial correlation
    (the link that is left after removing what control variables explain) and
    standardised regression with statsmodels.

Part B: reliability scoring
    A gradient-boosting classifier that scores each car's retirement risk from
    pre-race information, evaluated on seasons it never saw, and explained with
    SHAP values.

Synthetic inputs are flagged wherever results are tabulated. Relationships that
involve them were planted by ``src.synthetic`` and show that the method works;
they are not findings about real teams.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src import config, theme
from src.eda import FEATURES, EDAResult, describe_strength, label, origin, plain, sentence, short


# ============================================================ correlation engine
def correlation(frame: pd.DataFrame, x: str, y: str, method: str = "spearman") -> dict[str, float]:
    """Correlation between two columns with its p-value and sample size."""
    pair = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 5 or pair[x].nunique() < 2 or pair[y].nunique() < 2:
        return {"r": np.nan, "p": np.nan, "n": len(pair)}
    func = stats.spearmanr if method == "spearman" else stats.pearsonr
    r, p = func(pair[x], pair[y])
    return {"r": float(r), "p": float(p), "n": len(pair)}


def partial_correlation(frame: pd.DataFrame, x: str, y: str, covars: list[str],
                        method: str = "spearman") -> dict[str, float]:
    """Correlation between ``x`` and ``y`` after removing what ``covars`` explain.

    Both variables are regressed on the covariates and the residuals are
    correlated. With ``method="spearman"`` everything is rank-transformed
    first, which makes the result robust to skew and outliers.
    """
    covars = [c for c in covars if c not in (x, y) and c in frame.columns]
    data = frame[[x, y] + covars].apply(pd.to_numeric, errors="coerce").dropna()
    dof = len(data) - 2 - len(covars)
    if dof < 3 or data[x].nunique() < 2 or data[y].nunique() < 2:
        return {"r": np.nan, "p": np.nan, "n": len(data)}
    if method == "spearman":
        data = data.rank()
    design = np.column_stack([np.ones(len(data))] + [data[c].to_numpy(dtype=float) for c in covars])
    res_x = data[x].to_numpy(dtype=float) - design @ np.linalg.lstsq(design, data[x].to_numpy(dtype=float), rcond=None)[0]
    res_y = data[y].to_numpy(dtype=float) - design @ np.linalg.lstsq(design, data[y].to_numpy(dtype=float), rcond=None)[0]
    if res_x.std() < 1e-12 or res_y.std() < 1e-12:
        return {"r": np.nan, "p": np.nan, "n": len(data)}
    r = float(np.clip(np.corrcoef(res_x, res_y)[0, 1], -0.999999, 0.999999))
    t_stat = r * np.sqrt(dof / (1 - r ** 2))
    return {"r": r, "p": float(2 * stats.t.sf(abs(t_stat), dof)), "n": len(data)}


def cross_domain_table(frame: pd.DataFrame, factors: list[str], target: str,
                       covars: list[str] | None = None) -> pd.DataFrame:
    """Pearson, Spearman and partial correlation of every factor with a target."""
    rows = []
    for factor in factors:
        if factor not in frame.columns or factor == target:
            continue
        pearson = correlation(frame, factor, target, "pearson")
        spearman = correlation(frame, factor, target, "spearman")
        partial = partial_correlation(frame, factor, target, covars or [], "spearman") if covars else \
            {"r": np.nan, "p": np.nan}
        meta = FEATURES.get(factor, {})
        rows.append({
            "factor": factor, "label": label(factor), "domain": meta.get("domain", "Other"),
            "origin": origin(factor), "pearson_r": pearson["r"], "spearman_r": spearman["r"],
            "spearman_p": spearman["p"], "partial_r": partial["r"], "partial_p": partial["p"], "n": spearman["n"],
        })
    out = pd.DataFrame(rows)
    return out.reindex(out["spearman_r"].abs().sort_values(ascending=False).index).reset_index(drop=True) if len(out) else out


def cross_domain_chart(table: pd.DataFrame, target: str, controlled: bool = False) -> EDAResult:
    """Ranked bar chart of factor correlations with a target."""
    col = "partial_r" if controlled else "spearman_r"
    data = table.dropna(subset=[col]).copy()
    data = data.reindex(data[col].abs().sort_values(ascending=True).index)
    data["name"] = [f"{l} *" if o == "SYNTHETIC" else l for l, o in zip(data["label"], data["origin"])]
    fig = go.Figure(go.Bar(
        x=data[col], y=data["name"], orientation="h",
        marker_color=np.where(data[col] >= 0, config.RED, config.CYAN),
        customdata=data[["domain", "origin", "n"]],
        hovertemplate="%{y}<br>r = %{x:.2f}<br>%{customdata[0]} | %{customdata[1]} | n = %{customdata[2]}<extra></extra>"))
    fig.add_vline(x=0, line_color=theme.AXIS)
    kind = "Partial correlation" if controlled else "Spearman correlation"
    fig.update_layout(xaxis=dict(title=f"{kind} with {plain(target)}", range=[-1, 1]), yaxis_title="")
    if data.empty:
        return EDAResult(theme.finish(fig, f"What moves with {short(target)}?", legend=False), "No factors to rank.")
    top = data.iloc[-1]
    text = (f"{top['label'].split(' (')[0]} has the strongest link with {plain(target)} (r = {top[col]:.2f}, "
            f"{describe_strength(top[col])} relationship; {top['domain'].lower()}, {top['origin'].lower()} data). "
            f"Red bars rise with {plain(target)}, cyan bars fall, and a star marks synthetic data.")
    if controlled:
        text += " These are partial correlations: what is left after the control variables are accounted for."
    return EDAResult(theme.finish(fig, (f"{short(target)}: after allowing for last season" if controlled else f"What moves with {plain(target)}?"), height=max(380, 26 * len(data) + 140),
                                  legend=False), text, table)


@dataclass
class RegressionOutput:
    """Tidy regression result: coefficient table, fit statistics and the fitted model."""

    coefficients: pd.DataFrame
    r2: float
    adj_r2: float
    n: int
    kind: str
    model: object = field(repr=False, default=None)


def _design(frame: pd.DataFrame, target: str, factors: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    used = [f for f in factors if f in frame.columns and f != target]
    data = frame[used + [target]].apply(pd.to_numeric, errors="coerce").dropna()
    x = data[used]
    x = x.loc[:, x.nunique() > 1]
    return (x - x.mean()) / x.std(ddof=0), data[target]


def fit_ols(frame: pd.DataFrame, target: str, factors: list[str], cluster: str | None = None) -> RegressionOutput:
    """Standardised OLS: each beta is the change in the target, in standard
    deviations, for a one standard deviation rise in the factor, holding the
    other factors fixed. Standard errors are HC3-robust, or clustered when
    ``cluster`` names a grouping column (e.g. the team)."""
    x, y = _design(frame, target, factors)
    y_std = (y - y.mean()) / y.std(ddof=0)
    design = sm.add_constant(x)
    if cluster and cluster in frame.columns:
        groups = pd.factorize(frame.loc[x.index, cluster])[0]
        model = sm.OLS(y_std, design).fit(cov_type="cluster", cov_kwds={"groups": groups})
    else:
        model = sm.OLS(y_std, design).fit(cov_type="HC3")
    return RegressionOutput(_coef_table(model, x), float(model.rsquared), float(model.rsquared_adj), int(model.nobs),
                            "OLS", model)


def fit_logit(frame: pd.DataFrame, target: str, factors: list[str]) -> RegressionOutput:
    """Standardised logistic regression for a binary target such as a race win.

    A light L2 penalty keeps the fit stable when a factor separates the classes
    almost perfectly. ``r2`` holds McFadden's pseudo R-squared.
    """
    x, y = _design(frame, target, factors)
    design = sm.add_constant(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = sm.Logit(y.astype(float), design).fit(disp=0, maxiter=200)
            if not np.isfinite(model.bse).all():
                raise ValueError("unstable fit")
        except Exception:
            model = sm.Logit(y.astype(float), design).fit_regularized(alpha=1.0, L1_wt=0.0, disp=0)
    pseudo = float(getattr(model, "prsquared", np.nan))
    return RegressionOutput(_coef_table(model, x), pseudo, pseudo, int(model.nobs), "Logit", model)


def _coef_table(model: object, x: pd.DataFrame) -> pd.DataFrame:
    params = model.params.drop("const")
    conf = model.conf_int().drop("const") if hasattr(model, "conf_int") else None
    try:
        vif = [variance_inflation_factor(x.to_numpy(), i) for i in range(x.shape[1])] if x.shape[1] > 1 else [1.0]
    except Exception:
        vif = [np.nan] * x.shape[1]
    table = pd.DataFrame({
        "factor": params.index,
        "label": [label(c) for c in params.index],
        "domain": [FEATURES.get(c, {}).get("domain", "Other") for c in params.index],
        "origin": [origin(c) for c in params.index],
        "beta": params.to_numpy(),
        "ci_low": conf[0].to_numpy() if conf is not None else np.nan,
        "ci_high": conf[1].to_numpy() if conf is not None else np.nan,
        "p_value": model.pvalues.drop("const").to_numpy() if hasattr(model, "pvalues") else np.nan,
        "vif": vif,
    })
    return table.reindex(table["beta"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def coefficient_chart(out: RegressionOutput, target: str) -> EDAResult:
    """Standardised coefficients with 95% confidence intervals."""
    data = out.coefficients.iloc[::-1].copy()
    data["name"] = [f"{l} *" if o == "SYNTHETIC" else l for l, o in zip(data["label"], data["origin"])]
    solid = data["p_value"] < 0.05
    fig = go.Figure(go.Scatter(
        x=data["beta"], y=data["name"], mode="markers",
        marker=dict(size=10, color=np.where(data["beta"] >= 0, config.RED, config.CYAN),
                    symbol=np.where(solid, "circle", "circle-open"), line=dict(width=2)),
        error_x=dict(type="data", symmetric=False, array=data["ci_high"] - data["beta"],
                     arrayminus=data["beta"] - data["ci_low"], color=config.MUTED, thickness=1.2, width=0),
        customdata=data[["p_value", "vif"]],
        hovertemplate="%{y}<br>beta %{x:.3f}<br>p = %{customdata[0]:.3g}<br>VIF %{customdata[1]:.1f}<extra></extra>"))
    fig.add_vline(x=0, line_color=theme.AXIS)
    unit = "log-odds" if out.kind == "Logit" else "standard deviations"
    fig.update_layout(xaxis_title=f"Effect per +1 SD of the factor ({unit})", yaxis_title="")
    sig = out.coefficients[out.coefficients["p_value"] < 0.05]
    fit = (f"pseudo R-squared {out.r2:.2f}" if out.kind == "Logit" else f"R-squared {out.r2:.2f}")
    text = f"{out.kind} on {out.n} rows, {fit}. "
    if len(sig):
        top = sig.iloc[0]
        text += (f"Holding the others fixed, {sentence(top['label'])} has the largest clear effect "
                 f"(beta {top['beta']:+.2f}). Filled dots are significant at 5%; hollow dots are not. ")
    else:
        text += "No factor is individually significant once the others are held fixed. "
    high_vif = out.coefficients[out.coefficients["vif"] > 5]
    if len(high_vif):
        text += (f"{len(high_vif)} factors have VIF above 5, meaning they overlap heavily with others and their "
                 f"individual betas are unstable.")
    return EDAResult(theme.finish(fig, f"Regression: drivers of {short(target)}",
                                  height=max(380, 28 * len(data) + 140), legend=False), text, out.coefficients)


# ========================================================== reliability scoring
RELIABILITY_FEATURES: list[str] = [
    "grid_clean", "team_prior_dnf_rate", "driver_prior_dnf_rate", "season_progress",
    "spend_total_musd", "aero_share", "power_unit_share", "travel_km", "freight_tonnes", "customs_delay_h",
    "late_parts", "spares_coverage_pct", "crew_fatigue_idx", "setup_hours", "wind_tunnel_runs", "cfd_items",
    "sim_hours", "sim_track_corr_pct", "upgrade_count", "setup_iterations",
]
EXTRA_LABELS: dict[str, str] = {
    "grid_clean": "Grid slot", "team_prior_dnf_rate": "Team DNF rate, previous 6 races",
    "driver_prior_dnf_rate": "Driver DNF rate, previous 10 races", "season_progress": "Progress through season",
}


def feature_label(col: str) -> str:
    """Label for a model feature, marking SYNTHETIC inputs with an asterisk."""
    if col in EXTRA_LABELS:
        return EXTRA_LABELS[col]
    return f"{label(col)} *" if origin(col) == "SYNTHETIC" else label(col)


def add_history_features(driver_race: pd.DataFrame) -> pd.DataFrame:
    """Add pre-race history features. Every value uses earlier races only."""
    out = driver_race.sort_values(["season", "round"]).copy()
    out["dnf_num"] = out["dnf"].astype(float)
    team_rate = out.groupby(["lineage", "season", "round"], as_index=False)["dnf_num"].mean().sort_values(["season", "round"])
    team_rate["team_prior_dnf_rate"] = team_rate.groupby("lineage")["dnf_num"].transform(
        lambda s: s.shift(1).rolling(6, min_periods=2).mean())
    out = out.merge(team_rate[["lineage", "season", "round", "team_prior_dnf_rate"]], on=["lineage", "season", "round"],
                    how="left")
    out["driver_prior_dnf_rate"] = out.groupby("driver_id")["dnf_num"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    out["season_progress"] = out["round"] / out.groupby("season")["round"].transform("max")
    return out.drop(columns="dnf_num")


@dataclass
class ReliabilityModel:
    """A fitted risk model with its honest, out-of-sample evaluation."""

    features: list[str]
    metrics: dict[str, float]
    scored: pd.DataFrame
    importance: pd.DataFrame
    explainer: str
    shap_values: np.ndarray | None
    shap_features: pd.DataFrame | None
    calibration: pd.DataFrame
    train_seasons: list[int]
    test_seasons: list[int]
    estimator: object = field(repr=False, default=None)
    imputer: object = field(repr=False, default=None)


def train_reliability(driver_race: pd.DataFrame, target: str = "dnf", test_seasons: list[int] | None = None,
                      seed: int = config.SEED) -> ReliabilityModel:
    """Train the retirement-risk model on earlier seasons and test it on later ones.

    The split is by time, never random: a model that will be used on the next
    race must be judged on races that came after its training data.
    """
    data = add_history_features(driver_race)
    data = data[data["outcome"] != "other"].reset_index(drop=True)      # DNS / DSQ are not reliability events
    features = [f for f in RELIABILITY_FEATURES if f in data.columns and data[f].notna().any()]
    seasons = sorted(data["season"].unique())
    if test_seasons is None:
        test_seasons = seasons[-2:] if len(seasons) >= 4 else seasons[-1:]
    is_test = data["season"].isin(test_seasons).to_numpy()
    if is_test.all() or not is_test.any():                             # single season: hold out the last third
        cut = data["round"].quantile(0.67)
        is_test = (data["round"] > cut).to_numpy()
    y = data[target].astype(int).to_numpy()

    imputer = SimpleImputer(strategy="median").fit(data.loc[~is_test, features])
    x_all = pd.DataFrame(imputer.transform(data[features]), columns=features, index=data.index)
    estimator = GradientBoostingClassifier(n_estimators=150, learning_rate=0.04, max_depth=2, subsample=0.8,
                                           min_samples_leaf=20, random_state=seed)
    estimator.fit(x_all[~is_test], y[~is_test])
    risk = estimator.predict_proba(x_all)[:, 1]

    y_test, p_test = y[is_test], risk[is_test]
    base = float(y[~is_test].mean())
    both = len(np.unique(y_test)) == 2
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, p_test)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y_test, p_test)) if both else float("nan"),
        "brier": float(brier_score_loss(y_test, p_test)),
        "brier_baseline": float(brier_score_loss(y_test, np.full(len(y_test), base))),
        "base_rate_train": base,
        "base_rate_test": float(y_test.mean()),
        "n_train": int((~is_test).sum()),
        "n_test": int(is_test.sum()),
    }

    x_test = x_all[is_test]
    shap_values = None
    try:
        import shap

        values = shap.TreeExplainer(estimator).shap_values(x_test)
        shap_values = np.asarray(values[1] if isinstance(values, list) else values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]
        score, explainer = np.abs(shap_values).mean(axis=0), "SHAP (mean |value|, log-odds)"
    except Exception:  # SHAP unavailable: fall back to permutation importance
        perm = permutation_importance(estimator, x_test, y_test, n_repeats=10, random_state=seed, scoring="roc_auc")
        score, explainer = perm.importances_mean, "Permutation importance (drop in ROC AUC)"
    importance = pd.DataFrame({"feature": features, "label": [feature_label(f) for f in features],
                               "origin": [origin(f) if f in FEATURES else "REAL" for f in features],
                               "importance": score}).sort_values("importance", ascending=False).reset_index(drop=True)

    scored = data[["season", "round", "race_name", "driver_id", "driver_name", "constructor_id", "constructor_name",
                   "lineage", "team", "driver_race_key", "team_race_key", "grid_clean", "status", "outcome", target]].copy()
    scored["risk"] = risk
    scored["split"] = np.where(is_test, "test (unseen)", "train")
    scored["risk_band"] = pd.cut(scored["risk"], [0, base * 0.75, base * 1.5, 1.0], labels=["Low", "Elevated", "High"],
                                 include_lowest=True).astype(str)

    test = scored[is_test].copy()
    test["bin"] = pd.qcut(test["risk"], q=min(8, max(2, test["risk"].nunique())), duplicates="drop")
    calibration = test.groupby("bin", observed=True).agg(predicted=("risk", "mean"), observed=(target, "mean"),
                                                         n=(target, "size")).reset_index(drop=True)
    return ReliabilityModel(features, metrics, scored, importance, explainer, shap_values, x_test, calibration,
                            [int(s) for s in seasons if s not in test_seasons], [int(s) for s in test_seasons],
                            estimator, imputer)


def importance_chart(model: ReliabilityModel, top: int = 12) -> EDAResult:
    """Which inputs move the risk score most."""
    data = model.importance.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(x=data["importance"], y=data["label"], orientation="h",
                           marker_color=np.where(data["origin"] == "SYNTHETIC", "#6B6B78", config.RED),
                           customdata=data["origin"],
                           hovertemplate="%{y}<br>%{x:.4f}<br>%{customdata}<extra></extra>"))
    axis = "Average push on the risk score (SHAP)" if model.explainer.startswith("SHAP") else model.explainer
    fig.update_layout(xaxis_title=axis, yaxis_title="")
    best = model.importance.iloc[0]
    synthetic_share = model.importance.loc[model.importance["origin"] == "SYNTHETIC", "importance"].sum() / \
        max(model.importance["importance"].sum(), 1e-12)
    text = (f"{best['label'].replace(' *', '').split(' (')[0]} moves the risk score most. Synthetic inputs "
            f"(grey, starred) carry "
            f"{synthetic_share:.0%} of total importance; their signal was planted by the generator "
            f"(for example late parts at rounds with technical retirements), so it shows the model can find a "
            f"real pattern, not that these operations cause failures.")
    return EDAResult(theme.finish(fig, "What drives retirement risk", height=max(380, 30 * len(data) + 140),
                                  legend=False), text, model.importance)


def shap_dependence(model: ReliabilityModel, feature: str) -> EDAResult:
    """SHAP contribution of one feature against its value, one dot per test-season car."""
    if model.shap_values is None or feature not in model.features:
        return EDAResult(theme.finish(go.Figure(), "SHAP dependence"),
                         "SHAP values are not available, so permutation importance is shown instead.")
    idx = model.features.index(feature)
    data = pd.DataFrame({"value": model.shap_features[feature].to_numpy(), "shap": model.shap_values[:, idx]})
    fig = px.scatter(data, x="value", y="shap", opacity=0.6, color_discrete_sequence=[config.RED],
                     labels={"value": feature_label(feature), "shap": "Contribution to risk (log-odds)"})
    fig.add_hline(y=0, line_color=theme.AXIS, line_dash="dot")
    rho = stats.spearmanr(data["value"], data["shap"])[0] if data["value"].nunique() > 2 else 0.0
    way = "raise" if rho > 0 else "lower"
    text = (f"Higher values of {sentence(feature_label(feature))} tend to {way} predicted risk "
            f"(rank correlation {rho:+.2f}). Each dot is one car in the unseen test seasons; dots above zero push "
            f"the risk up, dots below pull it down.")
    return EDAResult(theme.finish(fig, f"SHAP dependence: {feature_label(feature)}", height=400, legend=False), text)


def shap_summary(model: ReliabilityModel, top: int = 10) -> EDAResult:
    """Beeswarm-style SHAP summary: spread and direction of each feature's effect."""
    if model.shap_values is None:
        return importance_chart(model, top)
    order = [model.features.index(f) for f in model.importance["feature"].head(top)][::-1]
    rng = np.random.default_rng(config.SEED)
    fig = go.Figure()
    for row, idx in enumerate(order):
        values = model.shap_features.iloc[:, idx].to_numpy(dtype=float)
        spread = np.ptp(values) or 1.0
        fig.add_trace(go.Scatter(
            x=model.shap_values[:, idx], y=row + rng.uniform(-0.3, 0.3, len(values)), mode="markers",
            marker=dict(size=4, color=(values - values.min()) / spread, colorscale=[[0, config.CYAN], [1, config.RED]],
                        opacity=0.6, line=dict(width=0)),
            hovertemplate=f"{feature_label(model.features[idx])}<br>SHAP %{{x:.3f}}<extra></extra>", showlegend=False))
    fig.add_vline(x=0, line_color=theme.AXIS)
    fig.update_layout(yaxis=dict(tickmode="array", tickvals=list(range(len(order))),
                                 ticktext=[feature_label(model.features[i]) for i in order], title=""),
                      xaxis_title="SHAP value (push on predicted risk)")
    text = ("Each dot is one car in the test seasons. Position shows whether that input pushed its risk up (right) "
            "or down (left); colour shows whether the input's value was low (cyan) or high (red). Red dots on the "
            "right mean high values raise risk.")
    return EDAResult(theme.finish(fig, "SHAP summary", height=max(400, 36 * len(order) + 140), legend=False), text)


def calibration_chart(model: ReliabilityModel) -> EDAResult:
    """Predicted risk against the retirement rate actually observed."""
    cal = model.calibration
    limit = float(max(cal["predicted"].max(), cal["observed"].max()) * 1.15) if len(cal) else 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, limit], y=[0, limit], mode="lines", name="Perfect calibration",
                             line=dict(color=config.MUTED, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=cal["predicted"], y=cal["observed"], mode="lines+markers", name="Model",
                             line=dict(color=config.RED), marker=dict(size=9), customdata=cal["n"],
                             hovertemplate="Predicted %{x:.1%}<br>Observed %{y:.1%}<br>%{customdata} cars<extra></extra>"))
    fig.update_layout(xaxis=dict(title="Predicted retirement risk", tickformat=".0%"),
                      yaxis=dict(title="Observed retirement rate", tickformat=".0%"))
    m = model.metrics
    better = m["brier"] < m["brier_baseline"]
    text = (f"On the unseen {', '.join(map(str, model.test_seasons))} seasons the model reaches ROC AUC "
            f"{m['roc_auc']:.2f} (0.5 is a coin flip) and a Brier score of {m['brier']:.4f} against "
            f"{m['brier_baseline']:.4f} for always predicting the base rate, so it is "
            f"{'better' if better else 'no better'} than the naive forecast. Retirements are mostly bad luck: "
            f"expect a modest ranking signal, not a crystal ball.")
    return EDAResult(theme.finish(fig, "Calibration on unseen seasons", height=400), text, cal)


def risk_by_team(model: ReliabilityModel, split: str = "test (unseen)") -> EDAResult:
    """Average predicted risk against the observed retirement rate, per team."""
    data = model.scored[model.scored["split"] == split] if split else model.scored
    agg = data.groupby("team").agg(predicted=("risk", "mean"), observed=("dnf", "mean"), entries=("risk", "size"))
    agg = agg.sort_values("predicted", ascending=True).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(y=agg["team"], x=agg["predicted"], orientation="h", name="Predicted risk",
                         marker_color=config.RED, hovertemplate="%{y}<br>Predicted %{x:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(y=agg["team"], x=agg["observed"], mode="markers", name="Observed retirement rate",
                             marker=dict(color=config.CYAN, size=11, symbol="diamond"),
                             hovertemplate="%{y}<br>Observed %{x:.1%}<extra></extra>"))
    fig.update_layout(xaxis=dict(title="Share of entries", tickformat=".0%"), yaxis_title="")
    if agg.empty:
        return EDAResult(theme.finish(fig, "Risk by team"), "No scored entries in the current selection.")
    rho = stats.spearmanr(agg["predicted"], agg["observed"])[0] if len(agg) > 3 else float("nan")
    text = (f"{agg['team'].iloc[-1].split(' /')[0]} carries the highest average predicted risk "
            f"({agg['predicted'].iloc[-1]:.1%}) and {agg['team'].iloc[0].split(' /')[0]} the lowest "
            f"({agg['predicted'].iloc[0]:.1%}). Team-level predicted and observed rates have a rank correlation of "
            f"{rho:+.2f}.")
    return EDAResult(theme.finish(fig, "Retirement risk by team", height=440), text, agg)
