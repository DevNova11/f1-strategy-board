"""EDA step 5 - dimensionality reduction and clustering.

Compress many operational variables into a few axes (PCA), look for structure a
straight line would miss (t-SNE, or UMAP when installed), and group team-seasons
into operating profiles with k-means, choosing k by silhouette score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src import config, theme
from src.eda import EDAResult, label, plain

TEAM_COLOUR_MAP = {theme.team_label(k): v for k, v in config.TEAM_COLOR.items()}

# Default feature set describing how a team operates over a season.
PROFILE_FEATURES: list[str] = [
    "spend_total_musd", "aero_share", "late_parts", "spares_coverage_pct", "crew_fatigue_idx",
    "customs_delay_h", "wind_tunnel_runs", "sim_hours", "sim_track_corr_pct", "upgrade_count",
    "pit_delta_s", "dnf_rate", "mean_grid",
]


def prepare_matrix(frame: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Median-impute and standardise the chosen columns.

    Columns that are entirely missing or constant are dropped; the list of
    columns actually used is returned alongside the matrix.
    """
    data = frame[[c for c in cols if c in frame.columns]].apply(pd.to_numeric, errors="coerce")
    data = data.loc[:, data.notna().any() & (data.nunique() > 1)]
    data = data.fillna(data.median())
    return StandardScaler().fit_transform(data), list(data.columns)


@dataclass
class PCAOutput:
    """Scores per row, loadings per feature and variance explained per component."""

    scores: pd.DataFrame
    loadings: pd.DataFrame
    explained: np.ndarray
    features: list[str]


def run_pca(frame: pd.DataFrame, cols: list[str], n_components: int | None = None) -> PCAOutput:
    """Principal component analysis on standardised features."""
    matrix, used = prepare_matrix(frame, cols)
    k = min(n_components or len(used), len(used), len(frame))
    pca = PCA(n_components=k, random_state=config.SEED).fit(matrix)
    names = [f"PC{i + 1}" for i in range(k)]
    scores = pd.DataFrame(pca.transform(matrix), columns=names, index=frame.index)
    loadings = pd.DataFrame(pca.components_.T, index=used, columns=names)
    return PCAOutput(scores, loadings, pca.explained_variance_ratio_, used)


def pca_scree(out: PCAOutput) -> EDAResult:
    """Variance explained by each component, with the cumulative line."""
    share = out.explained * 100
    cumulative = np.cumsum(share)
    names = list(out.scores.columns)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=share, name="Each component", marker_color=config.RED,
                         hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=names, y=cumulative, name="Cumulative", mode="lines+markers",
                             line=dict(color=config.CYAN, width=2),
                             hovertemplate="%{x}: %{y:.1f}% cumulative<extra></extra>"))
    fig.update_layout(yaxis=dict(title="Variance explained (%)", range=[0, 105]), xaxis_title="")
    need = int(np.argmax(cumulative >= 80) + 1) if (cumulative >= 80).any() else len(names)
    text = (f"The first two components capture {cumulative[min(1, len(cumulative) - 1)]:.0f}% of the variance in "
            f"{len(out.features)} features, and {need} components are needed to pass 80%. The operational "
            f"variables are heavily redundant: a few underlying dimensions drive most of them.")
    return EDAResult(theme.finish(fig, "PCA: variance explained", height=380), text)


def pca_biplot(out: PCAOutput, frame: pd.DataFrame, color: str = "team", hover: str | None = None,
               top_loadings: int = 6) -> EDAResult:
    """Rows on the first two components with the strongest feature loadings as arrows."""
    data = frame.join(out.scores[["PC1", "PC2"]])
    fig = px.scatter(data, x="PC1", y="PC2", color=color if color in data else None,
                     color_discrete_map=TEAM_COLOUR_MAP if color == "team" else None,
                     hover_name=hover if hover in data else None, opacity=0.85,
                     labels={"PC1": f"PC1 ({out.explained[0]:.0%} of variance)",
                             "PC2": f"PC2 ({out.explained[1]:.0%} of variance)", "team": "Team", "cluster": "Profile"})
    strength = (out.loadings["PC1"] ** 2 + out.loadings["PC2"] ** 2).sort_values(ascending=False)
    scale = float(np.abs(out.scores[["PC1", "PC2"]].to_numpy()).max()) * 0.9
    for feature in strength.head(top_loadings).index:
        x, y = out.loadings.loc[feature, "PC1"] * scale, out.loadings.loc[feature, "PC2"] * scale
        fig.add_annotation(x=x, y=y, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=2, arrowwidth=1.5, arrowcolor=config.MUTED)
        fig.add_annotation(x=x * 1.08, y=y * 1.08, text=label(feature).split(" (")[0], showarrow=False,
                           font=dict(size=11, color=config.TEXT))
    pc1 = out.loadings["PC1"].abs().sort_values(ascending=False).head(3).index
    pc2 = out.loadings["PC2"].abs().sort_values(ascending=False).head(3).index
    text = (f"PC1 is driven mainly by {', '.join(plain(f) for f in pc1)}; "
            f"PC2 by {', '.join(plain(f) for f in pc2)}. Points that sit close together "
            f"operate alike; arrows show which way each feature pulls.")
    return EDAResult(theme.finish(fig, "PCA biplot", height=520), text, out.loadings.round(3).reset_index(names="feature"))


def embed(frame: pd.DataFrame, cols: list[str], method: str = "tsne", seed: int = config.SEED) -> tuple[pd.DataFrame, str]:
    """Non-linear 2-D embedding. Falls back to t-SNE when UMAP is not installed.

    Returns the coordinates and the name of the method actually used.
    """
    matrix, _ = prepare_matrix(frame, cols)
    used = method
    if method == "umap":
        try:
            import umap  # optional dependency

            neighbours = max(2, min(15, len(frame) - 1))
            coords = umap.UMAP(n_neighbors=neighbours, min_dist=0.3, random_state=seed).fit_transform(matrix)
        except Exception:  # not installed, or too few rows for the chosen settings
            used = "tsne"
    if used == "tsne":
        perplexity = float(max(2, min(30, (len(frame) - 1) // 3)))
        coords = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(matrix)
    return pd.DataFrame(coords, columns=["x", "y"], index=frame.index), used


def embedding_plot(frame: pd.DataFrame, coords: pd.DataFrame, method: str, color: str = "team",
                   hover: str | None = None) -> EDAResult:
    """Scatter of a t-SNE or UMAP embedding."""
    data = frame.join(coords)
    name = {"tsne": "t-SNE", "umap": "UMAP"}[method]
    fig = px.scatter(data, x="x", y="y", color=color if color in data else None,
                     color_discrete_map=TEAM_COLOUR_MAP if color == "team" else None,
                     hover_name=hover if hover in data else None,
                     labels={"x": f"{name} 1", "y": f"{name} 2", "team": "Team", "cluster": "Profile"})
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    text = (f"{name} places rows with similar operating numbers next to each other. Only neighbourhoods are "
            f"meaningful: the axes have no units and distances between far-apart groups should not be read. "
            f"Tight single-colour islands mean a team operates in a consistent, distinctive way.")
    return EDAResult(theme.finish(fig, f"{name} embedding", height=520), text)


@dataclass
class ClusterOutput:
    """K-means result: labels, the silhouette score for each k tried, and centroids."""

    labels: pd.Series
    scores: pd.DataFrame
    best_k: int
    centroids: pd.DataFrame
    features: list[str]


def kmeans_profiles(frame: pd.DataFrame, cols: list[str], k_values: range = range(2, 8),
                    k: int | None = None, seed: int = config.SEED) -> ClusterOutput:
    """Cluster rows into operating profiles; pick k by the best silhouette score."""
    matrix, used = prepare_matrix(frame, cols)
    rows = []
    fitted: dict[int, np.ndarray] = {}
    for n in k_values:
        if n >= len(frame):
            break
        labels = KMeans(n_clusters=n, n_init=10, random_state=seed).fit_predict(matrix)
        fitted[n] = labels
        rows.append({"k": n, "silhouette": silhouette_score(matrix, labels)})
    scores = pd.DataFrame(rows)
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    chosen = k if k in fitted else best_k
    labels = pd.Series([f"Profile {i + 1}" for i in fitted[chosen]], index=frame.index, name="cluster")
    centroids = pd.DataFrame(matrix, columns=used, index=frame.index).groupby(labels).mean()
    return ClusterOutput(labels, scores, best_k, centroids, used)


def silhouette_plot(out: ClusterOutput) -> EDAResult:
    """Silhouette score for each candidate number of clusters."""
    colours = [config.RED if k == out.best_k else "#4A4A55" for k in out.scores["k"]]
    fig = go.Figure(go.Bar(x=out.scores["k"], y=out.scores["silhouette"], marker_color=colours,
                           text=out.scores["silhouette"].round(3), textposition="outside", cliponaxis=False,
                           hovertemplate="k = %{x}<br>Silhouette %{y:.3f}<extra></extra>"))
    fig.update_layout(xaxis=dict(title="Number of clusters (k)", dtick=1), yaxis_title="Mean silhouette score")
    best = out.scores["silhouette"].max()
    quality = "well separated" if best > 0.5 else ("reasonably distinct" if best > 0.25 else "overlapping")
    text = (f"k = {out.best_k} gives the best silhouette score ({best:.3f}). On a scale where 1 is perfectly "
            f"separated and 0 is overlapping, the profiles are {quality}: treat them as a useful summary, "
            f"not as hard categories.")
    return EDAResult(theme.finish(fig, "Choosing k: silhouette scores", height=360, legend=False), text, out.scores)


def cluster_heatmap(out: ClusterOutput) -> EDAResult:
    """Cluster centroids in standard deviations from the overall mean."""
    centroids = out.centroids
    names = [label(c).split(" (")[0] for c in centroids.columns]
    limit = float(max(1.0, np.abs(centroids.to_numpy()).max()))
    fig = px.imshow(centroids.values, x=names, y=list(centroids.index), zmin=-limit, zmax=limit,
                    color_continuous_scale=config.DIVERGING, aspect="auto", text_auto=".1f",
                    labels=dict(color="Std devs from mean"))
    fig.update_xaxes(tickangle=-35, showgrid=False)
    fig.update_yaxes(showgrid=False)
    parts = []
    for name, row in centroids.iterrows():
        top = row.abs().sort_values(ascending=False).head(2).index
        traits = " and ".join(f"{'high' if row[t] > 0 else 'low'} {plain(t)}" for t in top)
        parts.append(f"{name}: {traits}")
    text = ("; ".join(parts) + ". Red cells are above the average team-season and cyan cells below, "
            "measured in standard deviations.")
    return EDAResult(theme.finish(fig, "Team operating profiles", height=160 + 70 * len(centroids), legend=False),
                     text, centroids.round(2).reset_index())


def cluster_membership(frame: pd.DataFrame, out: ClusterOutput) -> pd.DataFrame:
    """Which team-seasons fall in each profile, with their average outcome."""
    data = frame.assign(cluster=out.labels)
    agg = {"members": ("constructor_name", "size")}
    if "total_points" in data:
        agg["avg_points"] = ("total_points", "mean")
    if "championship_position" in data:
        agg["avg_position"] = ("championship_position", "mean")
    summary = data.groupby("cluster").agg(**agg)
    summary["examples"] = data.groupby("cluster").apply(
        lambda g: ", ".join(f"{n} {s}" for n, s in zip(g["constructor_name"].head(4), g["season"].head(4))),
        include_groups=False)
    return summary.round(1).reset_index()
