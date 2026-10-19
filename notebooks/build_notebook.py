"""Build and execute ``01_eda_walkthrough.ipynb``.

The notebook is generated rather than hand-edited so that the findings written
in its markdown cells always match the data: each finding is the takeaway the
EDA engine computed for the chart directly above it. Re-run this script after
fetching new data.

    python notebooks/build_notebook.py            # build and execute
    python notebooks/build_notebook.py --no-run   # build only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import feedback, insights, pipeline  # noqa: E402
from src.eda import bivariate, factor_columns, outliers, reduction, timeseries, univariate  # noqa: E402

OUT = Path(__file__).with_name("01_eda_walkthrough.ipynb")

SETUP = '''import sys, warnings
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.io as pio
from src import config, feedback, insights, model, pipeline
from src.eda import bivariate, factor_columns, outliers, reduction, report, timeseries, univariate

pio.renderers.default = "plotly_mimetype+notebook_connected"
pd.set_option("display.max_columns", 40, "display.width", 200)

tables = pipeline.load()
tr, ts, dr = tables.team_race, tables.team_season, tables.driver_race
laps = tables.laps.sample(min(len(tables.laps), 25_000), random_state=config.SEED)   # keeps the notebook light
factors = factor_columns(tr)
print(f"Data: {'full download' if tables.meta['mode'] == 'raw' else 'bundled sample'} | seasons {min(tables.meta['seasons'])}-{max(tables.meta['seasons'])} | "
      f"{tables.meta['races']} races, {tables.meta['races_with_laps']} with lap data")
pd.DataFrame({name: frame.shape for name, frame in tables.frames().items()}, index=["rows", "columns"]).T'''

# (section number, heading, intro, [(caption, expression returning an EDAResult)])
STEPS: list[tuple[str, str, list[tuple[str, str]]]] = [
    ("1. Univariate profiling",
     "Start with one variable at a time. For each of the four headline variables (lap time, spend, pit delta, tyre "
     "wear) the questions are the same: where is the middle, how wide is the spread, is the distribution lopsided "
     "(skew) and how heavy are its tails (kurtosis)? The answers decide which statistics are safe to use later.",
     [("Lap time", "univariate.lap_time_distribution(laps)"),
      ("Spend by component (SYNTHETIC)", "univariate.spend_distribution(tables.component)"),
      ("Pit delta", "univariate.pit_delta_distribution(tables.pitstops)"),
      ("Tyre wear", "univariate.wear_distribution(tables.stints)"),
      ("Shape of every factor", "univariate.shape_overview(tr, factors)")]),
    ("2. Bivariate and multivariate analysis",
     "Now pairs and groups of variables. The heatmap gives the overview; the scatters test the two cross-domain "
     "questions in the brief, spend against reliability and weather against pace.",
     [("Correlation heatmap", "bivariate.correlation_heatmap(tr, factors + ['points', 'dnf'])"),
      ("Spend vs reliability", "bivariate.spend_vs_reliability(ts)"),
      ("Weather vs pace", "bivariate.weather_vs_pace(tables.races, tr)"),
      ("Wet vs dry races", "bivariate.wet_vs_dry(tables.races)"),
      ("Pair plot", "bivariate.pair_plot(ts, ['spend_total_musd', 'sim_hours', 'pit_delta_s', 'dnf_rate', 'total_points'])")]),
    ("3. Outlier and anomaly detection",
     "Three detectors with different blind spots (IQR fences, z-score, Isolation Forest) vote on every team-race "
     "row. Retirements and component failures, the sport's own anomalies, are summarised separately.",
     [("Detector agreement", "outliers.detector_agreement(scored)"),
      ("Anomaly map", "outliers.anomaly_scatter(scored, 'customs_delay_h', 'pit_median_s')"),
      ("Non-finishes by cause", "outliers.dnf_by_cause(dr)"),
      ("Failures by component", "outliers.failures_by_component(dr)"),
      ("Retirement rate by team and season", "outliers.dnf_heatmap(dr)")]),
    ("4. Time series and trends",
     "Two clocks matter: the season (round by round, year by year) for spend and results, and the stint (lap by "
     "lap of tyre age) for degradation.",
     [("Spend by season (SYNTHETIC)", "timeseries.multi_season_trend(ts, 'spend_total_musd')"),
      ("Points by season", "timeseries.multi_season_trend(ts, 'total_points')"),
      ("Spend vs points within a season", "timeseries.spend_vs_performance(tr, int(tr['season'].max()))"),
      ("Does spend lead or lag points?", "timeseries.spend_points_lag(ts)"),
      ("Tyre degradation curves", "timeseries.tyre_degradation_curves(tables.laps)"),
      ("Pit-stop performance", "timeseries.pit_trend(tables.pitstops)")]),
    ("5. Dimensionality reduction and clustering",
     "Thirteen operational variables per team-season are compressed with PCA, embedded with t-SNE, and grouped "
     "into operating profiles with k-means. The number of profiles is chosen by silhouette score.",
     [("PCA: variance explained", "reduction.pca_scree(pca)"),
      ("PCA biplot", "reduction.pca_biplot(pca, ts_clustered, color='cluster', hover='constructor_name')"),
      ("t-SNE embedding", "reduction.embedding_plot(ts_clustered, coords, method, color='team', hover='constructor_name')"),
      ("Choosing k", "reduction.silhouette_plot(clusters)"),
      ("Operating profiles", "reduction.cluster_heatmap(clusters)")]),
]

PREP = {
    "3. Outlier and anomaly detection":
        "anomaly_cols = ['spend_total_musd', 'pit_median_s', 'customs_delay_h', 'late_parts', 'cfd_items_raw', 'crew_fatigue_idx']\n"
        "scored = outliers.detect_anomalies(tr, anomaly_cols, by=['season'])\n"
        "scored = outliers.annotate_watchlist(scored, feedback.load_watchlist())   # feedback loop from the dashboard\n"
        "scored.loc[scored['anomaly'], ['team_race_key', 'race_name', 'anomaly_votes', 'anomaly_reason']]"
        ".sort_values('anomaly_votes', ascending=False).head(10)",
    "5. Dimensionality reduction and clustering":
        "pca = reduction.run_pca(ts, reduction.PROFILE_FEATURES)\n"
        "clusters = reduction.kmeans_profiles(ts, reduction.PROFILE_FEATURES)\n"
        "coords, method = reduction.embed(ts, reduction.PROFILE_FEATURES, 'tsne')\n"
        "ts_clustered = ts.assign(cluster=clusters.labels)\n"
        "reduction.cluster_membership(ts, clusters)",
}


def build() -> nbformat.NotebookNode:
    """Assemble the notebook, computing each finding from the current data."""
    tables = pipeline.load()
    space: dict = {
        "tables": tables, "tr": tables.team_race, "ts": tables.team_season, "dr": tables.driver_race,
        "laps": tables.laps, "factors": factor_columns(tables.team_race), "feedback": feedback,
        "univariate": univariate, "bivariate": bivariate, "outliers": outliers, "timeseries": timeseries,
        "reduction": reduction,
    }
    meta = tables.meta
    cells = [
        new_markdown_cell(
            "# F1 Strategy Board: EDA walkthrough\n\n"
            "**Mapping off-track operations to on-track performance.** This notebook walks through the six steps "
            "of the EDA engine in `src/eda/`, with the finding from each chart written out beneath it.\n\n"
            "> **Data provenance.** Results, laps, pit stops, tyre stints, standings and weather are **real** "
            "(Jolpica and FastF1). Financial, logistics and simulation / CFD data are **SYNTHETIC**: they come "
            "from the seeded generator in `src/synthetic.py`, which plants loose relationships with the real "
            "results. Findings that involve them show the method working; they are not evidence about real teams.\n\n"
            f"Built from **{meta['races']} races** ({min(meta['seasons'])}-{max(meta['seasons'])}), "
            f"**{meta['races_with_laps']}** with lap-level data, using the "
            f"**{'full download' if meta['mode'] == 'raw' else 'bundled sample'}**. "
            "The notebook is generated by `notebooks/build_notebook.py`, so the findings always match the data."),
        new_markdown_cell("## 0. Setup\n\nLoad the cleaned, joined tables built by layers 1 to 3 "
                          "(`src/sources.py`, `src/ingest.py`, `src/clean.py`)."),
        new_code_cell(SETUP),
    ]
    summary: list[str] = []
    for heading, intro, charts in STEPS:
        cells.append(new_markdown_cell(f"## {heading}\n\n{intro}"))
        if heading in PREP:
            exec(PREP[heading].rsplit("\n", 1)[0], space)        # noqa: S102 - our own literal code
            cells.append(new_code_cell(PREP[heading]))
        for caption, expression in charts:
            result = eval(expression, space)                     # noqa: S307 - our own literal code
            cells.append(new_markdown_cell(f"### {caption}"))
            cells.append(new_code_cell(f"result = {expression}\nresult.fig.show()"))
            cells.append(new_markdown_cell(f"**Finding.** {result.takeaway}"))
        summary.append(f"- **{heading.split('. ', 1)[1]}**: {result.takeaway}")

    rankings = insights.build_all(tables.team_race, tables.team_season)
    statements = insights.insight_statements(rankings["points"], "Points per race", 3)
    recovery = insights.recovery_check(tables.team_race)
    cells += [
        new_markdown_cell(
            "## 6. Automated EDA report\n\nEverything above is packaged by `src/eda/report.py` into a single "
            "interactive HTML file, plus Markdown and JSON versions of the findings. The same function backs the "
            "dashboard's *Report* tab."),
        new_code_cell("path = report.generate_report(tables)\nprint('Report written to', path.relative_to(ROOT))\n"
                      "print((config.REPORTS_DIR / 'eda_findings.md').read_text(encoding='utf-8')[:1200])"),
        new_markdown_cell(
            "### Where the EDA leads: ranked factors\n\nThe EDA feeds layers 5 and 6. Factors are ranked by the "
            "consensus of four methods (Spearman, partial correlation controlling for last season's strength, "
            "standardised regression and gradient-boosting permutation importance)."),
        new_code_cell("rankings = insights.build_all(tr, ts)\n"
                      "rankings['points'][['consensus_rank', 'label', 'domain', 'origin', 'spearman', 'partial', 'score']].head(10)"),
        new_markdown_cell("**Finding.** For points per race:\n\n" + "\n".join(f"- {s}" for s in statements)),
        new_code_cell("insights.recovery_check(tr)[['factor', 'target', 'planted', 'observed_r', 'recovered']]"),
        new_markdown_cell(
            f"**Finding.** {int(recovery['recovered'].sum())} of {len(recovery)} relationships planted in the "
            "synthetic data are recovered with the correct sign. Because the synthetic sources have a known "
            "ground truth, this is the appropriate test for them: it checks the engine, not the teams."),
        new_markdown_cell(
            "## Summary of findings\n\nThe last finding from each step:\n\n" + "\n".join(summary) + "\n\n"
            "### Caveats\n\n"
            "- Three of the five sources are synthetic, so every result that involves budget, logistics or "
            "simulation reflects the generator's design.\n"
            "- Everything here is association, not causation: points fund budgets as much as budgets buy points.\n"
            "- Retirement causes are not published from 2023, so component-level failure analysis covers 2019 to 2022.\n"
            "- Weather is shared by every team in a race, so it explains field-wide effects, not gaps between teams."),
    ]
    notebook = new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-run", action="store_true", help="write the notebook without executing it")
    args = parser.parse_args()
    notebook = build()
    if not args.no_run:
        from nbclient import NotebookClient

        NotebookClient(notebook, timeout=900, kernel_name="python3",
                       resources={"metadata": {"path": str(OUT.parent)}}).execute()
    nbformat.write(notebook, OUT)
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB, {len(notebook.cells)} cells)")


if __name__ == "__main__":
    main()
