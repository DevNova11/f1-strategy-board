"""EDA step 6 - automated report generation.

``generate_report`` runs every EDA module on the current tables and writes

* ``reports/eda_report.html`` - a self-contained, interactive report (Plotly
  is loaded from a CDN, everything else is inline);
* ``reports/eda_findings.md`` - the same findings as plain text;
* ``reports/eda_findings.json`` - machine-readable findings.

``profile_report`` additionally writes a ydata-profiling report when that
optional package is installed.

Run ``python -m src.eda.report`` from the project root.
"""

from __future__ import annotations

import html
import json
import re
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import config, feedback, insights
from src.eda import EDAResult, bivariate, factor_columns, outliers, reduction, timeseries, univariate

log = logging.getLogger("f1sb.report")

ANOMALY_COLUMNS: list[str] = ["spend_total_musd", "pit_median_s", "customs_delay_h", "late_parts", "cfd_items_raw",
                              "crew_fatigue_idx"]

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root {{ --bg:{config.BG}; --panel:{config.PANEL}; --border:{config.BORDER}; --text:{config.TEXT};
        --muted:{config.MUTED}; --red:{config.RED}; --cyan:{config.CYAN}; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font-family:'IBM Plex Sans','Segoe UI',Arial,sans-serif;
       line-height:1.55; }}
main {{ max-width:1180px; margin:0 auto; padding:0 20px 80px; }}
header {{ padding:36px 0 18px; border-bottom:1px solid var(--border); position:relative; }}
header::after {{ content:""; position:absolute; left:0; bottom:-1px; width:56px; height:3px; background:var(--red); }}
h1,h2,h3 {{ font-family:'Titillium Web','Segoe UI',sans-serif;
          margin:0; }}
h1 {{ font-size:38px; line-height:1.15; }}
h2 {{ font-size:26px; margin:48px 0 6px; padding-top:14px; border-top:1px solid var(--border); }}
h2 span {{ color:var(--muted); margin-right:10px; font-weight:600; }}
.sub {{ color:var(--muted); margin:8px 0 0; }}
.card {{ background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:14px 18px 6px;
        margin:18px 0; }}
.takeaway {{ padding:2px 4px; margin:4px 0 18px; color:var(--muted); }}
.banner {{ background:rgba(0,229,255,.07); border:1px solid rgba(0,229,255,.4); border-radius:12px;
          padding:12px 16px; margin:22px 0; }}
.banner b {{ color:var(--cyan); }}
table {{ border-collapse:collapse; width:100%; font-size:14px; margin:8px 0 16px; }}
th {{ text-align:left; color:var(--muted); font-weight:500; font-size:13px;
     border-bottom:1px solid var(--border); padding:8px 10px; }}
td {{ border-bottom:1px solid #202027; padding:7px 10px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; margin:24px 0; }}
.kpi {{ background:var(--panel); border:1px solid var(--border);
       border-radius:12px; padding:14px 16px; }}
.kpi b {{ display:block; font-family:'Titillium Web',sans-serif; font-size:30px; line-height:1.1; }}
.kpi span {{ color:var(--muted); font-size:13px; }}
ol.findings li {{ margin:8px 0; }}
footer {{ color:var(--muted); font-size:13px; margin-top:48px; border-top:1px solid var(--border); padding-top:14px; }}
"""


def _note(text: str) -> str:
    """Chart note with the finding in bold, as in the dashboard."""
    text = re.sub(r"\b(REAL|SYNTHETIC)\b", lambda m: m.group(0).lower(), text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip(), maxsplit=1)
    rest = f" {html.escape(parts[1])}" if len(parts) > 1 else ""
    return f"<strong style='color:#ECECEF;font-weight:500'>{html.escape(parts[0])}</strong>{rest}"


def _table_html(frame: pd.DataFrame, max_rows: int = 12) -> str:
    return frame.head(max_rows).to_html(index=False, border=0, float_format=lambda v: f"{v:,.3f}", na_rep="-")


def build_sections(tables, watchlist: pd.DataFrame | None = None) -> list[tuple[str, str, list[EDAResult]]]:
    """Run all EDA modules and return ``(title, intro, results)`` per section."""
    tr, ts, dr = tables.team_race, tables.team_season, tables.driver_race
    factors = factor_columns(tr)
    scored = outliers.detect_anomalies(tr, [c for c in ANOMALY_COLUMNS if c in tr], by=["season"])
    scored = outliers.annotate_watchlist(scored, watchlist if watchlist is not None else feedback.load_watchlist())
    pca = reduction.run_pca(ts, reduction.PROFILE_FEATURES)
    clusters = reduction.kmeans_profiles(ts, reduction.PROFILE_FEATURES)
    coords, method = reduction.embed(ts, reduction.PROFILE_FEATURES, "tsne")
    ts_clustered = ts.assign(cluster=clusters.labels)
    latest = int(tr["season"].max())
    return [
        ("Univariate profiling", "What each variable looks like on its own: centre, spread, skew and tails.", [
            univariate.lap_time_distribution(tables.laps), univariate.spend_distribution(tables.component),
            univariate.pit_delta_distribution(tables.pitstops), univariate.wear_distribution(tables.stints),
            univariate.shape_overview(tr, factors)]),
        ("Bivariate and multivariate analysis", "How variables move together, within and across domains.", [
            bivariate.correlation_heatmap(tr, factors + ["points", "dnf"]),
            bivariate.spend_vs_reliability(ts), bivariate.weather_vs_pace(tables.races, tr),
            bivariate.wet_vs_dry(tables.races),
            bivariate.pair_plot(ts, ["spend_total_musd", "sim_hours", "pit_delta_s", "dnf_rate", "total_points"])]),
        ("Outliers and anomalies", "Three detectors combined by vote, plus the sport's own anomalies: retirements.", [
            outliers.detector_agreement(scored), outliers.anomaly_scatter(scored, "customs_delay_h", "pit_median_s"),
            outliers.dnf_by_cause(dr), outliers.failures_by_component(dr), outliers.dnf_heatmap(dr)]),
        ("Time series and trends", "Season-long spend against performance, and tyre degradation lap by lap.", [
            timeseries.multi_season_trend(ts, "spend_total_musd"), timeseries.multi_season_trend(ts, "total_points"),
            timeseries.spend_vs_performance(tr, latest), timeseries.spend_points_lag(ts),
            timeseries.tyre_degradation_curves(tables.laps), timeseries.degradation_by_season(tables.stints),
            timeseries.pit_trend(tables.pitstops)]),
        ("Dimensionality reduction and clustering", "Compressing the operational variables and grouping team-seasons.", [
            reduction.pca_scree(pca), reduction.pca_biplot(pca, ts_clustered, color="cluster", hover="constructor_name"),
            reduction.embedding_plot(ts_clustered, coords, method, color="team", hover="constructor_name"),
            reduction.silhouette_plot(clusters), reduction.cluster_heatmap(clusters)]),
    ]


def generate_report(tables, out_dir: Path | None = None) -> Path:
    """Write the HTML, Markdown and JSON reports and return the HTML path."""
    out_dir = Path(out_dir or config.REPORTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    watchlist = feedback.load_watchlist()
    sections = build_sections(tables, watchlist)
    rankings = insights.build_all(tables.team_race, tables.team_season)
    recovery = insights.recovery_check(tables.team_race)
    meta = tables.meta
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    body: list[str] = []
    findings: list[dict] = []
    first_fig = True
    for number, (title, intro, results) in enumerate(sections, start=1):
        body.append(f"<h2><span>{number:02d}</span>{html.escape(title)}</h2><p class='sub'>{html.escape(intro)}</p>")
        for result in results:
            fig_html = result.fig.to_html(full_html=False, include_plotlyjs="cdn" if first_fig else False,
                                          config={"displaylogo": False})
            first_fig = False
            body.append(f"<div class='card'>{fig_html}</div><p class='takeaway'>{_note(result.takeaway)}</p>")
            findings.append({"section": title, "chart": result.fig.layout.title.text or "", "finding": result.takeaway})

    body.append("<h2><span>06</span>Ranked factors</h2><p class='sub'>Consensus of four methods "
                "(Spearman, partial correlation, regression, boosting). SYNTHETIC factors rank where the "
                "generator put them.</p>")
    for target, ranking in rankings.items():
        name = insights.TARGETS[target]["label"]
        cols = ["consensus_rank", "label", "domain", "origin", "spearman", "partial", "score"]
        body.append(f"<h3>{html.escape(name)}</h3>" + _table_html(ranking[cols], 10))
        for sentence in insights.insight_statements(ranking, name, 3):
            findings.append({"section": "Ranked factors", "chart": name, "finding": sentence.replace("**", "")})
    body.append("<h3>Planted-effect recovery check</h3>" + _table_html(
        recovery[["factor", "target", "planted", "observed_r", "recovered"]], 20))

    body.append("<h2><span>07</span>Watchlist (feedback loop)</h2>")
    if watchlist.empty:
        body.append("<p class='sub'>No rows have been sent to the watchlist from the dashboard yet.</p>")
    else:
        body.append(f"<p class='sub'>{len(watchlist)} rows flagged in the dashboard are ringed on the anomaly chart "
                    f"above.</p>" + _table_html(watchlist[["key", "reason", "votes", "note", "added_at"]], 40))

    kpis = [("Seasons", f"{min(meta['seasons'])}-{max(meta['seasons'])}"), ("Races", meta["races"]),
            ("Races with lap data", meta["races_with_laps"]), ("Driver entries", f"{len(tables.driver_race):,}"),
            ("Laps analysed", f"{len(tables.laps):,}"), ("Data", "Full" if meta["mode"] == "raw" else "Sample")]
    kpi_html = "".join(f"<div class='kpi'><b>{html.escape(str(v))}</b><span>{html.escape(k)}</span></div>" for k, v in kpis)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>F1 Strategy Board - EDA report</title>
<style>{CSS}</style></head><body><main>
<header><h1>F1 Strategy Board: EDA report</h1>
<p class="sub">Mapping off-track operations to on-track performance. Generated {stamp}.</p></header>
<div class="banner"><b>Data provenance.</b> Results, laps, pit stops, tyre stints, standings and weather are REAL
(Jolpica and FastF1). Financial, logistics and simulation / CFD data are <b>SYNTHETIC</b>, produced by a seeded
generator (seed {meta['seed']}) that plants loose relationships with the real results. Findings that involve them
demonstrate the method and are not evidence about real teams.</div>
<div class="kpis">{kpi_html}</div>
{''.join(body)}
<footer>F1 Strategy Board. Unofficial project, not associated with Formula 1 or any team.</footer>
</main></body></html>"""
    html_path = out_dir / "eda_report.html"
    html_path.write_text(page, encoding="utf-8")

    lines = ["# F1 Strategy Board - EDA findings", "", f"Generated {stamp}. Data mode: {meta['mode']}.", "",
             "> Financial, logistics and simulation / CFD data are SYNTHETIC.", ""]
    current = None
    for item in findings:
        if item["section"] != current:
            current = item["section"]
            lines += ["", f"## {current}", ""]
        lines.append(f"- **{item['chart'].title()}**: {item['finding']}")
    (out_dir / "eda_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "eda_findings.json").write_text(json.dumps(
        {"generated": stamp, "mode": meta["mode"], "findings": findings}, indent=2), encoding="utf-8")
    return html_path


def profile_report(tables, out_dir: Path | None = None) -> Path | None:
    """Write a ydata-profiling report for the team x race table (optional dependency)."""
    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        log.info("ydata-profiling is not installed; skipping (pip install -r requirements-optional.txt).")
        return None
    out_dir = Path(out_dir or config.REPORTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["season", "round", "team", "points", "dnf"] + factor_columns(tables.team_race)
    path = out_dir / "profile_team_race.html"
    ProfileReport(tables.team_race[cols], title="Team x race profile (includes SYNTHETIC columns)",
                  minimal=True).to_file(path)
    return path


def main() -> None:
    from src import pipeline

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tables = pipeline.load()
    log.info("Report written to %s", generate_report(tables))
    extra = profile_report(tables)
    if extra:
        log.info("Profile written to %s", extra)


if __name__ == "__main__":
    main()
