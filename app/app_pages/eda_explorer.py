"""Explore the data: one tab per EDA step. Only the open tab is computed.

Each tab leads with the two or three charts that answer its question; the rest
sit in a "More charts" expander so the page reads top to bottom without noise.
"""

import streamlit as st

from app import ui
from src import config, feedback
from src.eda import FEATURES, bivariate, factor_columns, label, outliers, reduction, report, tagged, timeseries, univariate

v = ui.view()
ui.header("Explore the data", "Six ways of looking at the same races: one variable at a time, pairs of variables, "
                              "unusual rows, change over time, groups of similar teams, and a printable report.")
ui.require(v.team_race, "team-race rows")

tr, ts, dr = v.team_race, v.team_season, v.driver_race
factors = factor_columns(tr)
numeric_choices = factors + ["points", "dnf", "mech_dnf", "positions_gained"]


def nice(col: str) -> str:
    """Readable option label for a column."""
    return tagged(col) if col in FEATURES else label(col)


def question(text: str) -> None:
    st.caption(f":material/help: {text}")


t_uni, t_bi, t_out, t_time, t_red, t_rep = st.tabs(
    ["Distributions", "Relationships", "Outliers", "Trends", "Team groups", "Report"],
    key="eda_tab", on_change="rerun", bind="query-params")

# -------------------------------------------------------------- distributions
if t_uni.open:
    with t_uni:
        question("What does each measure look like on its own: what is typical, and how spread out is it?")
        ui.section("Lap times", "Each lap compared with the median clean lap of the same race.")
        ui.show(ui.memo("uni_lap", lambda: univariate.lap_time_distribution(v.laps)), "uni_lap", table=True)
        left, right = st.columns(2)
        with left:
            ui.section("Pit stops", "Time lost or gained against the median stop in the same race.")
            ui.show(ui.memo("uni_pitdelta", lambda: univariate.pit_delta_distribution(v.pitstops)), "uni_pitdelta",
                    table=True)
        with right:
            ui.section("Tyre wear", "Seconds lost per lap as a set of tyres ages.")
            ui.show(ui.memo("uni_wear", lambda: univariate.wear_distribution(v.stints)), "uni_wear", table=True)

        ui.section("Look at any measure")
        c1, c2 = st.columns([2, 1])
        col = c1.selectbox("Measure", numeric_choices, format_func=nice, key="uni_var")
        split = c2.segmented_control("Split by", ["None", "Season", "Team"], default="None", key="uni_split")
        by = {"Season": "season", "Team": "team"}.get(split)
        data = tr.assign(season=tr["season"].astype(str)) if by == "season" else tr
        ui.show(ui.memo("uni_any", lambda: univariate.distribution(data, col, by)), "uni_any")

        with st.expander("More charts: spend by component, pit-lane time by season, skew of every measure"):
            ui.show(ui.memo("uni_spend", lambda: univariate.spend_distribution(v.component)), "uni_spend", table=True)
            ui.show(ui.memo("uni_pit", lambda: univariate.pit_distribution(v.pitstops)), "uni_pit", table=True)
            ui.show(ui.memo("uni_shape", lambda: univariate.shape_overview(tr, factors)), "uni_shape", table=True)
            st.dataframe(univariate.profile(tr, numeric_choices).drop(columns="column"), hide_index=True,
                         column_config={"label": "Measure", "origin": "Data",
                                        "missing_pct": st.column_config.NumberColumn("Missing %", format="%.1f")})

# -------------------------------------------------------------- relationships
if t_bi.open:
    with t_bi:
        question("Which measures move together, and does money or weather line up with results?")
        ui.section("How the main measures move together", "One cell per pair. Hover a cell for the exact value.")
        c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
        method = c1.segmented_control("Method", ["Spearman", "Pearson"], default="Spearman", key="bi_method",
                                      help="Spearman compares rankings and copes with outliers; Pearson assumes a "
                                           "straight-line relationship.")
        show_all = c2.toggle("Show every measure (about 30)", key="bi_all")
        cols = (factors + ["points", "dnf"]) if show_all else [c for c in bivariate.KEY_VARIABLES if c in tr.columns]
        ui.show(ui.memo("bi_heat", lambda: bivariate.correlation_heatmap(tr, cols, (method or "Spearman").lower())),
                "bi_heat")

        ui.section("Compare any two measures", "Each dot is one team at one race. The dashed line is the trend.")
        c1, c2, c3 = st.columns([2, 2, 1])
        x = c1.selectbox("Across", numeric_choices, index=numeric_choices.index("spend_total_musd"),
                         format_func=nice, key="bi_x")
        y = c2.selectbox("Up", numeric_choices, index=numeric_choices.index("points"), format_func=nice, key="bi_y")
        colour = c3.segmented_control("Colour", ["Team", "None"], default="Team", key="bi_colour")
        ui.show(ui.memo("bi_scatter", lambda: bivariate.scatter_trend(
            tr, x, y, "team" if colour == "Team" else None, hover="team_race_key")), "bi_scatter")

        left, right = st.columns(2)
        with left:
            ui.section("Does spending buy reliability?", "One bubble per team per season.")
            ui.show(ui.memo("bi_spend_rel", lambda: bivariate.spend_vs_reliability(ts)), "bi_spend_rel")
        with right:
            ui.section("Do hotter tracks wear tyres faster?", "One dot per race.")
            ui.show(ui.memo("bi_weather", lambda: bivariate.weather_vs_pace(v.races, tr)), "bi_weather")

        with st.expander("More charts: wet versus dry races, pair plot, many-measure profile"):
            ui.show(ui.memo("bi_wet", lambda: bivariate.wet_vs_dry(v.races)), "bi_wet", table=True)
            season_choices = [c for c in ["spend_total_musd", "total_points", "dnf_rate", "pit_delta_s", "sim_hours",
                                          "spares_coverage_pct", "late_parts", "crew_fatigue_idx", "mean_grid",
                                          "pace_delta_pct", "wind_tunnel_runs", "aero_share"] if c in ts.columns]
            dims = st.multiselect("Pair plot measures (up to six)", season_choices, default=season_choices[:4],
                                  max_selections=6, format_func=nice, key="bi_pair")
            if len(dims) >= 2:
                ui.show(ui.memo("bi_pairplot", lambda: bivariate.pair_plot(ts, dims)), "bi_pairplot")
            ui.show(ui.memo("bi_parallel", lambda: bivariate.parallel_profile(
                ts, ["spend_total_musd", "sim_hours", "spares_coverage_pct", "pit_delta_s", "dnf_rate"])),
                "bi_parallel")

# ------------------------------------------------------------------- outliers
if t_out.open:
    with t_out:
        question("Which team-race rows look unusual, and which cars failed to finish?")
        candidates = [c for c in ["spend_total_musd", "pit_median_s", "customs_delay_h", "late_parts", "cfd_items_raw",
                                  "crew_fatigue_idx", "freight_tonnes", "sim_hours", "pace_delta_pct", "lap_std_s"]
                      if c in tr.columns]
        with st.expander("Detector settings", icon=":material/tune:"):
            c1, c2 = st.columns([3, 1])
            cols = c1.multiselect("Measures to check", candidates, default=candidates[:6], key="out_cols",
                                  format_func=nice)
            contamination = c2.slider("Share flagged by Isolation Forest", 0.01, 0.10, 0.03, 0.01, key="out_contam")
            st.caption("Three tests look for unusual rows: values far outside the middle half of the data (IQR), "
                       "values more than three standard deviations from the mean (z-score), and odd combinations "
                       "of values (Isolation Forest). A row counts as an anomaly when at least two agree.")
        if len(cols) < 2:
            st.info("Pick at least two measures in Detector settings.", icon=":material/info:")
            st.stop()
        watch = feedback.load_watchlist()
        scored = outliers.annotate_watchlist(ui.cached_anomalies(v.key, tuple(cols), contamination), watch)

        ui.section("Unusual rows", "Grey dots are normal; the more tests that flag a row, the warmer its colour.")
        c1, c2 = st.columns(2)
        ax = c1.selectbox("Across", cols, index=min(2, len(cols) - 1), key="out_x", format_func=nice)
        ay = c2.selectbox("Up", cols, index=min(1, len(cols) - 1), key="out_y", format_func=nice)
        left, right = st.columns([3, 2])
        with left:
            ui.show(outliers.anomaly_scatter(scored, ax, ay), "out_scatter")
        with right:
            ui.show(outliers.detector_agreement(scored), "out_agree", table=True)

        ui.section("Keep an eye on a row", "Tick rows below and add them to the watchlist.")
        flagged = scored[scored["anomaly_votes"] >= 1].sort_values(["anomaly_votes", "iso_score"], ascending=False)
        table = flagged[["race_name", "team", "season", "anomaly_votes", "anomaly_reason", "on_watchlist", "team_race_key"]]
        event = st.dataframe(
            table, hide_index=True, height=300, on_select="rerun", selection_mode="multi-row", key="out_table",
            column_config={
                "race_name": "Race", "team": "Team", "season": st.column_config.NumberColumn("Season", format="%d"),
                "anomaly_votes": st.column_config.NumberColumn("Tests flagging it", format="%d"),
                "anomaly_reason": st.column_config.TextColumn("Unusual in", width="medium"),
                "on_watchlist": st.column_config.CheckboxColumn("On watchlist"),
                "team_race_key": st.column_config.TextColumn("Row id", width="small"),
            },
        )
        picked = table.iloc[event.selection.rows] if event.selection.rows else table.iloc[0:0]
        c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment="bottom")
        note = c1.text_input("Note (optional)", placeholder="e.g. check the freight records for this round",
                             key="out_note")
        if c2.button(f"Add {len(picked)} to watchlist", type="primary", disabled=picked.empty, key="out_send",
                     icon=":material/flag:"):
            feedback.add_to_watchlist(flagged.loc[picked.index], note=note)
            st.toast(f"Added {len(picked)} rows to the watchlist.", icon=":material/flag:")
            st.rerun()
        if c3.button("Clear watchlist", disabled=watch.empty, key="out_clear", icon=":material/delete:"):
            feedback.remove_from_watchlist(watch["key"].tolist())
            st.rerun()
        if watch.empty:
            st.caption("The watchlist is empty. Rows on it are ringed in the chart above, listed in the report, and "
                       "can be hidden from every chart with the sidebar switch.")
        else:
            with st.expander(f"Watchlist ({len(watch)} rows)", expanded=True):
                st.dataframe(watch, hide_index=True)
                st.download_button("Download as CSV", watch.to_csv(index=False), "watchlist.csv", "text/csv",
                                   key="out_dl", icon=":material/download:")

        ui.section("Cars that did not finish", "Real retirements, grouped by cause.")
        left, right = st.columns(2)
        with left:
            ui.show(ui.memo("out_dnf", lambda: outliers.dnf_by_cause(dr)), "out_dnf", table=True)
        with right:
            ui.show(ui.memo("out_comp", lambda: outliers.failures_by_component(dr)), "out_comp", table=True)
        with st.expander("More charts: retirement rate by team and season, spread of each measure"):
            ui.show(ui.memo("out_heat", lambda: outliers.dnf_heatmap(dr)), "out_heat")
            box_col = st.selectbox("Measure", cols, format_func=nice, key="out_box_col")
            ui.show(outliers.outlier_box(scored, box_col, "season"), "out_box")

# --------------------------------------------------------------------- trends
if t_time.open:
    with t_time:
        question("How do results, money and tyre wear change through a season and from year to year?")
        seasons = sorted(tr["season"].unique(), reverse=True)
        ui.section("Through one season")
        c1, c2 = st.columns([1, 3])
        season = c1.selectbox("Season", seasons, key="ts_season")
        metric_names = {"Points": "cum_points", "Spend (synthetic)": "cum_spend_musd", "Recent form": "points_roll3"}
        metric = c2.segmented_control("Show", list(metric_names), default="Points", key="ts_metric",
                                      help="Points and spend are running totals; recent form is the average of the "
                                           "last three races.")
        ui.show(ui.memo("ts_progress", lambda: timeseries.season_progress(
            tr, season, metric_names[metric or "Points"])), "ts_progress")

        ui.section("Season by season", "Renamed teams (Toro Rosso to RB, Renault to Alpine...) stay on one line.")
        trend_names = {"Points": "total_points", "Championship position": "championship_position",
                       "Retirement rate": "dnf_rate", "Spend (synthetic)": "spend_total_musd",
                       "Points per USD M": "points_per_musd"}
        trend = st.segmented_control("Show", list(trend_names), default="Points", key="ts_trend")
        ui.show(ui.memo("ts_multi", lambda: timeseries.multi_season_trend(ts, trend_names[trend or "Points"])),
                "ts_multi")

        left, right = st.columns(2)
        with left:
            ui.section("Tyre wear over a stint", "Real lap data, corrected for fuel burn.")
            with_laps = v.races[v.races["race_key"].isin(v.laps["race_key"].unique())]
            names = {"All races": None} | {f"{r.season} {r.race_name}": r.race_key
                                           for r in with_laps.sort_values(["season", "round"], ascending=False).itertuples()}
            pick = st.selectbox("Race", list(names), key="ts_race")
            laps = v.laps if names[pick] is None else v.laps[v.laps["race_key"] == names[pick]]
            ui.show(ui.memo("ts_deg", lambda: timeseries.tyre_degradation_curves(
                laps, min_laps=20 if names[pick] is None else 4)), "ts_deg", table=True)
        with right:
            ui.section("Pit crews compared", "Over all the selected seasons.")
            ui.show(ui.memo("ts_pit", lambda: timeseries.pit_trend(v.pitstops)), "ts_pit", table=True)

        with st.expander("More charts: spend against points, lead or lag, links by season, wear by season"):
            ui.show(ui.memo("ts_spend_perf", lambda: timeseries.spend_vs_performance(tr, season)), "ts_spend_perf")
            ui.show(ui.memo("ts_lag", lambda: timeseries.spend_points_lag(ts)), "ts_lag", table=True)
            factor = st.selectbox("Measure to track against points", factors, format_func=tagged, key="ts_factor",
                                  index=factors.index("spend_total_musd"))
            ui.show(ui.memo("ts_roll", lambda: timeseries.rolling_relationship(tr, factor)), "ts_roll", table=True)
            ui.show(ui.memo("ts_deg_season", lambda: timeseries.degradation_by_season(v.stints)), "ts_deg_season")

# ---------------------------------------------------------------- team groups
if t_red.open:
    with t_red:
        question("Do teams fall into a few distinct ways of operating?")
        if len(ts) < 12:
            st.info("Grouping needs at least 12 team-seasons. Widen the season range or clear the team filter.",
                    icon=":material/info:")
            st.stop()
        available = [c for c in reduction.PROFILE_FEATURES if c in ts.columns]
        with st.expander("Grouping settings", icon=":material/tune:"):
            feats = st.multiselect("Measures used to describe each team-season", available, default=available,
                                   key="red_feats", format_func=nice)
            k_choice = st.segmented_control("Number of groups", ["Auto", "2", "3", "4", "5", "6"], default="Auto",
                                            key="red_k", help="Auto picks the number that separates the groups best "
                                                              "(highest silhouette score).")
        if len(feats) < 3:
            st.info("Pick at least three measures in Grouping settings.", icon=":material/info:")
            st.stop()
        clusters = ui.memo("red_clusters", lambda: reduction.kmeans_profiles(
            ts, feats, k=None if k_choice in (None, "Auto") else int(k_choice)))
        clustered = ts.assign(cluster=clusters.labels)
        pca = ui.memo("red_pca", lambda: reduction.run_pca(ts, feats))

        ui.section("What each group looks like", "Each row is a group; each column a measure.")
        ui.show(ui.memo("red_heat", lambda: reduction.cluster_heatmap(clusters)), "red_heat", table=True)
        st.dataframe(reduction.cluster_membership(ts, clusters), hide_index=True,
                     column_config={"cluster": "Group", "members": "Team-seasons", "avg_points": "Average points",
                                    "avg_position": "Average championship position",
                                    "examples": st.column_config.TextColumn("Examples", width="large")})

        ui.section("Team-seasons on one map", "Points close together operate alike.")
        colour = st.segmented_control("Colour by", ["Group", "Team"], default="Group", key="red_colour")
        ui.show(ui.memo("red_biplot", lambda: reduction.pca_biplot(
            pca, clustered, "cluster" if colour != "Team" else "team", hover="constructor_name")), "red_biplot",
            table=True)

        with st.expander("More charts: how much the map captures, choosing the number of groups, t-SNE and LLE maps"):
            ui.show(ui.memo("red_scree", lambda: reduction.pca_scree(pca)), "red_scree")
            ui.show(ui.memo("red_sil", lambda: reduction.silhouette_plot(clusters)), "red_sil", table=True)
            method = st.segmented_control("Map method", ["t-SNE", "LLE", "UMAP"], default="t-SNE", key="red_method")
            coords, used = ui.cached_embedding(v.key, tuple(feats), {"LLE": "lle", "UMAP": "umap"}.get(method, "tsne"))
            if method == "UMAP" and used != "umap":
                st.caption("UMAP is not installed (pip install -r requirements-optional.txt); showing t-SNE instead.")
            ui.show(reduction.embedding_plot(clustered, coords, used, "cluster" if colour != "Team" else "team",
                                             hover="constructor_name"), "red_embed")
            st.caption("How well each 2-D map keeps the real structure. Trustworthiness: how many neighbours on the "
                       "map are real neighbours (1 is perfect). Separation: how far apart the groups sit on the map.")
            st.dataframe(ui.memo("red_quality", lambda: reduction.embedding_quality(ts, feats, clusters.labels)),
                         hide_index=True,
                         column_config={"method": "Map", "separation": st.column_config.NumberColumn("Separation", format="%.2f"),
                                        "trustworthiness": st.column_config.NumberColumn("Trustworthiness", format="%.2f")})

# --------------------------------------------------------------------- report
if t_rep.open:
    with t_rep:
        question("Want everything in one file to read offline or share?")
        ui.section("EDA report", "Every chart and note from these tabs, plus the factor rankings and the watchlist.")
        st.caption("The report always uses all the data, not the sidebar filters, so two people get the same file.")
        path = config.REPORTS_DIR / "eda_report.html"
        if st.button("Build the report", type="primary", key="rep_go", icon=":material/description:"):
            with st.spinner("Running every analysis..."):
                path = report.generate_report(ui.get_tables())
            st.toast("Report ready.", icon=":material/check_circle:")
        if path.exists():
            c1, c2 = st.columns(2)
            c1.download_button("Download report (HTML)", path.read_bytes(), path.name, "text/html", key="rep_dl",
                               icon=":material/download:")
            findings = config.REPORTS_DIR / "eda_findings.md"
            if findings.exists():
                c2.download_button("Download findings (Markdown)", findings.read_bytes(), findings.name,
                                   "text/markdown", key="rep_dl_md", icon=":material/download:")
                with st.container(border=True, height=520):
                    st.markdown(findings.read_text(encoding="utf-8"))
        else:
            st.info("No report yet. Build one with the button above.", icon=":material/info:")
