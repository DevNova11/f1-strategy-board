"""End-to-end tests on the bundled sample: offline fallback, synthetic generator, models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, insights, model, pipeline, sources, synthetic
from src.eda import factor_columns, reduction

pytestmark = pytest.mark.skipif(not (config.SAMPLE_DIR / "results.parquet").exists(),
                                reason="bundled sample missing (run `python -m src.fetch --make-sample`)")


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> pipeline.Tables:
    """Build the pipeline from data/sample only, as on a machine with no network."""
    tmp = tmp_path_factory.mktemp("f1sb")
    patch = pytest.MonkeyPatch()
    patch.setattr(config, "RAW_DIR", tmp / "raw")                # empty: forces the sample fallback
    patch.setattr(config, "SYNTHETIC_DIR", tmp / "synthetic")
    patch.setattr(config, "PROCESSED_DIR", tmp / "processed")
    try:
        yield pipeline.build(save=True)
    finally:
        patch.undo()


def test_falls_back_to_the_sample_without_raw_data(tables: pipeline.Tables) -> None:
    assert tables.meta["mode"] == "sample"
    assert tables.meta["races"] > 0 and tables.meta["races_with_laps"] > 0
    for name, frame in tables.frames().items():
        assert not frame.empty, f"{name} is empty"


def test_joined_tables_have_unique_keys(tables: pipeline.Tables) -> None:
    assert not tables.driver_race.duplicated(["season", "round", "driver_id"]).any()
    assert not tables.team_race.duplicated(["season", "round", "constructor_id"]).any()
    assert not tables.team_season.duplicated(["season", "constructor_id"]).any()
    assert not tables.component.duplicated(["season", "round", "constructor_id", "component"]).any()


def test_points_survive_the_joins(tables: pipeline.Tables) -> None:
    """Summing points through three levels of aggregation must not create or lose any."""
    total = tables.driver_race["points"].sum()
    assert tables.team_race["points"].sum() == pytest.approx(total)
    assert tables.team_season["points"].sum() == pytest.approx(total)
    # Race + sprint points rebuilt from individual results must equal the official
    # standings. The one real exception is Racing Point 2020, which was docked 15
    # constructors' points by the stewards.
    season = tables.team_season.set_index(["season", "constructor_id"])
    gap = season["total_points"] - season["official_points"]
    penalties = {(2020, "racing_point"): 15.0}
    for key, value in gap[gap.abs() > 0.5].items():
        assert penalties.get(key) == pytest.approx(value), f"unexplained points gap for {key}: {value}"


def test_every_join_on_team_keys_matches_fully(tables: pipeline.Tables) -> None:
    checks = pd.DataFrame(tables.meta["join_checks"]).set_index("name")["match_rate"]
    for name in checks.index:
        if "SYNTHETIC" in name or "team label" in name or "driver code" in name or "constructor" in name:
            assert checks[name] == 1.0, name


def test_cleaning_leaves_no_gaps_in_synthetic_features(tables: pipeline.Tables) -> None:
    cols = ["spend_total_musd", "freight_tonnes", "travel_km", "customs_delay_h", "crew_fatigue_idx",
            "wind_tunnel_runs", "cfd_items", "sim_hours", "sim_track_corr_pct"]
    assert tables.team_race[cols].notna().all().all()
    assert (tables.team_race["spend_total_musd"] > 0).all()
    assert tables.team_race["travel_km"].between(0, 20_100).all()


def test_synthetic_generator_is_seeded_and_labelled() -> None:
    track = sources.load_track("sample")
    args = (track["results"], track["schedule"], track["pitstops"], track["constructor_standings"])
    first, again = synthetic.generate_all(*args, seed=7), synthetic.generate_all(*args, seed=7)
    other = synthetic.generate_all(*args, seed=8)
    for name in ("financial", "logistics", "simulation"):
        pd.testing.assert_frame_equal(first[name], again[name])
        assert (first[name]["data_origin"] == "SYNTHETIC").all()
    assert not first["financial"]["Spend"].equals(other["financial"]["Spend"])


def test_synthetic_spend_respects_the_cost_cap(tables: pipeline.Tables) -> None:
    capped = tables.team_season[tables.team_season["season"] >= 2021]
    caps = capped["season"].map(synthetic.COST_CAP_MUSD)
    assert (capped["spend_total_musd"] <= caps * 1.06).all()     # small breaches and noise allowed


def test_partial_correlation_removes_a_common_cause() -> None:
    rng = np.random.default_rng(3)
    z = rng.normal(size=800)
    frame = pd.DataFrame({"z": z, "x": z + rng.normal(scale=0.5, size=800), "y": z + rng.normal(scale=0.5, size=800)})
    raw = model.correlation(frame, "x", "y", "pearson")
    partial = model.partial_correlation(frame, "x", "y", ["z"], method="pearson")
    assert raw["r"] > 0.7                                        # x and y look strongly linked...
    assert abs(partial["r"]) < 0.1 and partial["p"] > 0.01       # ...but only through z


def test_factor_ranking_is_complete_and_recovers_planted_effects(tables: pipeline.Tables) -> None:
    race = insights.add_prior_strength(tables.team_race, tables.team_season)
    ranking = insights.rank_factors(race, "points", factor_columns(race))
    assert sorted(ranking["consensus_rank"]) == list(range(1, len(ranking) + 1))
    assert ranking["score"].between(0, 1).all()
    assert set(ranking["origin"]) <= {"REAL", "SYNTHETIC"}
    recovery = insights.recovery_check(tables.team_race)
    assert recovery["recovered"].mean() >= 0.6


def test_reliability_model_is_scored_out_of_sample(tables: pipeline.Tables) -> None:
    fitted = model.train_reliability(tables.driver_race)
    assert set(fitted.train_seasons).isdisjoint(fitted.test_seasons)
    assert max(fitted.train_seasons) < min(fitted.test_seasons)  # trained on the past, tested on the future
    assert fitted.scored["risk"].between(0, 1).all()
    assert 0.5 < fitted.metrics["roc_auc"] <= 1.0
    assert len(fitted.importance) == len(fitted.features)


def test_clustering_picks_k_by_silhouette(tables: pipeline.Tables) -> None:
    out = reduction.kmeans_profiles(tables.team_season, reduction.PROFILE_FEATURES)
    assert out.best_k == int(out.scores.loc[out.scores["silhouette"].idxmax(), "k"])
    assert out.labels.nunique() == out.best_k
    assert out.scores["silhouette"].between(-1, 1).all()


def test_lle_map_and_quality_scores(tables: pipeline.Tables) -> None:
    frame, cols = tables.team_season, reduction.PROFILE_FEATURES
    coords, used = reduction.embed(frame, cols, "lle")
    assert used == "lle" and coords.shape == (len(frame), 2) and coords.notna().all().all()
    labels = reduction.kmeans_profiles(frame, cols).labels
    scores = reduction.embedding_quality(frame, cols, labels)
    assert list(scores["method"]) == ["PCA", "t-SNE", "LLE"]
    assert scores["trustworthiness"].between(0, 1).all()
    assert scores["separation"].between(-1, 1).all()
