"""Layer 2 tests: schema alignment, timestamp sync and key joins."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import clean, ingest
from src.taxonomy import normalise_team


@pytest.mark.parametrize("raw, expected", [
    ("Oracle Red Bull Racing", "red_bull"), ("RED BULL", "red_bull"), ("Haas F1 Team", "haas"),
    ("Alpine F1 Team", "alpine"), ("RB F1 Team", "rb"), ("Racing Bulls", "rb"), ("Kick Sauber", "sauber"),
    ("alfa", "alfa"), ("Alfa Romeo", "alfa"), ("aston_martin", "aston_martin"), ("McLaren", "mclaren"),
])
def test_normalise_team_resolves_every_spelling(raw: str, expected: str) -> None:
    assert normalise_team(raw) == expected


@pytest.mark.parametrize("raw", ["Not A Team", "", None, float("nan")])
def test_normalise_team_rejects_unknown_labels(raw: object) -> None:
    assert normalise_team(raw) is None


def test_align_financial_renames_and_reports_unmatched(financial_raw: pd.DataFrame) -> None:
    aligned, check = ingest.align_financial(financial_raw)
    assert {"season", "round", "constructor_id", "component", "spend", "currency", "scale"} <= set(aligned.columns)
    assert set(aligned["constructor_id"]) == {"ferrari"}
    assert set(aligned["component"]) == {"aero"}                 # lower-cased
    assert check.left_rows == len(financial_raw)
    assert check.matched_rows == len(financial_raw) - 1          # the "Not A Team" row is dropped
    assert check.match_rate == pytest.approx(1 - 1 / len(financial_raw))


def test_align_laps_joins_on_driver_code(laps: pd.DataFrame, results: pd.DataFrame) -> None:
    aligned, check = ingest.align_laps(laps, ingest.align_results(results))
    assert check.match_rate == 1.0
    assert set(aligned["driver_id"]) == {"hamilton"}
    assert set(aligned["constructor_id"]) == {"mercedes"}
    assert len(aligned) == len(laps)                             # a join must never duplicate laps


def test_align_laps_falls_back_to_team_name(laps: pd.DataFrame, results: pd.DataFrame) -> None:
    stranger = laps.assign(driver_code="XXX", team_name="Scuderia Ferrari")
    aligned, check = ingest.align_laps(stranger, ingest.align_results(results))
    assert check.matched_rows == 0
    assert set(aligned["constructor_id"]) == {"ferrari"}


def test_weather_sync_takes_latest_earlier_sample(laps: pd.DataFrame, weather: pd.DataFrame) -> None:
    synced, _ = ingest.sync_weather_to_laps(laps, weather, tolerance_s=300)
    by_lap = synced.set_index("lap_number")
    assert by_lap.loc[1, "track_temp_c"] == 30.0                 # lap starts at 0s   -> sample at 0s
    assert by_lap.loc[3, "track_temp_c"] == 30.0                 # lap starts at 200s -> still sample at 0s
    assert by_lap.loc[4, "track_temp_c"] == 31.0                 # lap starts at 300s -> sample at 250s
    assert by_lap.loc[9, "track_temp_c"] == 33.0                 # lap starts at 800s -> sample at 750s
    assert (synced["track_temp_c"] != 99.0).all()                # never borrows another race's weather


def test_weather_sync_respects_tolerance(laps: pd.DataFrame, weather: pd.DataFrame) -> None:
    synced, check = ingest.sync_weather_to_laps(laps, weather, tolerance_s=60)
    by_lap = synced.set_index("lap_number")
    assert np.isnan(by_lap.loc[3, "track_temp_c"])               # 200s after the last sample: too stale
    assert by_lap.loc[1, "track_temp_c"] == 30.0
    assert check.matched_rows < check.left_rows


def test_weather_sync_keeps_every_lap(laps: pd.DataFrame, weather: pd.DataFrame) -> None:
    laps = laps.copy()
    laps.loc[0, "lap_start_s"] = np.nan
    synced, _ = ingest.sync_weather_to_laps(laps, weather)
    assert len(synced) == len(laps)


def test_summarise_weather_flags_wet_races(weather: pd.DataFrame) -> None:
    summary = ingest.summarise_weather(weather).set_index("round")
    assert summary.loc[1, "rain_share"] == 0.5
    assert bool(summary.loc[1, "wet_race"])
    assert summary.loc[1, "track_temp_c"] == pytest.approx(31.5)
    assert summary.loc[1, "track_temp_range_c"] == pytest.approx(3.0)


def test_group_slope_recovers_known_slopes() -> None:
    frame = pd.DataFrame({"g": ["a"] * 6 + ["b"] * 6 + ["c"] * 2, "x": list(range(6)) * 2 + [0, 1]})
    frame["y"] = np.where(frame["g"] == "a", 2.0 * frame["x"] + 1, -0.5 * frame["x"] + 3)
    slopes = ingest.group_slope(frame, ["g"], "x", "y", min_points=5).set_index("g")["slope"]
    assert slopes["a"] == pytest.approx(2.0)
    assert slopes["b"] == pytest.approx(-0.5)
    assert np.isnan(slopes["c"])                                 # too few points to trust


def test_driver_and_team_joins_keep_one_row_per_key(results: pd.DataFrame) -> None:
    report = clean.CleaningReport()
    res = clean.clean_results(ingest.align_results(results), report)
    pit = pd.DataFrame({"season": 2024, "round": [1, 1, 2], "driver_id": ["hamilton", "leclerc", "hamilton"],
                        "stop": 1, "lap": 20, "time_of_day": "14:00:00", "duration_s": [22.0, 24.0, 400.0]})
    pit, check = ingest.align_pitstops(pit, res)
    assert check.match_rate == 1.0
    pit = clean.clean_pitstops(pit, report)
    driver_race, _ = ingest.build_driver_race(res, pd.DataFrame(), pd.DataFrame(), pit, pd.DataFrame())
    assert len(driver_race) == len(results)
    assert not driver_race.duplicated(["season", "round", "driver_id"]).any()

    ham = driver_race.query("driver_id == 'hamilton'").set_index("round")
    assert ham.loc[1, "pit_median_s"] == 22.0
    assert np.isnan(ham.loc[2, "pit_median_s"])                  # the 400s stop is not a valid pit stop

    team = driver_race.groupby(ingest.TEAM_KEYS, as_index=False).agg(points=("points", "sum"), dnf=("dnf", "sum"))
    assert len(team) == 4 and not team.duplicated(ingest.TEAM_KEYS).any()
    ferrari_r1 = team.query("constructor_id == 'ferrari' and round == 1").iloc[0]
    assert ferrari_r1["dnf"] == 2 and ferrari_r1["points"] == 0
