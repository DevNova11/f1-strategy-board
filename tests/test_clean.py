"""Layer 3 tests: missing values, unit harmonisation, flags and feature keys."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import clean, config, ingest
from src.synthetic import FX_TO_USD
from src.taxonomy import classify_status


@pytest.mark.parametrize("status, expected", [
    ("Finished", ("finished", "none")), ("+2 Laps", ("finished", "none")), ("Lapped", ("finished", "none")),
    ("Gearbox", ("mechanical", "gearbox")), ("Power Unit", ("mechanical", "power_unit")),
    ("Water pressure", ("mechanical", "hydraulics_cooling")), ("Brakes", ("mechanical", "brakes")),
    ("Collision damage", ("incident", "none")), ("Accident", ("incident", "none")),
    ("Retired", ("unspecified", "unknown")), ("Disqualified", ("other", "none")), (None, ("other", "none")),
])
def test_classify_status(status: object, expected: tuple[str, str]) -> None:
    assert classify_status(status) == expected


def test_clean_results_flags_and_grid_recode(results: pd.DataFrame) -> None:
    report = clean.CleaningReport()
    out = clean.clean_results(ingest.align_results(results), report)
    r1 = out[out["round"] == 1].set_index("driver_id")
    assert r1.loc["leclerc", "mech_dnf"] and r1.loc["leclerc", "component"] == "gearbox"
    assert r1.loc["sainz", "dnf"] and not r1.loc["sainz", "mech_dnf"]
    assert r1.loc["hamilton", "win"] and not r1.loc["russell", "win"]
    assert not r1.loc["russell", "dnf"]                          # "+1 Lap" is a finish
    pit_lane = out[(out["round"] == 2) & (out["driver_id"] == "sainz")].iloc[0]
    assert pit_lane["pit_lane_start"] and pit_lane["grid_clean"] == 4   # back of a 3-car grid
    assert {"deduplicate", "classify status", "recode grid"} <= set(report.to_frame()["step"])


def test_clean_financial_harmonises_units(financial_raw: pd.DataFrame) -> None:
    report = clean.CleaningReport()
    aligned, _ = ingest.align_financial(financial_raw)
    out = clean.clean_financial(aligned, report).set_index("round")
    eur = FX_TO_USD["EUR"]
    assert len(out) == 10                                        # duplicate removed
    assert out.loc[1, "spend_musd"] == pytest.approx(2.0 * eur)  # EUR millions -> USD millions
    assert out.loc[2, "spend_musd"] == pytest.approx(2.0 * eur)  # 2000 thousand EUR is the same money
    assert {"spend", "currency", "scale"}.isdisjoint(out.columns)


def test_clean_financial_imputes_and_flags(financial_raw: pd.DataFrame) -> None:
    report = clean.CleaningReport()
    aligned, _ = ingest.align_financial(financial_raw)
    out = clean.clean_financial(aligned, report).set_index("round")
    eur = FX_TO_USD["EUR"]
    assert out.loc[3, "spend_imputed"] and out["spend_imputed"].sum() == 1
    assert out.loc[3, "spend_musd"] == pytest.approx(2.0 * eur)  # filled with the group median
    assert out["spend_musd"].notna().all()
    assert out.loc[4, "spend_outlier"] and out["spend_outlier"].sum() == 1
    assert out.loc[4, "spend_musd"] == pytest.approx(40.0 * eur)       # raw value is kept
    assert out.loc[4, "spend_musd_adj"] == pytest.approx(2.0 * eur)    # adjusted value is the group median
    steps = report.to_frame().set_index("step")["rows_affected"]
    assert steps["deduplicate"] == 1 and steps["harmonise scale"] == 1 and steps["impute missing"] == 1


def test_clean_logistics_converts_kg_and_miles() -> None:
    raw = pd.DataFrame({
        "season": 2024, "round": [1, 2, 3], "constructor_id": ["haas", "williams", "ferrari"],
        "freight_weight": [40.0, 40_000.0, 40.0], "weight_unit": ["t", "kg", "t"],
        "travel_distance": [100.0, 100.0, 100.0], "distance_unit": ["mi", "km", "km"],
        "freight_mode": "air", "customs_delay_h": [5.0, np.nan, 5.0], "late_parts": 0,
        "spares_coverage_pct": 90.0, "crew_fatigue_idx": 40.0, "setup_hours": 30.0, "data_origin": "SYNTHETIC",
    })
    out = clean.clean_logistics(raw, clean.CleaningReport()).set_index("round")
    assert out["freight_tonnes"].tolist() == pytest.approx([40.0, 40.0, 40.0])
    assert out.loc[1, "travel_km"] == pytest.approx(160.9344)
    assert out.loc[2, "travel_km"] == pytest.approx(100.0)
    assert out.loc[2, "customs_delay_h_imputed"] and out.loc[2, "customs_delay_h"] == pytest.approx(5.0)
    assert {"weight_unit", "distance_unit", "freight_weight", "travel_distance"}.isdisjoint(out.columns)


def test_impute_group_median_uses_the_group_then_the_global_median() -> None:
    frame = pd.DataFrame({"g": ["a", "a", "a", "b", "b", "c"], "v": [1.0, 3.0, np.nan, 10.0, np.nan, np.nan]})
    out, filled = clean.impute_group_median(frame, "v", ["g"])
    assert filled == 3
    assert out["v"].tolist() == pytest.approx([1.0, 3.0, 2.0, 10.0, 10.0, 3.0])   # group c falls back to global
    assert out["v_imputed"].tolist() == [False, False, True, False, True, True]
    assert frame["v"].isna().sum() == 3                          # input is not mutated


def test_clean_laps_flags_only_real_racing_laps(laps: pd.DataFrame, results: pd.DataFrame, weather: pd.DataFrame) -> None:
    report = clean.CleaningReport()
    aligned, _ = ingest.align_laps(laps, ingest.align_results(results))
    synced, _ = ingest.sync_weather_to_laps(aligned, weather)
    out = clean.clean_laps(synced, report).set_index("lap_number")
    assert 11 not in out.index                                   # untimed lap dropped
    assert not out.loc[6, "is_clean"]                            # in lap
    assert not out.loc[7, "is_clean"]                            # out lap
    assert not out.loc[9, "is_clean"]                            # safety car
    assert out.loc[[1, 2, 3, 4, 5, 8, 10, 12], "is_clean"].all()
    assert set(out["compound"]) == {"SOFT", "HARD"}              # compound names upper-cased
    last = out.index.max()
    assert out.loc[last, "fuel_corrected_s"] == pytest.approx(out.loc[last, "lap_time_s"])
    assert out.loc[1, "fuel_corrected_s"] == pytest.approx(
        out.loc[1, "lap_time_s"] - config.FUEL_EFFECT_S_PER_LAP * (last - 1))


def test_clean_laps_applies_the_107_percent_rule(laps: pd.DataFrame, results: pd.DataFrame) -> None:
    laps = laps.copy()
    laps.loc[2, "lap_time_s"] = 90.0 * 1.2                       # green-flag lap, but far too slow
    aligned, _ = ingest.align_laps(laps, ingest.align_results(results))
    aligned = aligned.assign(track_temp_c=30.0)
    out = clean.clean_laps(aligned, clean.CleaningReport()).set_index("lap_number")
    assert not out.loc[3, "is_clean"]


def test_clean_pitstops_validity_and_delta() -> None:
    pit = pd.DataFrame({"season": 2024, "round": 1, "driver_id": list("abcd"), "constructor_id": "x", "stop": 1,
                        "lap": 10, "duration_s": [20.0, 24.0, 1200.0, np.nan]})
    out = clean.clean_pitstops(pit, clean.CleaningReport())
    assert out["pit_valid"].tolist() == [True, True, False, False]
    assert out["pit_delta_s"].iloc[:2].tolist() == pytest.approx([-2.0, 2.0])   # field median is 22s
    assert out["pit_delta_s"].iloc[2:].isna().all()


def test_add_feature_keys() -> None:
    frame = pd.DataFrame({"season": [2023, 2023], "round": [5, 12], "constructor_id": ["ferrari", "alphatauri"],
                          "driver_id": ["leclerc", "tsunoda"]})
    out = clean.add_feature_keys(frame)
    assert out["race_key"].tolist() == ["2023-05", "2023-12"]
    assert out["team_race_key"].tolist() == ["2023-05|ferrari", "2023-12|alphatauri"]
    assert out["driver_race_key"].tolist() == ["2023-05|leclerc", "2023-12|tsunoda"]
    assert out["lineage"].tolist() == ["ferrari", "rb"]          # rebrands collapse onto one lineage
    assert out["team"].tolist() == ["Ferrari", "RB / AlphaTauri / Toro Rosso"]


def test_missing_summary_lists_only_columns_with_gaps() -> None:
    summary = clean.missing_summary({"t": pd.DataFrame({"a": [1.0, np.nan, np.nan, 4.0], "b": [1, 2, 3, 4]})})
    assert summary["column"].tolist() == ["a"]
    assert summary["missing_pct"].iloc[0] == 50.0
