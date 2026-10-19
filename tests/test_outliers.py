"""EDA step 3 tests: IQR, z-score, MAD, Isolation Forest, the vote, and the watchlist loop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import clean, feedback
from src.eda import outliers


@pytest.fixture
def series() -> pd.Series:
    """1..9 plus one extreme value and one gap."""
    return pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100, np.nan], dtype=float)


def test_iqr_bounds_match_hand_calculation() -> None:
    low, high = outliers.iqr_bounds(pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float))
    assert (low, high) == pytest.approx((3 - 1.5 * 4, 7 + 1.5 * 4))    # Q1 = 3, Q3 = 7, IQR = 4


def test_iqr_bounds_of_empty_series_are_nan() -> None:
    low, high = outliers.iqr_bounds(pd.Series([np.nan, np.nan]))
    assert np.isnan(low) and np.isnan(high)


def test_iqr_flags_catch_the_extreme_and_never_flag_nan(series: pd.Series) -> None:
    flags = outliers.iqr_flags(series)
    assert flags.tolist() == [False] * 9 + [True, False]
    assert flags.dtype == bool


def test_iqr_multiplier_controls_strictness() -> None:
    values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 16], dtype=float)   # fences: 14.5 at k=1.5, 21.25 at k=3
    assert outliers.iqr_flags(values, k=1.5).sum() == 1
    assert outliers.iqr_flags(values, k=3.0).sum() == 0


def test_zscore_flags(series: pd.Series) -> None:
    assert outliers.zscore_flags(series, threshold=2.5).tolist() == [False] * 9 + [True, False]
    assert outliers.zscore_flags(series, threshold=10).sum() == 0


def test_zscore_of_constant_series_flags_nothing() -> None:
    assert not outliers.zscore_flags(pd.Series([5.0] * 10)).any()


def test_mad_is_not_masked_by_multiple_outliers() -> None:
    """Two huge values inflate the standard deviation enough to hide each other
    from the z-score, but not from the median-based score."""
    values = pd.Series([10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 10.3, 500.0, 520.0])
    assert outliers.zscore_flags(values).sum() == 0
    assert outliers.mad_flags(values).tolist() == [False] * 8 + [True, True]


def test_robust_zscore_handles_zero_spread() -> None:
    assert (outliers.robust_zscore(pd.Series([3.0, 3.0, 3.0, 3.0])) == 0).all()


def test_mad_flags_a_lone_extreme_among_identical_values() -> None:
    """MAD is zero when most values are equal; the fallback must still score the odd one out."""
    values = pd.Series([2.0] * 9 + [40.0])
    assert outliers.mad_flags(values, threshold=6.0).tolist() == [False] * 9 + [True]


def test_grouped_flags_judge_each_group_on_its_own_scale() -> None:
    frame = pd.DataFrame({"race": ["fast"] * 8 + ["slow"] * 8,
                          "lap": [80, 81, 80, 81, 80, 81, 80, 95] + [120, 121, 120, 121, 120, 121, 120, 121]})
    pooled = outliers.iqr_flags(frame["lap"])
    grouped = outliers.grouped_flags(frame, "lap", ["race"], "iqr")
    assert not pooled.any()                                      # pooled, 95s hides between the two tracks
    assert grouped.tolist() == [False] * 7 + [True] + [False] * 8
    assert grouped.index.equals(frame.index)


def test_isolation_forest_finds_the_odd_combination_and_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 300)
    frame = pd.DataFrame({"a": x, "b": x + rng.normal(0, 0.05, 300)})   # a and b move together...
    frame.loc[0, ["a", "b"]] = [2.0, -2.0]                               # ...except here
    first = outliers.isolation_forest_flags(frame, ["a", "b"], contamination=0.02)
    again = outliers.isolation_forest_flags(frame, ["a", "b"], contamination=0.02)
    assert first["iso_flag"].iloc[0]
    assert not outliers.iqr_flags(frame["a"]).iloc[0]            # univariate rules cannot see it
    assert first["iso_flag"].sum() == pytest.approx(300 * 0.02, abs=2)
    pd.testing.assert_frame_equal(first, again)


def test_isolation_forest_tolerates_missing_values_and_tiny_frames() -> None:
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 3.0]})
    out = outliers.isolation_forest_flags(frame, ["a", "b"])
    assert len(out) == 3 and not out["iso_flag"].any()


def test_detect_anomalies_votes() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"season": 2024, "spend": rng.normal(10, 1, 200), "delay": rng.normal(5, 1, 200)})
    frame.loc[7, ["spend", "delay"]] = [60.0, 40.0]              # blatant in both columns
    scored = outliers.detect_anomalies(frame, ["spend", "delay"], by=["season"])
    assert {"iqr_hits", "z_hits", "iso_flag", "iso_score", "anomaly_votes", "anomaly", "anomaly_reason"} <= set(scored.columns)
    assert scored.loc[7, "anomaly_votes"] == 3 and scored.loc[7, "anomaly"]
    assert scored.loc[7, "iqr_hits"] == 2
    assert scored["anomaly_votes"].between(0, 3).all()
    assert (scored["anomaly"] == (scored["anomaly_votes"] >= 2)).all()
    assert scored["anomaly"].mean() < 0.1
    assert scored.loc[7, "iso_score"] == scored["iso_score"].max()


def test_clean_flag_outliers_adds_one_flag_per_column() -> None:
    frame = pd.DataFrame({"season": [1] * 10, "v": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100.0]})
    out = clean.flag_outliers(frame, ["v", "absent"], by=["season"])
    assert out["v_outlier"].sum() == 1 and "absent_outlier" not in out


def test_watchlist_round_trip_feeds_back_into_the_eda(tmp_path) -> None:
    path = tmp_path / "watchlist.csv"
    frame = pd.DataFrame({"team_race_key": ["2024-01|ferrari", "2024-02|ferrari", "2024-03|haas"],
                          "anomaly_votes": [3, 2, 0], "anomaly_reason": ["Late parts", "Customs delay", ""],
                          "value": [1.0, 2.0, 3.0]})
    assert feedback.load_watchlist(path).empty

    feedback.add_to_watchlist(frame.iloc[:2], note="check freight", path=path)
    stored = feedback.load_watchlist(path)
    assert stored["key"].tolist() == ["2024-01|ferrari", "2024-02|ferrari"]
    assert set(stored["note"]) == {"check freight"} and set(stored["source"]) == {"dashboard"}

    feedback.add_to_watchlist(frame.iloc[[0]], note="re-flagged", path=path)       # no duplicate keys
    stored = feedback.load_watchlist(path)
    assert len(stored) == 2 and stored.set_index("key").loc["2024-01|ferrari", "note"] == "re-flagged"

    flagged = feedback.apply_watchlist(frame, stored, mode="flag")
    assert flagged["on_watchlist"].tolist() == [True, True, False]
    assert outliers.annotate_watchlist(frame, stored)["on_watchlist"].tolist() == [True, True, False]
    excluded = feedback.apply_watchlist(frame, stored, mode="exclude")
    assert excluded["team_race_key"].tolist() == ["2024-03|haas"]

    assert feedback.remove_from_watchlist(["2024-01|ferrari"], path=path)["key"].tolist() == ["2024-02|ferrari"]
