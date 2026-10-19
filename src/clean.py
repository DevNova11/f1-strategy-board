"""Layer 3 - cleaning: missing values, unit harmonisation, outlier flags, feature keys.

Principles
----------
* **Flag, do not delete.** Suspect values get a boolean flag column; the raw
  value stays so the EDA can study it. Only exact duplicates and rows with no
  usable key are dropped.
* **Impute conservatively and visibly.** Gaps are filled with the median of the
  closest sensible group and marked with an ``*_imputed`` flag.
* **One unit per quantity.** Spend in USD millions, distance in km, weight in
  tonnes, time in seconds.
* **Everything is logged.** Each step appends a row to a :class:`CleaningReport`
  that the dashboard shows on the Cleaning Report page.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config
from src.eda.outliers import grouped_flags, iqr_flags, mad_flags
from src.synthetic import FX_TO_USD
from src.taxonomy import classify_status

MILES_TO_KM = 1.609344
PIT_VALID_RANGE_S = (12.0, 60.0)   # pit-lane time; longer means red flag, repairs or a penalty
SLOW_LAP_FACTOR = 1.07             # the sport's own 107% rule, applied to the race's fastest lap


@dataclass
class CleaningReport:
    """Audit trail of every cleaning action."""

    steps: list[dict] = field(default_factory=list)

    def add(self, table: str, step: str, detail: str, affected: int, total: int) -> None:
        self.steps.append({"table": table, "step": step, "detail": detail, "rows_affected": int(affected),
                           "rows_total": int(total),
                           "share": round(affected / total, 4) if total else 0.0})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


def missing_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Missing-value share per table and column (only columns with gaps)."""
    rows = [
        {"table": name, "column": col, "missing": int(frame[col].isna().sum()),
         "missing_pct": round(float(frame[col].isna().mean()) * 100, 2)}
        for name, frame in tables.items() if not frame.empty
        for col in frame.columns if frame[col].isna().any()
    ]
    return pd.DataFrame(rows, columns=["table", "column", "missing", "missing_pct"])


def impute_group_median(frame: pd.DataFrame, col: str, by: list[str]) -> tuple[pd.DataFrame, int]:
    """Fill gaps in ``col`` with the group median (global median as a last resort).

    Adds ``<col>_imputed`` so imputed cells stay identifiable. Returns the new
    frame and the number of cells filled.
    """
    out = frame.copy()
    gaps = out[col].isna()
    out[f"{col}_imputed"] = gaps
    if gaps.any():
        out[col] = out[col].fillna(out.groupby(by)[col].transform("median")).fillna(out[col].median())
    return out, int(gaps.sum())


# ------------------------------------------------------------------- real data
def clean_results(results: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Classify outcomes and derive the result flags used everywhere downstream."""
    out = results.drop_duplicates(["season", "round", "driver_id"]).copy()
    report.add("results", "deduplicate", "one row per driver per race", len(results) - len(out), len(results))

    classified = out["status"].map(classify_status)
    out["outcome"] = [c[0] for c in classified]
    out["component"] = [c[1] for c in classified]
    out["dnf"] = out["outcome"].isin(["mechanical", "incident", "unspecified"])
    out["mech_dnf"] = out["outcome"] == "mechanical"
    out["win"] = out["position"] == 1
    out["podium"] = out["position"] <= 3
    report.add("results", "classify status", "status text -> finished / mechanical / incident / unspecified / other",
               int(out["dnf"].sum()), len(out))

    pit_lane = out["grid"] == 0
    field_size = out.groupby(["season", "round"])["grid"].transform("max")
    out["grid_clean"] = out["grid"].where(~pit_lane, field_size + 1).astype(float)
    out["pit_lane_start"] = pit_lane
    report.add("results", "recode grid", "grid 0 (pit-lane start) -> back of the field", int(pit_lane.sum()), len(out))

    out["positions_gained"] = out["grid_clean"] - out["position"]
    out["race_time_s"] = out["race_time_ms"] / 1000.0
    return out


def clean_laps(laps: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Drop untimed laps and flag the ones that represent real racing pace."""
    if laps.empty:
        return laps.assign(is_clean=pd.Series(dtype=bool), fuel_corrected_s=pd.Series(dtype=float),
                           lap_outlier=pd.Series(dtype=bool))
    out = laps.dropna(subset=["lap_time_s", "lap_number"]).copy()
    report.add("laps", "drop untimed laps", "no lap time recorded (red flags, retirements)",
               len(laps) - len(out), len(laps))

    out["compound"] = out["compound"].str.upper().where(
        out["compound"].str.upper().isin(["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]))
    out["is_green"] = out["track_status"].astype(str) == "1"
    fastest = out.groupby(["season", "round"])["lap_time_s"].transform("min")
    out["is_clean"] = (
        out["is_accurate"].fillna(False).astype(bool) & out["is_green"]
        & ~out["pit_in"] & ~out["pit_out"] & (out["lap_time_s"] <= SLOW_LAP_FACTOR * fastest)
    )
    report.add("laps", "flag clean laps",
               "green flag, accurate timing, not an in/out lap, within 107% of the race's fastest lap",
               int((~out["is_clean"]).sum()), len(out))

    race_laps = out.groupby(["season", "round"])["lap_number"].transform("max")
    out["fuel_corrected_s"] = out["lap_time_s"] - config.FUEL_EFFECT_S_PER_LAP * (race_laps - out["lap_number"])
    out["lap_outlier"] = grouped_flags(out, "lap_time_s", ["season", "round"], "iqr", k=3.0)
    report.add("laps", "flag outliers", "lap time outside 3xIQR fences within its race",
               int(out["lap_outlier"].sum()), len(out))
    return out.reset_index(drop=True)


def clean_pitstops(pitstops: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Flag implausible pit-lane times and express each stop relative to the field."""
    out = pitstops.dropna(subset=["constructor_id"]).copy()
    low, high = PIT_VALID_RANGE_S
    out["pit_valid"] = out["duration_s"].between(low, high)
    report.add("pitstops", "flag implausible stops",
               f"pit-lane time outside {low:.0f}-{high:.0f}s or missing (red-flag waits, repairs, penalties)",
               int((~out["pit_valid"]).sum()), len(out))
    median = out[out["pit_valid"]].groupby(["season", "round"])["duration_s"].median().rename("field_pit_s")
    out = out.merge(median, how="left", left_on=["season", "round"], right_index=True)
    out["pit_delta_s"] = (out["duration_s"] - out["field_pit_s"]).where(out["pit_valid"])
    return out


# -------------------------------------------------------------- synthetic data
def clean_financial(financial: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Harmonise the SYNTHETIC ledger to USD millions, then impute and flag."""
    n0 = len(financial)
    out = financial.drop_duplicates(["season", "round", "constructor_id", "component", "spend"]).copy()
    out = out.drop_duplicates(["season", "round", "constructor_id", "component"])
    report.add("financial", "deduplicate", "repeated ledger rows removed", n0 - len(out), n0)

    in_thousands = out["scale"].str.upper() == "K"
    non_usd = out["currency"] != "USD"
    out["spend_musd"] = out["spend"] * np.where(in_thousands, 1e-3, 1.0) * out["currency"].map(FX_TO_USD)
    report.add("financial", "harmonise scale", "thousands -> millions", int(in_thousands.sum()), len(out))
    report.add("financial", "harmonise currency", "EUR and GBP -> USD at fixed documented rates",
               int(non_usd.sum()), len(out))

    group = ["season", "constructor_id", "component"]
    out, filled = impute_group_median(out, "spend_musd", group)
    out = out.rename(columns={"spend_musd_imputed": "spend_imputed"})
    report.add("financial", "impute missing", "median of the same team, season and component", filled, len(out))

    out["spend_outlier"] = grouped_flags(out, "spend_musd", group, "mad", threshold=6.0)
    typical = out.groupby(group)["spend_musd"].transform("median")
    out["spend_musd_adj"] = out["spend_musd"].where(~out["spend_outlier"], typical)
    report.add("financial", "flag outliers",
               "robust z > 6 within team-season-component; adjusted column uses the group median",
               int(out["spend_outlier"].sum()), len(out))
    return out.drop(columns=["spend", "currency", "scale"]).reset_index(drop=True)


def clean_logistics(logistics: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Harmonise the SYNTHETIC logistics log to km and tonnes, then impute and flag."""
    out = logistics.copy()
    in_kg = out["weight_unit"].str.lower() == "kg"
    in_miles = out["distance_unit"].str.lower() == "mi"
    out["freight_tonnes"] = out["freight_weight"] * np.where(in_kg, 1e-3, 1.0)
    out["travel_km"] = out["travel_distance"] * np.where(in_miles, MILES_TO_KM, 1.0)
    report.add("logistics", "harmonise weight", "kilograms -> tonnes", int(in_kg.sum()), len(out))
    report.add("logistics", "harmonise distance", "miles -> kilometres", int(in_miles.sum()), len(out))
    out = out.drop(columns=["freight_weight", "travel_distance", "weight_unit", "distance_unit"])

    for col in ("freight_tonnes", "customs_delay_h", "crew_fatigue_idx", "spares_coverage_pct"):
        out, filled = impute_group_median(out, col, ["season", "constructor_id"])
        report.add("logistics", "impute missing", f"{col}: median of the same team and season", filled, len(out))
    out["customs_outlier"] = iqr_flags(out["customs_delay_h"], k=3.0)
    report.add("logistics", "flag outliers", "customs delay outside 3xIQR fences",
               int(out["customs_outlier"].sum()), len(out))
    return out


def clean_simulation(simulation: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Impute and flag the SYNTHETIC simulation log."""
    out = simulation.copy()
    for col in ("wind_tunnel_runs", "cfd_items", "sim_hours", "sim_track_corr_pct"):
        out, filled = impute_group_median(out, col, ["season", "constructor_id"])
        report.add("simulation", "impute missing", f"{col}: median of the same team and season", filled, len(out))
    out["cfd_outlier"] = grouped_flags(out, "cfd_items", ["season", "constructor_id"], "mad", threshold=6.0)
    typical = out.groupby(["season", "constructor_id"])["cfd_items"].transform("median")
    out["cfd_items_raw"] = out["cfd_items"]
    out["cfd_items"] = out["cfd_items"].where(~out["cfd_outlier"], typical)
    report.add("simulation", "flag outliers", "CFD items with robust z > 6 replaced by the team-season median",
               int(out["cfd_outlier"].sum()), len(out))
    return out


# ---------------------------------------------------------------- feature keys
def add_feature_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the string keys used for joins, selection and the watchlist.

    ``race_key`` = ``2023-05``; ``team_race_key`` = ``2023-05|ferrari``;
    ``driver_race_key`` = ``2023-05|leclerc``. Keys are only added when the
    columns they are built from exist.
    """
    out = frame.copy()
    if {"season", "round"} <= set(out.columns):
        out["race_key"] = out["season"].astype(int).astype(str) + "-" + out["round"].astype(int).astype(str).str.zfill(2)
        if "constructor_id" in out:
            out["team_race_key"] = out["race_key"] + "|" + out["constructor_id"].astype(str)
        if "driver_id" in out:
            out["driver_race_key"] = out["race_key"] + "|" + out["driver_id"].astype(str)
    if "constructor_id" in out and "lineage" not in out:
        out["lineage"] = out["constructor_id"].map(config.TEAM_LINEAGE)
    if "lineage" in out:
        out["team"] = out["lineage"].map(config.LINEAGE_LABEL)
    return out


def flag_outliers(frame: pd.DataFrame, cols: list[str], by: list[str] | None = None,
                  method: str = "iqr") -> pd.DataFrame:
    """Add a ``<col>_outlier`` flag for each column, optionally within groups."""
    out = frame.copy()
    for col in cols:
        if col in out:
            out[f"{col}_outlier"] = grouped_flags(out, col, by, method)
    return out


__all__ = [
    "CleaningReport", "missing_summary", "impute_group_median", "clean_results", "clean_laps",
    "clean_pitstops", "clean_financial", "clean_logistics", "clean_simulation", "add_feature_keys",
    "flag_outliers", "mad_flags",
]
