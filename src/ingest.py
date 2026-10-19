"""Layer 2 - schema alignment, timestamp synchronisation and key joins.

The five sources arrive with different column names, team spellings and
grains. This module

1. **aligns schemas** so every table speaks ``season / round / constructor_id /
   driver_id / component``;
2. **synchronises timestamps**, attaching the weather sample in force at the
   start of each lap (``merge_asof`` on session time);
3. **joins on keys** to build the analysis tables (driver x race, team x race,
   team x season, team x round x component) and records how many rows matched.

Alignment runs before cleaning and the joins run after it, because spend can
only be summed once currencies and scales have been harmonised. ``src.pipeline``
does the sequencing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src import config
from src.taxonomy import normalise_team

RACE_KEYS = ["season", "round"]
TEAM_KEYS = ["season", "round", "constructor_id"]
DRIVER_KEYS = ["season", "round", "driver_id"]


@dataclass
class JoinCheck:
    """Outcome of one alignment or join: how many left rows found a partner."""

    name: str
    left_rows: int
    matched_rows: int

    @property
    def match_rate(self) -> float:
        return self.matched_rows / self.left_rows if self.left_rows else 1.0

    def to_dict(self) -> dict:
        return {**asdict(self), "unmatched_rows": self.left_rows - self.matched_rows,
                "match_rate": round(self.match_rate, 4)}


# ------------------------------------------------------------- schema alignment
def align_results(results: pd.DataFrame) -> pd.DataFrame:
    """Type the real classification table and add the team lineage."""
    out = results.copy()
    out["season"] = out["season"].astype(int)
    out["round"] = out["round"].astype(int)
    out["lineage"] = out["constructor_id"].map(config.TEAM_LINEAGE)
    return out


def _align_team_table(frame: pd.DataFrame, rename: dict[str, str], team_col: str,
                      name: str) -> tuple[pd.DataFrame, JoinCheck]:
    out = frame.rename(columns=rename).copy()
    out["constructor_id"] = out[team_col].map(normalise_team)
    check = JoinCheck(f"{name}: team label -> constructor_id", len(out), int(out["constructor_id"].notna().sum()))
    out = out.dropna(subset=["constructor_id"]).drop(columns=[team_col] if team_col != "constructor_id" else [])
    out["season"] = out["season"].astype(int)
    out["round"] = out["round"].astype(int)
    return out, check


def align_financial(financial: pd.DataFrame) -> tuple[pd.DataFrame, JoinCheck]:
    """SYNTHETIC ledger -> canonical keys. Spend is left in its original unit."""
    out, check = _align_team_table(
        financial,
        {"Year": "season", "GP": "round", "Component": "component", "Spend": "spend",
         "Currency": "currency", "Scale": "scale"},
        "Team", "financial",
    )
    out["component"] = out["component"].str.strip().str.lower()
    return out, check


def align_logistics(logistics: pd.DataFrame) -> tuple[pd.DataFrame, JoinCheck]:
    """SYNTHETIC logistics log -> canonical keys. Units are left as logged."""
    return _align_team_table(logistics, {"rnd": "round"}, "team", "logistics")


def align_simulation(simulation: pd.DataFrame) -> tuple[pd.DataFrame, JoinCheck]:
    """SYNTHETIC simulation log -> canonical keys."""
    return _align_team_table(
        simulation, {"season_year": "season", "round_no": "round", "constructor": "team_label"},
        "team_label", "simulation",
    )


def align_laps(laps: pd.DataFrame, results: pd.DataFrame) -> tuple[pd.DataFrame, JoinCheck]:
    """Attach ``driver_id`` and ``constructor_id`` to FastF1 laps.

    FastF1 identifies drivers by three-letter code and teams by a display
    name; Jolpica uses ids. The join key is ``(season, round, driver_code)``,
    with the normalised FastF1 team name as a fallback for the constructor.
    """
    if laps.empty:
        return laps.assign(driver_id=pd.Series(dtype=str), constructor_id=pd.Series(dtype=str)), \
            JoinCheck("laps: driver code -> driver_id", 0, 0)
    lookup = results[["season", "round", "driver_code", "driver_id", "constructor_id"]].drop_duplicates(
        ["season", "round", "driver_code"])
    out = laps.merge(lookup, how="left", on=["season", "round", "driver_code"])
    check = JoinCheck("laps: driver code -> driver_id", len(out), int(out["driver_id"].notna().sum()))
    out["constructor_id"] = out["constructor_id"].fillna(out["team_name"].map(normalise_team))
    out["lineage"] = out["constructor_id"].map(config.TEAM_LINEAGE)
    return out, check


def align_pitstops(pitstops: pd.DataFrame, results: pd.DataFrame) -> tuple[pd.DataFrame, JoinCheck]:
    """Attach the constructor to each real pit stop via ``(season, round, driver_id)``."""
    lookup = results[DRIVER_KEYS + ["constructor_id"]].drop_duplicates(DRIVER_KEYS)
    out = pitstops.merge(lookup, how="left", on=DRIVER_KEYS)
    return out, JoinCheck("pit stops: driver -> constructor", len(out), int(out["constructor_id"].notna().sum()))


# -------------------------------------------------------------- timestamp sync
WEATHER_COLS = ["air_temp_c", "track_temp_c", "humidity_pct", "wind_speed_ms", "rainfall"]


def sync_weather_to_laps(laps: pd.DataFrame, weather: pd.DataFrame,
                         tolerance_s: float = 180.0) -> tuple[pd.DataFrame, JoinCheck]:
    """Give every lap the weather sample in force when the lap started.

    Laps and weather are on the same session clock (seconds from session
    start) but sampled at different rates, so this is a backward as-of join per
    race with a tolerance: a lap only gets weather measured in the previous
    ``tolerance_s`` seconds.
    """
    if laps.empty or weather.empty:
        out = laps.copy()
        for col in WEATHER_COLS:
            out[col] = np.nan
        return out, JoinCheck("laps <- weather (as-of on session time)", len(out), 0)

    timed = laps[laps["lap_start_s"].notna()].sort_values("lap_start_s")
    untimed = laps[laps["lap_start_s"].isna()]
    wx = weather[RACE_KEYS + ["time_s"] + WEATHER_COLS].dropna(subset=["time_s"]).sort_values("time_s")
    merged = pd.merge_asof(timed, wx, left_on="lap_start_s", right_on="time_s", by=RACE_KEYS,
                           direction="backward", tolerance=tolerance_s).drop(columns="time_s")
    out = pd.concat([merged, untimed], ignore_index=True).sort_values(
        RACE_KEYS + ["driver_code", "lap_number"]).reset_index(drop=True)
    return out, JoinCheck("laps <- weather (as-of on session time)", len(out),
                          int(out["track_temp_c"].notna().sum()))


def summarise_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Race-level weather summary (one row per season and round)."""
    cols = RACE_KEYS + ["air_temp_c", "track_temp_c", "humidity_pct", "wind_speed_ms",
                        "rain_share", "wet_race", "track_temp_range_c"]
    if weather.empty:
        return pd.DataFrame(columns=cols)
    out = weather.groupby(RACE_KEYS, as_index=False).agg(
        air_temp_c=("air_temp_c", "mean"), track_temp_c=("track_temp_c", "mean"),
        humidity_pct=("humidity_pct", "mean"), wind_speed_ms=("wind_speed_ms", "mean"),
        rain_share=("rainfall", "mean"),
        track_temp_range_c=("track_temp_c", lambda s: s.max() - s.min()),
    )
    out["wet_race"] = out["rain_share"] > 0.10
    return out[cols]


# ---------------------------------------------------------------------- joins
def group_slope(frame: pd.DataFrame, keys: list[str], x: str, y: str, min_points: int = 5) -> pd.DataFrame:
    """Least-squares slope of ``y`` on ``x`` within each group, vectorised.

    Returns ``keys + ["slope", "n"]``; groups with fewer than ``min_points``
    rows or no spread in ``x`` get a NaN slope.
    """
    data = frame.dropna(subset=[x, y]).copy()
    data["_x"], data["_y"] = data[x].astype(float), data[y].astype(float)
    data["_xy"], data["_xx"] = data["_x"] * data["_y"], data["_x"] ** 2
    agg = data.groupby(keys).agg(n=("_x", "size"), sx=("_x", "sum"), sy=("_y", "sum"),
                                 sxy=("_xy", "sum"), sxx=("_xx", "sum")).reset_index()
    denom = agg["n"] * agg["sxx"] - agg["sx"] ** 2
    slope = (agg["n"] * agg["sxy"] - agg["sx"] * agg["sy"]) / denom.where(denom.abs() > 1e-9)
    agg["slope"] = slope.where(agg["n"] >= min_points)
    return agg[keys + ["slope", "n"]]


def build_stints(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per stint with its tyre-degradation slope (seconds per lap).

    The slope is fitted on clean, fuel-corrected laps only, so it reflects tyre
    wear rather than traffic, safety cars or the car getting lighter.
    """
    keys = ["season", "round", "driver_id", "constructor_id", "stint"]
    if laps.empty:
        return pd.DataFrame(columns=keys + ["compound", "stint_laps", "deg_slope_s_per_lap"])
    base = laps.dropna(subset=["stint", "driver_id"]).groupby(keys, as_index=False).agg(
        compound=("compound", "first"), stint_laps=("lap_number", "count"),
        start_lap=("lap_number", "min"))
    slopes = group_slope(laps[laps["is_clean"]], keys, "tyre_life", "fuel_corrected_s", min_points=6)
    out = base.merge(slopes.rename(columns={"slope": "deg_slope_s_per_lap", "n": "clean_laps"}),
                     how="left", on=keys)
    # A real stint does not gain or lose more than about a second a lap.
    out.loc[out["deg_slope_s_per_lap"].abs() > 1.0, "deg_slope_s_per_lap"] = np.nan
    return out


def build_driver_race(results: pd.DataFrame, laps: pd.DataFrame, stints: pd.DataFrame,
                      pitstops: pd.DataFrame, sprints: pd.DataFrame) -> tuple[pd.DataFrame, list[JoinCheck]]:
    """Join everything known about one driver in one race."""
    out = results.copy()
    checks: list[JoinCheck] = []

    if not laps.empty:
        clean = laps[laps["is_clean"] & laps["driver_id"].notna()]
        lap_feat = clean.groupby(DRIVER_KEYS, as_index=False).agg(
            clean_laps=("lap_time_s", "size"), lap_median_s=("lap_time_s", "median"),
            lap_std_s=("lap_time_s", "std"), track_temp_lap_c=("track_temp_c", "mean"))
        field = lap_feat.groupby(RACE_KEYS)["lap_median_s"].transform("median")
        lap_feat["pace_delta_pct"] = (lap_feat["lap_median_s"] / field - 1) * 100
        lap_feat.loc[lap_feat["clean_laps"] < 8, ["pace_delta_pct", "lap_std_s"]] = np.nan
        out = out.merge(lap_feat, how="left", on=DRIVER_KEYS)
        checks.append(JoinCheck("results <- lap features", len(out), int(out["lap_median_s"].notna().sum())))

        stint_feat = stints.groupby(DRIVER_KEYS, as_index=False).agg(
            n_stints=("stint", "nunique"), stint_laps_mean=("stint_laps", "mean"),
            deg_slope_s_per_lap=("deg_slope_s_per_lap", "mean"))
        out = out.merge(stint_feat, how="left", on=DRIVER_KEYS)
    else:
        for col in ("clean_laps", "lap_median_s", "lap_std_s", "track_temp_lap_c", "pace_delta_pct",
                    "n_stints", "stint_laps_mean", "deg_slope_s_per_lap"):
            out[col] = np.nan

    valid = pitstops[pitstops["pit_valid"]]
    pit_feat = valid.groupby(DRIVER_KEYS, as_index=False).agg(
        n_stops=("stop", "count"), pit_median_s=("duration_s", "median"), pit_delta_s=("pit_delta_s", "mean"))
    out = out.merge(pit_feat, how="left", on=DRIVER_KEYS)
    checks.append(JoinCheck("results <- pit stops", len(out), int(out["pit_median_s"].notna().sum())))

    if not sprints.empty:
        sp = sprints.groupby(DRIVER_KEYS, as_index=False)["points"].sum().rename(columns={"points": "sprint_points"})
        out = out.merge(sp, how="left", on=DRIVER_KEYS)
    else:
        out["sprint_points"] = np.nan
    out["sprint_points"] = out["sprint_points"].fillna(0.0)
    return out, checks


def build_team_race(driver_race: pd.DataFrame, financial: pd.DataFrame, logistics: pd.DataFrame,
                    simulation: pd.DataFrame, weather_race: pd.DataFrame,
                    schedule: pd.DataFrame) -> tuple[pd.DataFrame, list[JoinCheck]]:
    """The cross-domain table: one row per team per race, all five sources joined."""
    team = driver_race.groupby(TEAM_KEYS + ["constructor_name", "lineage"], as_index=False).agg(
        points=("points", "sum"), sprint_points=("sprint_points", "sum"),
        best_position=("position", "min"), win=("win", "max"), podiums=("podium", "sum"),
        dnf=("dnf", "sum"), mech_dnf=("mech_dnf", "sum"), mean_grid=("grid_clean", "mean"),
        best_grid=("grid_clean", "min"), positions_gained=("positions_gained", "mean"),
        pace_delta_pct=("pace_delta_pct", "mean"), lap_std_s=("lap_std_s", "mean"),
        deg_slope_s_per_lap=("deg_slope_s_per_lap", "mean"), n_stops=("n_stops", "sum"),
        pit_median_s=("pit_median_s", "mean"), pit_delta_s=("pit_delta_s", "mean"),
        cars=("driver_id", "nunique"))
    checks: list[JoinCheck] = []

    spend = financial.pivot_table(index=TEAM_KEYS, columns="component", values="spend_musd_adj",
                                  aggfunc="sum").add_prefix("spend_").add_suffix("_musd")
    spend["spend_total_musd"] = spend.sum(axis=1)
    for comp in ("aero", "power_unit"):
        col = f"spend_{comp}_musd"
        if col in spend:
            spend[f"{comp}_share"] = spend[col] / spend["spend_total_musd"]
    out = team.merge(spend.reset_index(), how="left", on=TEAM_KEYS)
    checks.append(JoinCheck("team x race <- financial (SYNTHETIC)", len(out), int(out["spend_total_musd"].notna().sum())))

    log_cols = [c for c in logistics.columns if c not in TEAM_KEYS + ["data_origin", "weight_unit", "distance_unit"]]
    out = out.merge(logistics[TEAM_KEYS + log_cols], how="left", on=TEAM_KEYS)
    checks.append(JoinCheck("team x race <- logistics (SYNTHETIC)", len(out), int(out["travel_km"].notna().sum())))

    sim_cols = [c for c in simulation.columns if c not in TEAM_KEYS + ["data_origin"]]
    out = out.merge(simulation[TEAM_KEYS + sim_cols], how="left", on=TEAM_KEYS)
    checks.append(JoinCheck("team x race <- simulation (SYNTHETIC)", len(out), int(out["atr_allocation_pct"].notna().sum())))

    out = out.merge(weather_race, how="left", on=RACE_KEYS)
    checks.append(JoinCheck("team x race <- weather", len(out), int(out["track_temp_c"].notna().sum())))

    sched = schedule[RACE_KEYS + ["race_name", "circuit_id", "country", "date"]].copy()
    sched["date"] = pd.to_datetime(sched["date"])
    out = out.merge(sched, how="left", on=RACE_KEYS).sort_values(TEAM_KEYS).reset_index(drop=True)

    grp = out.groupby(["season", "constructor_id"])
    out["cum_points"] = grp["points"].cumsum()
    out["cum_spend_musd"] = grp["spend_total_musd"].cumsum()
    out["points_roll3"] = grp["points"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    return out, checks


def build_component(financial: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Team x round x component: SYNTHETIC spend next to REAL failures of that component."""
    keys = TEAM_KEYS + ["component"]
    spend = financial.groupby(keys, as_index=False).agg(
        spend_musd=("spend_musd_adj", "sum"), spend_outlier=("spend_outlier", "max"),
        spend_imputed=("spend_imputed", "max"))
    fails = (results[results["mech_dnf"]].groupby(keys).size().rename("failures").reset_index())
    out = spend.merge(fails, how="left", on=keys)
    out["failures"] = out["failures"].fillna(0).astype(int)
    out["lineage"] = out["constructor_id"].map(config.TEAM_LINEAGE)
    return out


def build_team_season(team_race: pd.DataFrame, constructor_standings: pd.DataFrame) -> pd.DataFrame:
    """Season roll-up per team, joined to the real final championship position."""
    mean_cols = [c for c in (
        "pace_delta_pct", "lap_std_s", "deg_slope_s_per_lap", "pit_median_s", "pit_delta_s", "mean_grid",
        "aero_share", "power_unit_share", "customs_delay_h", "spares_coverage_pct", "crew_fatigue_idx",
        "setup_hours", "freight_tonnes", "travel_km", "atr_allocation_pct", "sim_hours",
        "sim_track_corr_pct", "setup_iterations") if c in team_race]
    sum_cols = [c for c in ("points", "sprint_points", "win", "podiums", "dnf", "mech_dnf", "cars",
                            "spend_total_musd", "late_parts", "wind_tunnel_runs", "cfd_items",
                            "upgrade_count") if c in team_race]
    agg = {c: (c, "mean") for c in mean_cols} | {c: (c, "sum") for c in sum_cols}
    out = team_race.groupby(["season", "constructor_id", "constructor_name", "lineage"], as_index=False).agg(
        races=("round", "nunique"), **agg)
    out = out.rename(columns={"win": "wins"})
    out["total_points"] = out["points"] + out["sprint_points"]
    out["dnf_rate"] = out["dnf"] / out["cars"]
    out["mech_dnf_rate"] = out["mech_dnf"] / out["cars"]
    out["finish_rate"] = 1 - out["dnf_rate"]
    out["points_per_musd"] = out["total_points"] / out["spend_total_musd"]
    stand = constructor_standings.rename(columns={"position": "championship_position",
                                                  "points": "official_points"})
    out = out.merge(stand[["season", "constructor_id", "championship_position", "official_points"]],
                    how="left", on=["season", "constructor_id"])
    return out.sort_values(["season", "championship_position"]).reset_index(drop=True)


def build_races(schedule: pd.DataFrame, weather_race: pd.DataFrame, driver_race: pd.DataFrame) -> pd.DataFrame:
    """One row per race: calendar, weather and field-level statistics."""
    field = driver_race.groupby(RACE_KEYS, as_index=False).agg(
        starters=("driver_id", "nunique"), dnfs=("dnf", "sum"), mech_dnfs=("mech_dnf", "sum"),
        field_lap_median_s=("lap_median_s", "median"), field_pit_median_s=("pit_median_s", "median"))
    field["dnf_rate"] = field["dnfs"] / field["starters"]
    out = schedule.merge(field, how="inner", on=RACE_KEYS).merge(weather_race, how="left", on=RACE_KEYS)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(RACE_KEYS).reset_index(drop=True)
