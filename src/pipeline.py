"""Orchestrates layers 1-3: sources -> align -> clean -> join -> keys and flags.

``load()`` is what the app, notebook and report call. It returns cached parquet
tables from ``data/processed`` and rebuilds them only when the inputs changed.

Run ``python -m src.pipeline`` to force a rebuild from the command line.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import clean, config, ingest, sources

log = logging.getLogger("f1sb.pipeline")

TABLE_NAMES: tuple[str, ...] = (
    "races", "driver_race", "team_race", "team_season", "component", "laps", "stints", "pitstops",
)
# Team-level operations columns copied onto each driver row for the reliability model.
TEAM_FEATURES_FOR_DRIVERS: tuple[str, ...] = (
    "spend_total_musd", "aero_share", "power_unit_share", "travel_km", "freight_tonnes", "customs_delay_h",
    "late_parts", "spares_coverage_pct", "crew_fatigue_idx", "setup_hours", "atr_allocation_pct",
    "wind_tunnel_runs", "cfd_items", "sim_hours", "sim_track_corr_pct", "upgrade_count", "setup_iterations",
    "air_temp_c", "track_temp_c", "humidity_pct", "wind_speed_ms", "rain_share", "wet_race",
    "race_name", "date",
)


@dataclass
class Tables:
    """The cleaned, joined analysis tables plus build metadata."""

    races: pd.DataFrame
    driver_race: pd.DataFrame
    team_race: pd.DataFrame
    team_season: pd.DataFrame
    component: pd.DataFrame
    laps: pd.DataFrame
    stints: pd.DataFrame
    pitstops: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def frames(self) -> dict[str, pd.DataFrame]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "meta"}


def processed_dir() -> Path:
    """Cache folder for the current data mode (full download or bundled sample)."""
    return config.PROCESSED_DIR / ("full" if sources.data_mode() == "raw" else "sample")


def _stamp() -> dict:
    """Fingerprint of the inputs; a change triggers a rebuild."""
    mode = sources.data_mode()
    base = config.RAW_DIR if mode == "raw" else config.SAMPLE_DIR
    results = base / "results.parquet"
    return {
        "mode": mode,
        "seed": config.SEED,
        "results_mtime": int(results.stat().st_mtime) if results.exists() else 0,
        "lap_files": len(list((config.RAW_DIR / "laps").glob("laps_*.parquet"))) if mode == "raw" else -1,
    }


def build(save: bool = True) -> Tables:
    """Run sources, ingest and clean end to end and (optionally) cache the result."""
    bundle = sources.load_all()
    report = clean.CleaningReport()
    checks: list[ingest.JoinCheck] = []
    track = bundle.track

    # --- layer 2a: schema alignment and timestamp sync
    results = ingest.align_results(track["results"])
    financial, c1 = ingest.align_financial(bundle.financial)
    logistics, c2 = ingest.align_logistics(bundle.logistics)
    simulation, c3 = ingest.align_simulation(bundle.simulation)
    laps, c4 = ingest.align_laps(track["laps"], results)
    pitstops, c5 = ingest.align_pitstops(track["pitstops"], results)
    laps, c6 = ingest.sync_weather_to_laps(laps, bundle.weather)
    checks += [c1, c2, c3, c4, c5, c6]
    aligned = {"results": results, "laps": laps, "pitstops": pitstops, "weather": bundle.weather,
               "financial": financial, "logistics": logistics, "simulation": simulation}
    missing_before = clean.missing_summary(aligned)
    rows_before = {name: len(frame) for name, frame in aligned.items()}

    # --- layer 3a: per-source cleaning
    results = clean.clean_results(results, report)
    laps = clean.clean_laps(laps, report)
    pitstops = clean.clean_pitstops(pitstops, report)
    financial = clean.clean_financial(financial, report)
    logistics = clean.clean_logistics(logistics, report)
    simulation = clean.clean_simulation(simulation, report)
    cleaned = {"results": results, "laps": laps, "pitstops": pitstops, "financial": financial,
               "logistics": logistics, "simulation": simulation}

    # --- layer 2b: joins on (season, round, constructor, driver, component)
    stints = ingest.build_stints(laps)
    weather_race = ingest.summarise_weather(bundle.weather)
    driver_race, c7 = ingest.build_driver_race(results, laps, stints, pitstops, track["sprints"])
    team_race, c8 = ingest.build_team_race(driver_race, financial, logistics, simulation, weather_race,
                                           track["schedule"])
    checks += c7 + c8
    component = ingest.build_component(financial, results)
    team_season = ingest.build_team_season(team_race, track["constructor_standings"])
    races = ingest.build_races(track["schedule"], weather_race, driver_race)
    extra = [c for c in TEAM_FEATURES_FOR_DRIVERS if c in team_race.columns]
    driver_race = driver_race.merge(team_race[ingest.TEAM_KEYS + extra], how="left", on=ingest.TEAM_KEYS)

    # --- layer 3b: feature keys and outlier flags on the joined tables
    driver_race = clean.add_feature_keys(driver_race)
    team_race = clean.add_feature_keys(team_race)
    team_race = clean.flag_outliers(team_race, ["spend_total_musd", "pit_median_s", "customs_delay_h"], ["season"])
    team_season = clean.add_feature_keys(team_season)
    component = clean.add_feature_keys(component)
    races = clean.add_feature_keys(races)
    laps = clean.add_feature_keys(laps)
    stints = clean.add_feature_keys(stints)
    pitstops = clean.add_feature_keys(pitstops)

    lap_races = int(laps[["season", "round"]].drop_duplicates().shape[0]) if not laps.empty else 0
    meta = {
        "mode": bundle.mode,
        "seed": config.SEED,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stamp": _stamp(),
        "seasons": sorted(int(s) for s in results["season"].unique()),
        "races": int(len(races)),
        "races_with_laps": lap_races,
        "sources": [asdict(i) for i in bundle.info],
        "join_checks": [c.to_dict() for c in checks],
        "cleaning_steps": report.steps,
        "missing_before": missing_before.to_dict("records"),
        "missing_after": clean.missing_summary(cleaned).to_dict("records"),
        "rows_before": rows_before,
        "rows_after": {name: len(frame) for name, frame in cleaned.items()},
    }
    tables = Tables(races, driver_race, team_race, team_season, component, laps, stints, pitstops, meta)
    if save:
        out_dir = processed_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.frames().items():
            frame.to_parquet(out_dir / f"{name}.parquet", index=False)
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return tables


def load(force: bool = False) -> Tables:
    """Return the analysis tables, rebuilding only if the inputs changed."""
    meta_path = processed_dir() / "meta.json"
    if not force and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        files = {n: processed_dir() / f"{n}.parquet" for n in TABLE_NAMES}
        if meta.get("stamp") == _stamp() and all(f.exists() for f in files.values()):
            return Tables(**{n: pd.read_parquet(f) for n, f in files.items()}, meta=meta)
    return build(save=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tables = build(save=True)
    log.info("Built from %s data: %s", tables.meta["mode"],
             {name: frame.shape for name, frame in tables.frames().items()})


if __name__ == "__main__":
    main()
