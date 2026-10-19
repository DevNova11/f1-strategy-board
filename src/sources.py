"""Layer 1 - loaders for the five data sources.

===========================  =========  ==========================================
Source                       Origin     Provider
===========================  =========  ==========================================
1. Track timing & results    REAL       Jolpica (Ergast) + FastF1
2. Weather                   REAL       FastF1 weather channel
3. Financial / cost cap      SYNTHETIC  ``src.synthetic`` (seeded generator)
4. Logistics                 SYNTHETIC  ``src.synthetic`` (seeded generator)
5. Simulation / CFD          SYNTHETIC  ``src.synthetic`` (seeded generator)
===========================  =========  ==========================================

Real data is read from the local cache in ``data/raw`` (filled by
``python -m src.fetch``). When that cache is absent, for example on a machine
with no network, the loaders fall back to the small bundle in ``data/sample``
so the app always runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src import config, synthetic

JOLPICA_TABLES: tuple[str, ...] = (
    "schedule", "results", "sprints", "pitstops", "driver_standings", "constructor_standings",
)


@dataclass
class SourceInfo:
    """Provenance card for one source, shown on the Data Sources page."""

    name: str
    origin: str          # "REAL" or "SYNTHETIC"
    provider: str
    grain: str
    rows: int
    description: str


@dataclass
class SourceBundle:
    """Everything layer 1 hands to the ingest layer."""

    track: dict[str, pd.DataFrame]
    weather: pd.DataFrame
    financial: pd.DataFrame
    logistics: pd.DataFrame
    simulation: pd.DataFrame
    mode: str                                   # "raw" (full cache) or "sample"
    info: list[SourceInfo] = field(default_factory=list)


def data_mode() -> str:
    """Return ``"raw"`` when the downloaded cache exists, otherwise ``"sample"``."""
    return "raw" if (config.RAW_DIR / "results.parquet").exists() else "sample"


def _base_dir(mode: str) -> Path:
    return config.RAW_DIR if mode == "raw" else config.SAMPLE_DIR


def _read_partitioned(kind: str, mode: str) -> pd.DataFrame:
    """Read per-race lap or weather files (raw mode) or the single sample file."""
    if mode == "raw":
        files = sorted((config.RAW_DIR / kind).glob(f"{kind}_*.parquet"))
        if files:
            return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    sample = config.SAMPLE_DIR / f"{kind}.parquet"
    if sample.exists():
        return pd.read_parquet(sample)
    return pd.DataFrame()


def derive_stints(laps: pd.DataFrame) -> pd.DataFrame:
    """Collapse lap rows into one row per tyre stint."""
    if laps.empty:
        return pd.DataFrame(columns=["season", "round", "driver_code", "stint", "compound",
                                     "start_lap", "end_lap", "stint_laps", "fresh_tyre"])
    valid = laps.dropna(subset=["stint"])
    return (
        valid.groupby(["season", "round", "driver_code", "stint"], as_index=False)
        .agg(compound=("compound", "first"), start_lap=("lap_number", "min"),
             end_lap=("lap_number", "max"), stint_laps=("lap_number", "count"),
             fresh_tyre=("fresh_tyre", "first"))
    )


def load_track(mode: str | None = None) -> dict[str, pd.DataFrame]:
    """Source 1 (REAL): calendar, results, sprints, pit stops, laps, stints, standings."""
    mode = mode or data_mode()
    base = _base_dir(mode)
    missing = [t for t in JOLPICA_TABLES if not (base / f"{t}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"No track data found in {base} (missing {missing}). Run `python -m src.fetch` "
            "or restore the bundled files in data/sample."
        )
    track = {name: pd.read_parquet(base / f"{name}.parquet") for name in JOLPICA_TABLES}
    track["laps"] = _read_partitioned("laps", mode)
    track["stints"] = derive_stints(track["laps"])
    return track


def load_weather(mode: str | None = None) -> pd.DataFrame:
    """Source 2 (REAL): per-minute weather samples for each race session."""
    return _read_partitioned("weather", mode or data_mode())


def _synthetic_tables(track: dict[str, pd.DataFrame], mode: str) -> dict[str, pd.DataFrame]:
    """Return the three SYNTHETIC tables, regenerating them when stale.

    The CSVs written to ``data/synthetic`` are a transparency aid: anyone can
    open them and see exactly what was fabricated. A manifest records the seed
    and the size of the real input so a changed input triggers regeneration.
    """
    out_dir = config.SYNTHETIC_DIR / ("full" if mode == "raw" else "sample")
    manifest_path = out_dir / "manifest.json"
    stamp = {"seed": config.SEED, "mode": mode, "result_rows": int(len(track["results"])),
             "pitstop_rows": int(len(track["pitstops"]))}
    names = ("financial", "logistics", "simulation")
    files = {n: out_dir / f"SYNTHETIC_{n}.csv" for n in names}
    if manifest_path.exists() and all(f.exists() for f in files.values()):
        if json.loads(manifest_path.read_text(encoding="utf-8")) == stamp:
            return {n: pd.read_csv(f) for n, f in files.items()}

    tables = synthetic.generate_all(track["results"], track["schedule"], track["pitstops"],
                                    track["constructor_standings"], seed=config.SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(files[name], index=False)
    manifest_path.write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    # Round-trip through CSV so a fresh build and a cached build are identical.
    return {n: pd.read_csv(f) for n, f in files.items()}


def load_financial(track: dict[str, pd.DataFrame] | None = None, mode: str | None = None) -> pd.DataFrame:
    """Source 3 (SYNTHETIC): cost-cap ledger per team, round and component."""
    mode = mode or data_mode()
    return _synthetic_tables(track or load_track(mode), mode)["financial"]


def load_logistics(track: dict[str, pd.DataFrame] | None = None, mode: str | None = None) -> pd.DataFrame:
    """Source 4 (SYNTHETIC): freight, customs and crew log per team and round."""
    mode = mode or data_mode()
    return _synthetic_tables(track or load_track(mode), mode)["logistics"]


def load_simulation(track: dict[str, pd.DataFrame] | None = None, mode: str | None = None) -> pd.DataFrame:
    """Source 5 (SYNTHETIC): wind-tunnel, CFD and simulator log per team and round."""
    mode = mode or data_mode()
    return _synthetic_tables(track or load_track(mode), mode)["simulation"]


def load_all() -> SourceBundle:
    """Load all five sources and attach provenance cards."""
    mode = data_mode()
    track = load_track(mode)
    weather = load_weather(mode)
    synth = _synthetic_tables(track, mode)
    lap_races = track["laps"][["season", "round"]].drop_duplicates().shape[0] if not track["laps"].empty else 0
    info = [
        SourceInfo("Track timing & results", "REAL", "Jolpica (Ergast) API + FastF1",
                   "driver x race, lap, pit stop, stint",
                   int(len(track["results"]) + len(track["laps"]) + len(track["pitstops"])),
                   f"Race and sprint classification, pit stops and standings for every round; "
                   f"lap times and tyre stints for {lap_races} races."),
        SourceInfo("Weather", "REAL", "FastF1 weather channel", "one sample per minute per race",
                   int(len(weather)), "Air and track temperature, humidity, pressure, wind and rainfall."),
        SourceInfo("Financial / cost cap", "SYNTHETIC", "src.synthetic (seed %d)" % config.SEED,
                   "team x round x component", int(len(synth["financial"])),
                   "Fabricated spend ledger in mixed currencies and scales."),
        SourceInfo("Logistics", "SYNTHETIC", "src.synthetic (seed %d)" % config.SEED,
                   "team x round", int(len(synth["logistics"])),
                   "Fabricated freight, customs, spares and crew-fatigue log. "
                   "Travel distance is computed from real coordinates."),
        SourceInfo("Simulation / CFD", "SYNTHETIC", "src.synthetic (seed %d)" % config.SEED,
                   "team x round", int(len(synth["simulation"])),
                   "Fabricated wind-tunnel, CFD and simulator log. Aero testing allocation "
                   "follows the real sliding-scale rule."),
    ]
    return SourceBundle(track=track, weather=weather, financial=synth["financial"],
                        logistics=synth["logistics"], simulation=synth["simulation"],
                        mode=mode, info=info)
