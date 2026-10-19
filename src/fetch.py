"""Network layer: download REAL Formula 1 data and cache it under ``data/raw``.

Two public sources are used:

* **Jolpica** (the maintained successor to the Ergast API) for the calendar,
  race and sprint results, pit stops and final standings.
* **FastF1** for lap-by-lap timing, tyre stints and weather samples.

Everything is cached and the download is resumable: re-running only fetches what
is missing. Nothing in the app calls this module at import time; if it has never
been run (or the network is down) ``src.sources`` falls back to ``data/sample``.

Usage::

    python -m src.fetch                      # all seasons, 6 races of lap data each
    python -m src.fetch --laps-per-season 0  # every race (~150 sessions, see note)

FastF1 limits itself to 500 API calls per hour (about 45 race sessions). When
the limit is hit the run stops cleanly; run the same command again an hour
later and it carries on from where it stopped.
    python -m src.fetch --make-sample        # rebuild the small bundled sample
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src import config

log = logging.getLogger("f1sb.fetch")

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
PAGE_SIZE = 100
MIN_INTERVAL_S = 0.4  # stay well inside Jolpica's 4 requests/second burst limit
_last_call = 0.0


class FetchError(RuntimeError):
    """Raised when a remote source cannot be reached after retries."""


# ----------------------------------------------------------------------- Jolpica
def _cache_path(endpoint: str, offset: int) -> Path:
    safe = endpoint.strip("/").replace("/", "_")
    return config.RAW_DIR / "jolpica" / f"{safe}__{offset}.json"


def _get_json(endpoint: str, offset: int = 0, retries: int = 5) -> dict[str, Any]:
    """GET one page from Jolpica, using the on-disk cache when present."""
    global _last_call
    path = _cache_path(endpoint, offset)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    url = f"{JOLPICA_BASE}/{endpoint.strip('/')}.json"
    for attempt in range(retries):
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        try:
            resp = requests.get(url, params={"limit": PAGE_SIZE, "offset": offset}, timeout=30)
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise FetchError(f"network error for {url}: {exc}") from exc
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 429:
            time.sleep(float(resp.headers.get("Retry-After", 5 * (attempt + 1))))
            continue
        if resp.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        if resp.status_code != 200:
            raise FetchError(f"{url} returned HTTP {resp.status_code}")
        payload = resp.json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload
    raise FetchError(f"gave up on {url} after {retries} attempts")


def _paged(endpoint: str, table: str, key: str) -> list[dict[str, Any]]:
    """Collect every page of an endpoint and return the concatenated item list."""
    items: list[dict[str, Any]] = []
    offset = 0
    while True:
        data = _get_json(endpoint, offset)["MRData"]
        items.extend(data[table][key])
        offset += PAGE_SIZE
        if offset >= int(data["total"]):
            return items


def _to_seconds(text: str | None) -> float:
    """Convert ``'1:23.456'`` or ``'23.456'`` to seconds (NaN when unparsable)."""
    if not text:
        return float("nan")
    try:
        parts = [float(p) for p in str(text).split(":")]
    except ValueError:
        return float("nan")
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def fetch_schedule(season: int) -> pd.DataFrame:
    """Race calendar with circuit coordinates for one season."""
    races = _paged(f"{season}", "RaceTable", "Races")
    return pd.DataFrame(
        {
            "season": int(r["season"]),
            "round": int(r["round"]),
            "race_name": r["raceName"],
            "circuit_id": r["Circuit"]["circuitId"],
            "circuit_name": r["Circuit"]["circuitName"],
            "country": r["Circuit"]["Location"]["country"],
            "lat": float(r["Circuit"]["Location"]["lat"]),
            "lon": float(r["Circuit"]["Location"]["long"]),
            "date": r["date"],
        }
        for r in races
    )


def _result_rows(races: list[dict[str, Any]], key: str) -> pd.DataFrame:
    rows = []
    for race in races:
        for res in race.get(key, []):
            fastest = res.get("FastestLap", {})
            rows.append(
                {
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "driver_id": res["Driver"]["driverId"],
                    "driver_code": res["Driver"].get("code"),
                    "driver_name": f'{res["Driver"]["givenName"]} {res["Driver"]["familyName"]}',
                    "constructor_id": res["Constructor"]["constructorId"],
                    "constructor_name": res["Constructor"]["name"],
                    "grid": int(res.get("grid", 0) or 0),
                    "position": int(res["position"]) if str(res.get("position", "")).isdigit() else None,
                    "position_text": res.get("positionText"),
                    "points": float(res.get("points", 0) or 0),
                    "laps": int(res.get("laps", 0) or 0),
                    "status": res.get("status"),
                    "race_time_ms": float(res["Time"]["millis"]) if "Time" in res and "millis" in res["Time"] else None,
                    "fastest_lap_rank": int(fastest["rank"]) if str(fastest.get("rank", "")).isdigit() else None,
                    "fastest_lap_s": _to_seconds(fastest.get("Time", {}).get("time")),
                }
            )
    return pd.DataFrame(rows)


def fetch_results(season: int) -> pd.DataFrame:
    """Grand Prix classification for every round of a season."""
    return _result_rows(_paged(f"{season}/results", "RaceTable", "Races"), "Results")


def fetch_sprints(season: int) -> pd.DataFrame:
    """Sprint classification (empty before 2021)."""
    return _result_rows(_paged(f"{season}/sprint", "RaceTable", "Races"), "SprintResults")


def fetch_pitstops(season: int, rnd: int) -> pd.DataFrame:
    """Pit stops for one race. ``duration_s`` is total pit-lane time."""
    races = _paged(f"{season}/{rnd}/pitstops", "RaceTable", "Races")
    rows = [
        {
            "season": season,
            "round": rnd,
            "driver_id": p["driverId"],
            "stop": int(p["stop"]),
            "lap": int(p["lap"]),
            "time_of_day": p.get("time"),
            "duration_s": _to_seconds(p.get("duration")),
        }
        for race in races
        for p in race.get("PitStops", [])
    ]
    return pd.DataFrame(rows)


def fetch_standings(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Final driver and constructor championship tables for a season."""
    d_lists = _paged(f"{season}/driverstandings", "StandingsTable", "StandingsLists")
    c_lists = _paged(f"{season}/constructorstandings", "StandingsTable", "StandingsLists")
    drivers = pd.DataFrame(
        {
            "season": season,
            "driver_id": s["Driver"]["driverId"],
            "position": int(s.get("position", 0) or 0),
            "points": float(s["points"]),
            "wins": int(s["wins"]),
        }
        for lst in d_lists
        for s in lst["DriverStandings"]
    )
    constructors = pd.DataFrame(
        {
            "season": season,
            "constructor_id": s["Constructor"]["constructorId"],
            "position": int(s.get("position", 0) or 0),
            "points": float(s["points"]),
            "wins": int(s["wins"]),
        }
        for lst in c_lists
        for s in lst["ConstructorStandings"]
    )
    return drivers, constructors


# ------------------------------------------------------------------------ FastF1
def _td_seconds(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series).dt.total_seconds()


def fetch_fastf1_race(season: int, rnd: int, purge_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lap timing and weather samples for one race via FastF1.

    Telemetry and race-control messages are skipped: only timing, tyre and
    weather channels are needed, which keeps each session to a few seconds.
    """
    import fastf1

    logging.getLogger("fastf1").setLevel(logging.ERROR)
    cache_dir = config.FASTF1_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    session = fastf1.get_session(season, rnd, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    raw = session.laps
    laps = pd.DataFrame(
        {
            "season": season,
            "round": rnd,
            "event_name": session.event["EventName"],
            "driver_code": raw["Driver"].astype(str),
            "team_name": raw["Team"].astype(str),
            "lap_number": raw["LapNumber"].astype("Int64"),
            "lap_time_s": _td_seconds(raw["LapTime"]),
            "sector1_s": _td_seconds(raw["Sector1Time"]),
            "sector2_s": _td_seconds(raw["Sector2Time"]),
            "sector3_s": _td_seconds(raw["Sector3Time"]),
            "stint": raw["Stint"].astype("Int64"),
            "compound": raw["Compound"].astype(str),
            "tyre_life": raw["TyreLife"].astype("Float64"),
            "fresh_tyre": raw["FreshTyre"].astype("boolean"),
            "track_status": raw["TrackStatus"].astype(str),
            "is_accurate": raw["IsAccurate"].astype("boolean"),
            "pit_in": raw["PitInTime"].notna(),
            "pit_out": raw["PitOutTime"].notna(),
            "lap_start_s": _td_seconds(raw["LapStartTime"]),
            "position": raw["Position"].astype("Float64"),
        }
    ).reset_index(drop=True)

    wx = session.weather_data
    weather = pd.DataFrame(
        {
            "season": season,
            "round": rnd,
            "time_s": _td_seconds(wx["Time"]),
            "air_temp_c": wx["AirTemp"].astype(float),
            "track_temp_c": wx["TrackTemp"].astype(float),
            "humidity_pct": wx["Humidity"].astype(float),
            "pressure_mbar": wx["Pressure"].astype(float),
            "rainfall": wx["Rainfall"].astype(bool),
            "wind_speed_ms": wx["WindSpeed"].astype(float),
        }
    ).reset_index(drop=True)

    if purge_cache:  # the compact parquet files written by the caller are the cache
        fastf1.Cache.clear_cache(str(cache_dir), deep=True)
    return laps, weather


def pick_rounds(rounds: list[int], per_season: int) -> list[int]:
    """Choose ``per_season`` evenly spaced rounds (all of them when 0)."""
    if per_season <= 0 or per_season >= len(rounds):
        return rounds
    idx = np.linspace(0, len(rounds) - 1, per_season).round().astype(int)
    return [rounds[i] for i in sorted(set(idx))]


def lap_plan(rounds_by_season: dict[int, list[int]], per_season: int) -> list[tuple[int, int]]:
    """Order the lap downloads so an interrupted run still covers every season evenly.

    Within a season the rounds are visited in ever finer even spacing (2, then
    4, 6, 8, 12 ... races); across seasons the downloads are interleaved.
    """
    queues: dict[int, list[int]] = {}
    for season, rounds in rounds_by_season.items():
        wanted = set(pick_rounds(rounds, per_season))
        ordered: list[int] = []
        for k in (2, 4, 6, 8, 12, 16, len(rounds)):
            ordered += [r for r in pick_rounds(rounds, k) if r in wanted and r not in ordered]
        queues[season] = ordered
    depth = max((len(q) for q in queues.values()), default=0)
    return [(season, q[i]) for i in range(depth) for season, q in queues.items() if i < len(q)]


# ------------------------------------------------------------------ orchestration
def fetch_all(seasons: list[int], laps_per_season: int = 6, skip_laps: bool = False) -> dict[str, int]:
    """Download everything, writing tidy parquet tables into ``data/raw``."""
    config.ensure_dirs()
    raw = config.RAW_DIR
    frames: dict[str, list[pd.DataFrame]] = {k: [] for k in (
        "schedule", "results", "sprints", "pitstops", "driver_standings", "constructor_standings")}

    for season in seasons:
        log.info("Jolpica: season %s", season)
        schedule = fetch_schedule(season)
        frames["schedule"].append(schedule)
        frames["results"].append(fetch_results(season))
        frames["sprints"].append(fetch_sprints(season))
        drivers, constructors = fetch_standings(season)
        frames["driver_standings"].append(drivers)
        frames["constructor_standings"].append(constructors)
        for rnd in schedule["round"]:
            frames["pitstops"].append(fetch_pitstops(season, int(rnd)))

    counts: dict[str, int] = {}
    for name, parts in frames.items():
        parts = [p for p in parts if not p.empty]
        table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        table.to_parquet(raw / f"{name}.parquet", index=False)
        counts[name] = len(table)

    if not skip_laps:
        from fastf1.exceptions import RateLimitExceededError

        schedule_all = pd.read_parquet(raw / "schedule.parquet")
        results_all = pd.read_parquet(raw / "results.parquet")
        (raw / "laps").mkdir(exist_ok=True)
        (raw / "weather").mkdir(exist_ok=True)
        run = {s: sorted(results_all.loc[results_all["season"] == s, "round"].unique().tolist()) for s in seasons}
        for season, rnd in lap_plan(run, laps_per_season):
            lap_file = raw / "laps" / f"laps_{season}_{rnd:02d}.parquet"
            wx_file = raw / "weather" / f"weather_{season}_{rnd:02d}.parquet"
            if lap_file.exists() and wx_file.exists():
                continue
            name = schedule_all.query("season == @season and round == @rnd")["race_name"].iloc[0]
            try:
                laps, weather = fetch_fastf1_race(season, int(rnd))
            except RateLimitExceededError as exc:
                # FastF1 caps itself at 500 API calls an hour to protect the upstream
                # services. Respect it: stop here and let a later run resume.
                log.warning("FastF1 hourly rate limit reached (%s). Stopping; re-run in an hour to "
                            "continue, already downloaded races are kept.", exc)
                counts["rate_limited"] = 1
                break
            except Exception as exc:  # one bad session must not stop the run
                log.warning("FastF1 failed for %s R%s (%s): %s", season, rnd, name, exc)
                continue
            laps.to_parquet(lap_file, index=False)
            weather.to_parquet(wx_file, index=False)
            log.info("FastF1: %s R%02d %s - %s laps", season, rnd, name, len(laps))
        counts["lap_sessions"] = len(list((raw / "laps").glob("laps_*.parquet")))
    return counts


def make_sample(per_season: int = 2) -> None:
    """Rebuild ``data/sample`` from the raw cache (two lap sessions per season)."""
    raw, sample = config.RAW_DIR, config.SAMPLE_DIR
    sample.mkdir(parents=True, exist_ok=True)
    for name in ("schedule", "results", "sprints", "pitstops", "driver_standings", "constructor_standings"):
        shutil.copy2(raw / f"{name}.parquet", sample / f"{name}.parquet")
    for kind in ("laps", "weather"):
        files = sorted((raw / kind).glob(f"{kind}_*.parquet"))
        by_season: dict[str, list[Path]] = {}
        for f in files:
            by_season.setdefault(f.stem.split("_")[1], []).append(f)
        keep = [fs[i] for fs in by_season.values()
                for i in sorted({0, len(fs) // 2})][: per_season * len(by_season)]
        pd.concat([pd.read_parquet(f) for f in keep], ignore_index=True).to_parquet(
            sample / f"{kind}.parquet", index=False)
    log.info("Sample written to %s", sample)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download real F1 data into data/raw")
    parser.add_argument("--seasons", nargs=2, type=int, default=[config.SEASONS[0], config.SEASONS[-1]],
                        metavar=("FIRST", "LAST"))
    parser.add_argument("--laps-per-season", type=int, default=6,
                        help="races per season to pull lap/weather data for (0 = all)")
    parser.add_argument("--skip-laps", action="store_true", help="Jolpica tables only")
    parser.add_argument("--make-sample", action="store_true", help="rebuild data/sample from data/raw")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.make_sample:
        make_sample()
        return
    try:
        counts = fetch_all(list(range(args.seasons[0], args.seasons[1] + 1)),
                           args.laps_per_season, args.skip_laps)
    except FetchError as exc:
        log.error("Fetch failed: %s", exc)
        log.error("The app will fall back to the bundled sample in data/sample.")
        raise SystemExit(1) from exc
    log.info("Done: %s", counts)


if __name__ == "__main__":
    main()
