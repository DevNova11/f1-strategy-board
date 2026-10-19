"""Shared fixtures: small hand-built frames, so the unit tests need no network or files."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def results() -> pd.DataFrame:
    """Two races, two teams, two cars each."""
    rows = []
    entries = [("hamilton", "HAM", "mercedes", "Mercedes"), ("russell", "RUS", "mercedes", "Mercedes"),
               ("leclerc", "LEC", "ferrari", "Ferrari"), ("sainz", "SAI", "ferrari", "Ferrari")]
    statuses = {1: ["Finished", "+1 Lap", "Gearbox", "Collision"], 2: ["Finished", "Finished", "Finished", "Retired"]}
    for rnd in (1, 2):
        for i, (driver, code, team, name) in enumerate(entries):
            status = statuses[rnd][i]
            finished = status in ("Finished", "+1 Lap")
            rows.append({
                "season": 2024, "round": rnd, "driver_id": driver, "driver_code": code, "driver_name": driver.title(),
                "constructor_id": team, "constructor_name": name, "grid": 0 if (rnd == 2 and i == 3) else i + 1,
                "position": i + 1 if finished else None, "position_text": str(i + 1) if finished else "R",
                "points": [25.0, 18.0, 15.0, 12.0][i] if finished else 0.0, "laps": 50 if finished else 20,
                "status": status, "race_time_ms": 5_400_000.0 if i == 0 else None,
                "fastest_lap_rank": i + 1, "fastest_lap_s": 90.0 + i,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def laps() -> pd.DataFrame:
    """Twelve laps for one driver in one race, including a pit lap and a safety-car lap."""
    n = 12
    frame = pd.DataFrame({
        "season": 2024, "round": 1, "event_name": "Test Grand Prix", "driver_code": "HAM", "team_name": "Mercedes",
        "lap_number": np.arange(1, n + 1), "lap_time_s": 90.0 + 0.1 * np.arange(n),
        "sector1_s": 30.0, "sector2_s": 30.0, "sector3_s": 30.0, "stint": [1] * 6 + [2] * 6,
        "compound": ["SOFT"] * 6 + ["hard"] * 6, "tyre_life": list(range(1, 7)) * 2, "fresh_tyre": True,
        "track_status": "1", "is_accurate": True, "pit_in": False, "pit_out": False,
        "lap_start_s": 100.0 * np.arange(n), "position": 1.0,
    })
    frame.loc[5, ["pit_in", "lap_time_s"]] = [True, 112.0]      # in lap
    frame.loc[6, "pit_out"] = True                              # out lap
    frame.loc[8, "track_status"] = "4"                          # safety car
    frame.loc[10, "lap_time_s"] = np.nan                        # untimed
    return frame


@pytest.fixture
def weather() -> pd.DataFrame:
    """One weather sample every 250 seconds for race 1, plus a decoy for race 2."""
    race1 = pd.DataFrame({"season": 2024, "round": 1, "time_s": [0.0, 250.0, 500.0, 750.0],
                          "air_temp_c": [20.0, 21.0, 22.0, 23.0], "track_temp_c": [30.0, 31.0, 32.0, 33.0],
                          "humidity_pct": 50.0, "pressure_mbar": 1010.0, "rainfall": [False, False, True, True],
                          "wind_speed_ms": 2.0})
    race2 = race1.assign(round=2, track_temp_c=99.0)
    return pd.concat([race1, race2], ignore_index=True)


@pytest.fixture
def financial_raw() -> pd.DataFrame:
    """A messy SYNTHETIC-style ledger: mixed currency and scale, a gap, a duplicate, an outlier."""
    rows = [{"Year": 2024, "GP": rnd, "Team": "FERRARI", "Component": "Aero", "Spend": 2.0, "Currency": "EUR",
             "Scale": "M", "data_origin": "SYNTHETIC"} for rnd in range(1, 11)]
    frame = pd.DataFrame(rows)
    frame.loc[1, ["Spend", "Scale"]] = [2000.0, "K"]            # same value booked in thousands
    frame.loc[2, "Spend"] = np.nan                              # gap
    frame.loc[3, "Spend"] = 40.0                                # keying error
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)   # duplicate row
    unknown = frame.iloc[[0]].assign(Team="Not A Team")
    return pd.concat([frame, unknown], ignore_index=True)
