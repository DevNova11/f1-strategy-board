"""Feedback loop: anomalies flagged in the dashboard return to the EDA layer.

An analyst reviewing the anomaly table in the dashboard can send rows to a
**watchlist**. The watchlist is a plain CSV in ``data/watchlist`` so it is easy
to inspect, version and share. On the next pass the EDA layer reads it back:

* ``src.eda.outliers.annotate_watchlist`` rings watchlisted rows on charts;
* ``apply_watchlist(..., mode="exclude")`` re-runs any analysis without them;
* ``src.eda.report`` lists the watchlist in the generated report.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import config

COLUMNS: list[str] = ["key", "table", "reason", "votes", "note", "source", "added_at"]


def watchlist_path() -> Path:
    """Location of the watchlist CSV."""
    return config.WATCHLIST_DIR / "watchlist.csv"


def load_watchlist(path: Path | None = None) -> pd.DataFrame:
    """Read the watchlist (an empty, correctly typed frame if none exists yet)."""
    path = path or watchlist_path()
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={"key": str, "note": str}).reindex(columns=COLUMNS)


def add_to_watchlist(rows: pd.DataFrame, note: str = "", source: str = "dashboard", table: str = "team_race",
                     key: str = "team_race_key", path: Path | None = None) -> pd.DataFrame:
    """Append flagged rows to the watchlist, keeping the newest entry per key."""
    path = path or watchlist_path()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new = pd.DataFrame({
        "key": rows[key].astype(str).to_numpy(),
        "table": table,
        "reason": rows["anomaly_reason"].to_numpy() if "anomaly_reason" in rows else "",
        "votes": rows["anomaly_votes"].to_numpy() if "anomaly_votes" in rows else 0,
        "note": note,
        "source": source,
        "added_at": now,
    })
    existing = load_watchlist(path)
    combined = new if existing.empty else pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates("key", keep="last").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def remove_from_watchlist(keys: list[str], path: Path | None = None) -> pd.DataFrame:
    """Drop the given keys from the watchlist."""
    path = path or watchlist_path()
    remaining = load_watchlist(path)
    remaining = remaining[~remaining["key"].isin(keys)].reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    remaining.to_csv(path, index=False)
    return remaining


def apply_watchlist(frame: pd.DataFrame, watchlist: pd.DataFrame | None = None, key: str = "team_race_key",
                    mode: str = "flag") -> pd.DataFrame:
    """Feed the watchlist back into an analysis table.

    ``mode="flag"`` adds an ``on_watchlist`` column; ``mode="exclude"`` drops
    the watchlisted rows so an analysis can be re-run without them.
    """
    watchlist = load_watchlist() if watchlist is None else watchlist
    keys = set(watchlist["key"].astype(str))
    out = frame.copy()
    out["on_watchlist"] = out[key].astype(str).isin(keys) if key in out else False
    if mode == "exclude":
        return out[~out["on_watchlist"]].reset_index(drop=True)
    return out
