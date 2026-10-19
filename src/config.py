"""Central configuration: paths, seeds, palette and domain constants.

Every path is derived from the location of this file, so the project can be
moved or cloned anywhere without edits. Override the raw-data location with the
``F1SB_RAW_DIR`` environment variable if you want the cache elsewhere.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = Path(os.environ.get("F1SB_RAW_DIR", DATA_DIR / "raw"))
# FastF1 keeps its own large HTTP cache. It is scratch space (the compact parquet
# files in data/raw are the real cache), so it lives in the OS temp folder.
FASTF1_CACHE_DIR: Path = Path(os.environ.get("F1SB_FASTF1_CACHE", Path(tempfile.gettempdir()) / "f1sb_fastf1_cache"))
SAMPLE_DIR: Path = DATA_DIR / "sample"
SYNTHETIC_DIR: Path = DATA_DIR / "synthetic"
PROCESSED_DIR: Path = DATA_DIR / "processed"
WATCHLIST_DIR: Path = DATA_DIR / "watchlist"
REPORTS_DIR: Path = ROOT / "reports"
ASSETS_DIR: Path = ROOT / "assets"

# ----------------------------------------------------------------- reproducibility
SEED: int = 42
SEASONS: tuple[int, ...] = tuple(range(2019, 2026))

# ------------------------------------------------------------------------ palette
BG = "#0E0E10"
PANEL = "#17171B"
BORDER = "#2A2A31"
TEXT = "#FFFFFF"
MUTED = "#9A9AA5"
RED = "#E10600"
CYAN = "#00E5FF"
# First eight slots were checked for colour-blind separation on the dark surface
# (adjacent-pair CVD dE >= 8, contrast >= 3:1). The last two neutrals only appear
# when all ten teams are drawn at once, where the legend and hover carry identity.
COLORWAY: list[str] = [
    RED, CYAN, "#FFB800", "#7C5CFF", "#3DDC97",
    "#FF5FA2", "#4D8DFF", "#FF8A3D", "#D5D8DF", "#8A8D93",
]
# Diverging scale for correlations: cyan (negative) -> panel (zero) -> red (positive).
# Ends are deepened from the accent colours so white numbers stay readable on every cell.
DIVERGING: list[list] = [[0.0, "#0B8FA3"], [0.5, "#1B1B20"], [1.0, "#B3121F"]]
SEQUENTIAL: list[list] = [[0.0, "#1B1B20"], [0.5, "#7A0A06"], [1.0, RED]]

# ------------------------------------------------------------------ team metadata
# Jolpica constructor ids are the canonical team key throughout the project.
# ``TEAM_LINEAGE`` collapses rebrands so multi-season trends stay continuous.
TEAM_LINEAGE: dict[str, str] = {
    "mercedes": "mercedes",
    "ferrari": "ferrari",
    "red_bull": "red_bull",
    "mclaren": "mclaren",
    "renault": "alpine",
    "alpine": "alpine",
    "racing_point": "aston_martin",
    "aston_martin": "aston_martin",
    "toro_rosso": "rb",
    "alphatauri": "rb",
    "rb": "rb",
    "alfa": "sauber",
    "sauber": "sauber",
    "haas": "haas",
    "williams": "williams",
}

LINEAGE_LABEL: dict[str, str] = {
    "mercedes": "Mercedes",
    "ferrari": "Ferrari",
    "red_bull": "Red Bull",
    "mclaren": "McLaren",
    "alpine": "Alpine / Renault",
    "aston_martin": "Aston Martin / Racing Point",
    "rb": "RB / AlphaTauri / Toro Rosso",
    "sauber": "Sauber / Alfa Romeo",
    "haas": "Haas",
    "williams": "Williams",
}

# Colour follows the team, never its rank, so a filter never repaints survivors.
TEAM_COLOR: dict[str, str] = {
    "ferrari": COLORWAY[0],
    "mercedes": COLORWAY[1],
    "mclaren": COLORWAY[2],
    "red_bull": COLORWAY[3],
    "aston_martin": COLORWAY[4],
    "alpine": COLORWAY[5],
    "williams": COLORWAY[6],
    "rb": COLORWAY[7],
    "haas": COLORWAY[8],
    "sauber": COLORWAY[9],
}

# Every spelling of a team that appears in any source, normalised to lower-case
# alphanumerics, mapped to the canonical constructor id. Used by ``ingest``.
TEAM_ALIASES: dict[str, str] = {
    "mercedes": "mercedes", "mercedesamgpetronas": "mercedes",
    "ferrari": "ferrari", "scuderiaferrari": "ferrari",
    "redbull": "red_bull", "redbullracing": "red_bull", "oracleredbullracing": "red_bull",
    "mclaren": "mclaren",
    "renault": "renault",
    "alpine": "alpine", "alpinef1team": "alpine", "alpinef1": "alpine",
    "racingpoint": "racing_point",
    "astonmartin": "aston_martin",
    "tororosso": "toro_rosso", "scuderiatororosso": "toro_rosso",
    "alphatauri": "alphatauri", "scuderiaalphatauri": "alphatauri",
    "rb": "rb", "rbf1team": "rb", "racingbulls": "rb", "visacashapprb": "rb",
    "alfa": "alfa", "alfaromeo": "alfa", "alfaromeoracing": "alfa",
    "sauber": "sauber", "kicksauber": "sauber", "stakef1teamkicksauber": "sauber",
    "haas": "haas", "haasf1team": "haas",
    "williams": "williams", "williamsracing": "williams",
}

# Team headquarters (public knowledge) used to derive real travel distances.
TEAM_HQ: dict[str, tuple[float, float]] = {
    "mercedes": (52.022, -1.146),      # Brackley
    "ferrari": (44.532, 10.864),       # Maranello
    "red_bull": (52.008, -0.693),      # Milton Keynes
    "mclaren": (51.345, -0.548),       # Woking
    "alpine": (51.920, -1.390),        # Enstone
    "aston_martin": (52.074, -1.024),  # Silverstone
    "rb": (44.289, 11.883),            # Faenza
    "sauber": (47.304, 8.820),         # Hinwil
    "haas": (52.062, -1.340),          # Banbury
    "williams": (51.617, -1.413),      # Grove
}

# ------------------------------------------------------------- failure taxonomy
# Maps a retirement status string onto the car component held responsible.
# Matching is by lower-case substring, first hit wins.
COMPONENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "power_unit": ("engine", "power unit", "turbo", "ers", "mgu", "exhaust", "power loss",
                   "ignition", "spark", "fuel pressure", "fuel pump", "fuel system", "fuel leak"),
    "gearbox": ("gearbox", "transmission", "driveshaft", "clutch", "differential", "drivetrain"),
    "suspension": ("suspension", "wheel", "steering", "tyre", "puncture", "vibration"),
    "brakes": ("brake",),
    "electronics": ("electrical", "electronics", "battery", "software"),
    "hydraulics_cooling": ("hydraulic", "water", "oil", "cooling", "overheating", "radiator",
                           "pneumatic"),
    "chassis": ("chassis", "undertray", "front wing", "rear wing", "floor", "mechanical",
                "technical"),
}
INCIDENT_KEYWORDS: tuple[str, ...] = (
    "accident", "collision", "spun", "damage", "debris", "crash",
)
COMPONENTS: tuple[str, ...] = (
    "aero", "power_unit", "chassis", "suspension", "gearbox",
    "brakes", "electronics", "hydraulics_cooling", "operations",
)

# Approximate lap-time cost of carrying one extra lap of fuel (seconds). A
# widely used rule of thumb; documented as an assumption in the Methodology page.
FUEL_EFFECT_S_PER_LAP: float = 0.055


def ensure_dirs() -> None:
    """Create every output directory the project writes to."""
    for path in (RAW_DIR, SYNTHETIC_DIR, PROCESSED_DIR, WATCHLIST_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
