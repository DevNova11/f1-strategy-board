"""SYNTHETIC data generator for the three off-track sources that are not public.

======================================================================
 EVERYTHING PRODUCED BY THIS MODULE IS SYNTHETIC. IT IS NOT REAL DATA.
======================================================================

Formula 1 teams do not publish cost-cap ledgers, freight manifests or wind
tunnel / CFD logs. To let the analytics platform be built and demonstrated end
to end, this module fabricates those three sources with a seeded random
generator. Every row carries ``data_origin == "SYNTHETIC"``.

How realism is achieved
-----------------------
The generator is *conditioned on the real results*: it reads the real
classification, calendar and pit-stop tables and plants loose, noisy
relationships with them (see ``PLANTED_EFFECTS``). Publicly known rules are
respected where they exist: the cost-cap level per season and the sliding aero
testing allocation by previous championship position.

What that means for interpretation
----------------------------------
Any relationship the EDA finds between a synthetic factor and a real outcome
exists *because it was planted here*. Rankings that involve synthetic factors
demonstrate that the method recovers known structure; they are not evidence
about how real teams operate.

Deliberate mess
---------------
The outputs mimic three separately maintained systems: different column names,
team spellings, currencies, units, ~3% missing values, a few gross outliers and
some duplicated ledger rows. ``src.ingest`` and ``src.clean`` have to fix these.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.taxonomy import classify_status

ORIGIN = "SYNTHETIC"

# Published cost-cap baseline in USD millions (no cap before 2021).
COST_CAP_MUSD: dict[int, float | None] = {
    2019: None, 2020: None, 2021: 145.0, 2022: 140.0, 2023: 135.0, 2024: 135.0, 2025: 135.0,
}
# Fixed, documented conversion rates used both to create and to harmonise spend.
FX_TO_USD: dict[str, float] = {"USD": 1.00, "EUR": 1.10, "GBP": 1.27}
TEAM_CURRENCY: dict[str, str] = {
    "ferrari": "EUR", "rb": "EUR", "sauber": "EUR", "haas": "USD",
}  # every other team reports in GBP
BASE_SHARES: dict[str, float] = {
    "aero": 0.27, "power_unit": 0.21, "chassis": 0.13, "suspension": 0.08, "gearbox": 0.07,
    "brakes": 0.04, "electronics": 0.06, "hydraulics_cooling": 0.04, "operations": 0.10,
}

# The ground truth the insight layer should be able to recover.
PLANTED_EFFECTS: list[dict[str, str]] = [
    {"factor": "spend_total_musd", "target": "points", "direction": "+",
     "mechanism": "Season budget scales with (mostly prior-season) points share."},
    {"factor": "aero_share", "target": "points", "direction": "+",
     "mechanism": "Stronger teams tilt spend towards aerodynamics."},
    {"factor": "upgrade_count", "target": "points", "direction": "+",
     "mechanism": "Upgrade bursts are more likely just before a real upturn in points."},
    {"factor": "late_parts", "target": "dnf", "direction": "+",
     "mechanism": "Late parts are more frequent at rounds with a real technical retirement "
                  "(cause-unknown retirements count half)."},
    {"factor": "spares_coverage_pct", "target": "dnf", "direction": "-",
     "mechanism": "Coverage falls with late parts and rises with team strength."},
    {"factor": "crew_fatigue_idx", "target": "pit_delta_s", "direction": "+",
     "mechanism": "Fatigue rises with back-to-backs, travel and slower real pit stops."},
    {"factor": "sim_track_corr_pct", "target": "grid", "direction": "-",
     "mechanism": "Simulator correlation is higher when real grid slots beat the team norm."},
    {"factor": "atr_allocation_pct", "target": "points", "direction": "-",
     "mechanism": "Real rule: better previous championship position means less aero testing."},
    {"factor": "component spend", "target": "component failure", "direction": "-",
     "mechanism": "A component is under-funded before it fails and over-funded just after."},
]


def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def build_context(results: pd.DataFrame, schedule: pd.DataFrame, pitstops: pd.DataFrame) -> pd.DataFrame:
    """Summarise the REAL data the generator conditions on.

    Returns one row per ``(season, round, constructor_id)`` with points, mean
    grid slot, mechanical retirements, pit-stop pace relative to the field and
    a 0-1 ``strength`` score (share of the season's best team's points).
    """
    res = results.copy()
    classified = res["status"].map(classify_status)
    res["mech_dnf"] = [c[0] == "mechanical" for c in classified]
    res["unspec_dnf"] = [c[0] == "unspecified" for c in classified]
    res["component"] = [c[1] for c in classified]
    res["lineage"] = res["constructor_id"].map(config.TEAM_LINEAGE)

    ctx = (
        res.groupby(["season", "round", "constructor_id", "constructor_name", "lineage"], as_index=False)
        .agg(points=("points", "sum"), mean_grid=("grid", "mean"), mech_dnf=("mech_dnf", "sum"),
             unspec_dnf=("unspec_dnf", "sum"))
    )
    failed = (
        res[res["mech_dnf"]].groupby(["season", "round", "constructor_id"])["component"]
        .agg(lambda s: ",".join(sorted(set(s)))).rename("failed_components").reset_index()
    )
    ctx = ctx.merge(failed, how="left", on=["season", "round", "constructor_id"])
    ctx["failed_components"] = ctx["failed_components"].fillna("")

    season_pts = ctx.groupby(["season", "lineage"])["points"].sum().rename("season_points").reset_index()
    season_pts["strength"] = season_pts["season_points"] / season_pts.groupby("season")["season_points"].transform("max")
    prev = season_pts.assign(season=season_pts["season"] + 1)[["season", "lineage", "strength"]]
    season_pts = season_pts.merge(prev.rename(columns={"strength": "prev_strength"}), how="left", on=["season", "lineage"])
    season_pts["prev_strength"] = season_pts["prev_strength"].fillna(season_pts["strength"])
    ctx = ctx.merge(season_pts[["season", "lineage", "strength", "prev_strength"]], on=["season", "lineage"])

    # Real pit-lane pace relative to the field in the same race (z-score).
    if not pitstops.empty:
        pit = pitstops.merge(res[["season", "round", "driver_id", "constructor_id"]],
                             on=["season", "round", "driver_id"], how="left")
        pit = pit[pit["duration_s"].between(10, 60)]
        team_pit = pit.groupby(["season", "round", "constructor_id"])["duration_s"].median().rename("pit").reset_index()
        grp = team_pit.groupby(["season", "round"])["pit"]
        team_pit["pit_z"] = ((team_pit["pit"] - grp.transform("mean")) / grp.transform("std")).clip(-3, 3)
        ctx = ctx.merge(team_pit[["season", "round", "constructor_id", "pit_z"]], how="left",
                        on=["season", "round", "constructor_id"])
    else:
        ctx["pit_z"] = 0.0
    ctx["pit_z"] = ctx["pit_z"].fillna(0.0)

    sched = schedule.copy()
    sched["date"] = pd.to_datetime(sched["date"])
    sched = sched.sort_values(["season", "round"])
    sched["days_since_prev"] = sched.groupby("season")["date"].diff().dt.days
    sched["back_to_back"] = (sched["days_since_prev"] <= 8).astype(int)
    ctx = ctx.merge(sched[["season", "round", "lat", "lon", "back_to_back"]], on=["season", "round"], how="left")
    ctx["n_rounds"] = ctx.groupby("season")["round"].transform("max")
    return ctx.sort_values(["season", "constructor_id", "round"]).reset_index(drop=True)


def _add_upgrade_bursts(ctx: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Flag ~3 upgrade rounds per team-season, biased towards real upturns."""
    out = []
    for _, grp in ctx.groupby(["season", "constructor_id"], sort=False):
        pts = grp["points"].to_numpy(dtype=float)
        ahead = pd.Series(pts[::-1]).rolling(3, min_periods=1).mean().to_numpy()[::-1]
        behind = pd.Series(pts).shift(1).rolling(3, min_periods=1).mean().fillna(pts.mean()).to_numpy()
        gain = ahead - behind
        z = (gain - gain.mean()) / (gain.std() + 1e-9)
        prob = np.exp(1.2 * z)
        prob /= prob.sum()
        k = min(len(grp), int(rng.integers(2, 5)))
        flag = np.zeros(len(grp), dtype=int)
        flag[rng.choice(len(grp), size=k, replace=False, p=prob)] = 1
        out.append(grp.assign(upgrade_round=flag))
    return pd.concat(out, ignore_index=True)


def generate_financial(ctx: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """SYNTHETIC cost-cap ledger: spend per team, round and car component."""
    components = list(BASE_SHARES)
    base = np.array([BASE_SHARES[c] for c in components])
    rows: list[pd.DataFrame] = []
    for (season, team), grp in ctx.groupby(["season", "constructor_id"], sort=False):
        grp = grp.sort_values("round")
        n = len(grp)
        blend = 0.6 * grp["prev_strength"].iloc[0] + 0.4 * grp["strength"].iloc[0]
        cap = COST_CAP_MUSD[int(season)]
        if cap is None:
            budget = 140 + 290 * blend + rng.normal(0, 12)
        else:
            budget = cap * float(np.clip(0.80 + 0.19 * blend + rng.normal(0, 0.02), 0.76, 1.012))

        rounds = grp["round"].to_numpy()
        weights = (1.25 - 0.5 * rounds / rounds.max() + 0.6 * grp["upgrade_round"].to_numpy()) \
            * rng.lognormal(0, 0.12, n)
        round_spend = budget * weights / weights.sum()

        shares = rng.dirichlet(base * 80)
        shares[0] *= 1 + 0.25 * grp["strength"].iloc[0]      # strong teams lean on aero
        mult = np.ones((n, len(components)))
        for i, failed in enumerate(grp["failed_components"]):
            for comp in filter(None, failed.split(",")):
                if comp in components:
                    j = components.index(comp)
                    mult[: i + 1, j] *= 0.80                 # under-funded before failing
                    mult[i + 1: i + 3, j] *= 1.9             # reactive fix afterwards
        mix = shares * mult * rng.lognormal(0, 0.10, (n, len(components)))
        mix /= mix.sum(axis=1, keepdims=True)
        spend = mix * round_spend[:, None]

        currency = TEAM_CURRENCY.get(config.TEAM_LINEAGE[team], "GBP")
        rows.append(pd.DataFrame({
            "Year": season,
            "GP": np.repeat(rounds, len(components)),
            "Team": grp["constructor_name"].iloc[0].upper(),
            "Component": np.tile(components, n),
            "Spend": (spend / FX_TO_USD[currency]).ravel(),
            "Currency": currency,
            "Scale": "M",
        }))
    ledger = pd.concat(rows, ignore_index=True)

    thousands = rng.random(len(ledger)) < 0.10            # some rows booked in thousands
    ledger.loc[thousands, "Spend"] *= 1000
    ledger.loc[thousands, "Scale"] = "K"
    outlier = rng.random(len(ledger)) < 0.008             # keying errors
    ledger.loc[outlier, "Spend"] *= rng.uniform(6, 12, outlier.sum())
    ledger.loc[rng.random(len(ledger)) < 0.03, "Spend"] = np.nan
    dupes = ledger.sample(frac=0.005, random_state=int(rng.integers(1_000_000)))
    ledger = pd.concat([ledger, dupes], ignore_index=True)
    ledger["Spend"] = ledger["Spend"].round(4)
    ledger["data_origin"] = ORIGIN
    return ledger


def generate_logistics(ctx: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """SYNTHETIC freight and crew log per team and round.

    Travel distance is the one real-derived column: the great-circle distance
    from the team's headquarters to the real circuit coordinates.
    """
    n = len(ctx)
    hq = ctx["lineage"].map(config.TEAM_HQ)
    km = np.array([_haversine_km(h[0], h[1], lat, lon) for h, lat, lon in zip(hq, ctx["lat"], ctx["lon"])])
    flyaway = (km > 2500).astype(int)
    b2b = ctx["back_to_back"].to_numpy()
    strength = ctx["strength"].to_numpy()
    # From 2023 the public feed logs most retirements as plain "Retired" with no
    # cause, so those count as half a technical failure when planting the link.
    mech = (ctx["mech_dnf"] + 0.5 * ctx["unspec_dnf"]).clip(upper=1).to_numpy()

    tonnes = np.where(flyaway == 1, rng.normal(38, 4, n), rng.normal(52, 5, n)) + 6 * strength
    customs = rng.gamma(2.0, 3.0, n) * (1 + 1.2 * flyaway + 0.6 * b2b)
    late = rng.poisson(np.exp(-0.7 + 0.5 * b2b + 0.35 * flyaway - 1.2 * strength + 1.0 * mech))
    spares = np.clip(92 - 6 * late + 8 * strength + rng.normal(0, 4, n), 40, 100)
    fatigue = np.clip(35 + 12 * b2b + 6 * ctx["pit_z"].to_numpy() + 0.0012 * km + rng.normal(0, 7, n), 0, 100)
    setup = rng.normal(30, 3, n) + 4 * flyaway + 0.15 * customs
    mode = np.where(flyaway == 0, "road", np.where(rng.random(n) < 0.12, "sea+air", "air"))

    log = pd.DataFrame({
        "season": ctx["season"].to_numpy(),
        "rnd": ctx["round"].to_numpy(),
        "team": ctx["constructor_name"].to_numpy(),
        "freight_weight": tonnes,
        "weight_unit": "t",
        "travel_distance": km,
        "distance_unit": "km",
        "freight_mode": mode,
        "customs_delay_h": customs,
        "late_parts": late,
        "spares_coverage_pct": spares,
        "crew_fatigue_idx": fatigue,
        "setup_hours": setup,
    })
    in_kg = ctx["lineage"].isin(["williams", "sauber"]).to_numpy()      # two teams log kilograms
    log.loc[in_kg, "freight_weight"] *= 1000
    log.loc[in_kg, "weight_unit"] = "kg"
    in_miles = (ctx["lineage"] == "haas").to_numpy()                    # one team logs miles
    log.loc[in_miles, "travel_distance"] /= 1.609344
    log.loc[in_miles, "distance_unit"] = "mi"

    stuck = rng.random(n) < 0.006                                       # freight held for days
    log.loc[stuck, "customs_delay_h"] *= rng.uniform(8, 15, stuck.sum())
    for col in ("customs_delay_h", "crew_fatigue_idx", "spares_coverage_pct", "freight_weight"):
        log.loc[rng.random(n) < 0.03, col] = np.nan
    log["data_origin"] = ORIGIN
    return log.round(3)


def generate_simulation(ctx: pd.DataFrame, constructor_standings: pd.DataFrame,
                        rng: np.random.Generator) -> pd.DataFrame:
    """SYNTHETIC wind-tunnel, CFD and driver-in-loop simulator log per team and round."""
    n = len(ctx)
    prev = constructor_standings.assign(
        season=constructor_standings["season"] + 1,
        lineage=constructor_standings["constructor_id"].map(config.TEAM_LINEAGE),
    )[["season", "lineage", "position"]].rename(columns={"position": "prev_position"})
    pos = ctx.merge(prev, how="left", on=["season", "lineage"])["prev_position"].fillna(10).to_numpy()
    season = ctx["season"].to_numpy()
    atr = np.where(season <= 2020, 100.0,
                   np.where(season == 2021, 90 + 2.5 * (pos - 1), 70 + 5.0 * (pos - 1)))

    strength = ctx["strength"].to_numpy()
    per_round = 6 / ctx["n_rounds"].to_numpy()                          # six testing periods a year
    tunnel = 320 * per_round * atr / 100 * rng.uniform(0.88, 1.0, n)
    cfd = 2000 * per_round * atr / 100 * rng.uniform(0.85, 1.0, n)
    upgrades = rng.poisson(0.25 + 1.2 * ctx["upgrade_round"].to_numpy())
    sim_hours = rng.normal(28, 5, n) + 30 * (0.6 * ctx["prev_strength"].to_numpy() + 0.4 * strength) + 4 * upgrades
    grid_gap = ctx["mean_grid"] - ctx.groupby(["season", "constructor_id"])["mean_grid"].transform("mean")
    corr = np.clip(91 + 6 * strength - 0.35 * grid_gap.to_numpy() + rng.normal(0, 1.5, n), 70, 99.5)
    iterations = rng.poisson(14 + 10 * strength)

    sim = pd.DataFrame({
        "season_year": season,
        "round_no": ctx["round"].to_numpy(),
        "constructor": ctx["constructor_id"].to_numpy(),
        "atr_allocation_pct": atr,
        "wind_tunnel_runs": tunnel,
        "cfd_items": cfd,
        "sim_hours": sim_hours,
        "sim_track_corr_pct": corr,
        "upgrade_count": upgrades,
        "setup_iterations": iterations,
    })
    glitch = rng.random(n) < 0.006                                      # double-counted CFD batches
    sim.loc[glitch, "cfd_items"] *= rng.uniform(4, 7, glitch.sum())
    for col in ("wind_tunnel_runs", "cfd_items", "sim_hours", "sim_track_corr_pct"):
        sim.loc[rng.random(n) < 0.03, col] = np.nan
    sim["data_origin"] = ORIGIN
    return sim.round(3)


def generate_all(results: pd.DataFrame, schedule: pd.DataFrame, pitstops: pd.DataFrame,
                 constructor_standings: pd.DataFrame, seed: int = config.SEED) -> dict[str, pd.DataFrame]:
    """Generate the three SYNTHETIC sources. Same inputs and seed, same output."""
    rng = np.random.default_rng(seed)
    ctx = _add_upgrade_bursts(build_context(results, schedule, pitstops), rng)
    return {
        "financial": generate_financial(ctx, rng),
        "logistics": generate_logistics(ctx, rng),
        "simulation": generate_simulation(ctx, constructor_standings, rng),
    }
