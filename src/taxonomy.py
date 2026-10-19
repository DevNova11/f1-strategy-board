"""Shared vocabulary: team-name normalisation and retirement classification."""

from __future__ import annotations

import re

import pandas as pd

from src import config

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_LAPPED = re.compile(r"^\+\d+ laps?$")


def normalise_team(label: object) -> str | None:
    """Map any spelling of a team name onto the canonical constructor id.

    >>> normalise_team("Oracle Red Bull Racing")
    'red_bull'
    >>> normalise_team("HAAS F1 TEAM")
    'haas'
    """
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return None
    key = _NON_ALNUM.sub("", str(label).lower())
    if key in config.TEAM_ALIASES:
        return config.TEAM_ALIASES[key]
    canonical = str(label).strip().lower()
    return canonical if canonical in config.TEAM_LINEAGE else None


def classify_status(status: object) -> tuple[str, str]:
    """Classify a result status into ``(outcome, component)``.

    ``outcome`` is one of ``finished``, ``mechanical``, ``incident``,
    ``unspecified`` (a retirement the source gives no cause for) or ``other``
    (did not start, disqualified, withdrew). ``component`` names the failed
    system for mechanical retirements and is ``none``/``unknown`` otherwise.

    >>> classify_status("+1 Lap")
    ('finished', 'none')
    >>> classify_status("Gearbox")
    ('mechanical', 'gearbox')
    """
    text = "" if status is None or (isinstance(status, float) and pd.isna(status)) else str(status).strip().lower()
    if text in {"finished", "lapped"} or _LAPPED.match(text):
        return "finished", "none"
    if any(word in text for word in config.INCIDENT_KEYWORDS):
        return "incident", "none"
    for component, words in config.COMPONENT_KEYWORDS.items():
        if any(word in text for word in words):
            return "mechanical", component
    if text == "retired":
        return "unspecified", "unknown"
    return "other", "none"
