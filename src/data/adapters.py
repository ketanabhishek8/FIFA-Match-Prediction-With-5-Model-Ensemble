"""Normalize post-FIFA-23 player datasets into this project's canonical schema.

EA renamed the series after FIFA 23, and stefanoleone992 -- the source every
`players_2015..2024.csv` came from -- stopped publishing after EA Sports FC 24.
The FC 25 and FC 26 datasets that exist are from other authors and use their own
column names, so they need mapping before `collect_players.load_player_data()`
can stack them with the older years.

What the mapping cannot recover, and why it matters:

  * `nation_position` -- neither source carries national-team call-up tags. The
    column is emitted as all-NaN, which sends every nation down
    `clean.filter_national_team_squad()`'s top-N-by-`overall` fallback path.
    That fallback is already load-bearing for unlicensed federations in the
    older years, so this is a difference of degree, not of kind -- but it means
    2025+ squads are proxies throughout, never real call-up lists.
  * `nation_jersey_number` -- absent, emitted as NaN. Unused by the pipeline.
  * goalkeeping_* -- these sources populate GK attributes only for actual
    keepers, where the FIFA-era files gave every outfielder a low value. The
    squad mean therefore reflects only the keepers in a 2025+ profile. Arguably
    more correct, but NOT comparable to how earlier years were built.
"""
from __future__ import annotations

import re

import pandas as pd

from src.data.collect_players import AGGREGATE_COLUMNS

# EA Sports FC 25 -- nyagami/ea-sports-fc-25-database-ratings-and-stats
# (male_players.csv). Chosen over the sofifa-derived FC 25 sets because those
# either ship empty attribute columns or carry national-team membership in place
# of a per-player nationality, which the top-N fallback needs.
FC25_COLUMN_MAP = {
    "Name": "short_name", "Age": "age", "Nation": "nationality_name",
    "Team": "club_name", "Preferred foot": "preferred_foot",
    "Position": "player_positions", "OVR": "overall",
    # EA's six headline summaries, and the two trait ratings. Present here all
    # along and simply never mapped, which left them 100% null for 2025 and so
    # excluded from the model for every year.
    "PAC": "pace", "SHO": "shooting", "PAS": "passing",
    "DRI": "dribbling", "DEF": "defending", "PHY": "physic",
    "Skill moves": "skill_moves", "Weak foot": "weak_foot",
    "Crossing": "attacking_crossing", "Finishing": "attacking_finishing",
    "Heading Accuracy": "attacking_heading_accuracy",
    "Short Passing": "attacking_short_passing", "Volleys": "attacking_volleys",
    "Dribbling": "skill_dribbling", "Curve": "skill_curve",
    "Free Kick Accuracy": "skill_fk_accuracy", "Long Passing": "skill_long_passing",
    "Ball Control": "skill_ball_control",
    "Acceleration": "movement_acceleration", "Sprint Speed": "movement_sprint_speed",
    "Agility": "movement_agility", "Reactions": "movement_reactions",
    "Balance": "movement_balance",
    "Shot Power": "power_shot_power", "Jumping": "power_jumping",
    "Stamina": "power_stamina", "Strength": "power_strength",
    "Long Shots": "power_long_shots",
    "Aggression": "mentality_aggression", "Interceptions": "mentality_interceptions",
    "Positioning": "mentality_positioning", "Vision": "mentality_vision",
    "Penalties": "mentality_penalties", "Composure": "mentality_composure",
    "Def Awareness": "defending_marking_awareness",
    "Standing Tackle": "defending_standing_tackle",
    "Sliding Tackle": "defending_sliding_tackle",
    "GK Diving": "goalkeeping_diving", "GK Handling": "goalkeeping_handling",
    "GK Kicking": "goalkeeping_kicking", "GK Positioning": "goalkeeping_positioning",
    "GK Reflexes": "goalkeeping_reflexes",
}

# EA Sports FC 26 -- justdhia/ea-sports-fc-26-player-ratings (ea_fc26_players.csv),
# which is the combined outfield + goalkeeper file.
FC26_COLUMN_MAP = {
    "commonName": "short_name", "age": "age", "nationality": "nationality_name",
    "team": "club_name", "preferredFoot": "preferred_foot",
    # NOTE: the six aggregate ratings and skill_moves / weak_foot are mapped for
    # FC 25 but not here -- this file's column names have not been checked
    # against the source, and guessing them would silently map nothing while
    # looking like it worked (normalize_modern_players drops absent keys). Add
    # them once 2026 is actually pulled.
    "position": "player_positions", "overallRating": "overall",
    "crossing": "attacking_crossing", "finishing": "attacking_finishing",
    "headingAccuracy": "attacking_heading_accuracy",
    "shortPassing": "attacking_short_passing", "volleys": "attacking_volleys",
    "dribbling": "skill_dribbling", "curve": "skill_curve",
    "freeKickAccuracy": "skill_fk_accuracy", "longPassing": "skill_long_passing",
    "ballControl": "skill_ball_control",
    "acceleration": "movement_acceleration", "sprintSpeed": "movement_sprint_speed",
    "agility": "movement_agility", "reactions": "movement_reactions",
    "balance": "movement_balance",
    "shotPower": "power_shot_power", "jumping": "power_jumping",
    "stamina": "power_stamina", "strength": "power_strength",
    "longShots": "power_long_shots",
    "aggression": "mentality_aggression", "interceptions": "mentality_interceptions",
    "positioning": "mentality_positioning", "vision": "mentality_vision",
    "penalties": "mentality_penalties", "composure": "mentality_composure",
    "defensiveAwareness": "defending_marking_awareness",
    "standingTackle": "defending_standing_tackle",
    "slidingTackle": "defending_sliding_tackle",
    "gkDiving": "goalkeeping_diving", "gkHandling": "goalkeeping_handling",
    "gkKicking": "goalkeeping_kicking", "gkPositioning": "goalkeeping_positioning",
    "gkReflexes": "goalkeeping_reflexes",
}


def _leading_number(value) -> float:
    """First number in a string like `182cm / 6'0\"` or `75kg / 165lb` -> 182.0 / 75.0.

    Returns the value unchanged when it's already numeric, and NaN when nothing
    parses -- these columns feed `height_cm` / `weight_kg`, which are real model
    features, so a silent 0 would be worse than a NaN the imputer can handle.
    """
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else float("nan")


def normalize_modern_players(df: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    """Rename `df` into the canonical schema and fill in what it lacks."""
    out = df.rename(columns=column_map)
    keep = [c for c in column_map.values() if c in out.columns]
    out = out[keep].copy()

    # Height/weight arrive as display strings in some sources ("182cm / 6'0\"").
    for src, dest in (("height", "height_cm"), ("weight", "weight_kg"),
                      ("Height", "height_cm"), ("Weight", "weight_kg")):
        if src in df.columns:
            out[dest] = df[src].map(_leading_number)

    if "long_name" not in out.columns:
        if {"firstName", "lastName"} <= set(df.columns):
            out["long_name"] = (df["firstName"].fillna("") + " " + df["lastName"].fillna("")).str.strip()
        else:
            out["long_name"] = out.get("short_name")

    # The six aggregate ratings mean something different for goalkeepers in
    # these sources: EA reuses the PAC/SHO/PAS/DRI/DEF/PHY slots to show
    # Diving/Handling/Kicking/Reflexes/Speed/Positioning, so Donnarumma ships
    # with pace=90 and dribbling=90. The FIFA-era files leave all six null for
    # keepers instead. Matching that keeps the columns comparable across years,
    # and the keeper-specific information is already carried by goalkeeping_*.
    if "player_positions" in out.columns:
        is_keeper = (out["player_positions"].astype(str)
                     .str.split(",").str[0].str.strip().str.upper() == "GK")
        aggregates = [c for c in AGGREGATE_COLUMNS if c in out.columns]
        out.loc[is_keeper, aggregates] = pd.NA

    # Absent upstream -- see the module docstring. nation_position being all-NaN
    # is what routes every nation to the top-N-by-overall fallback.
    for missing in ("nation_position", "nation_jersey_number"):
        out[missing] = pd.NA

    return out
