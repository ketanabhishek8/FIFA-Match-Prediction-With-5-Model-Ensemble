"""Cleaning and team-year profile assembly.

Implements the paper's Feature Selection principles (Section III.A.2):
  1. Relevance   -- keep only attributes that quantify offensive/defensive/overall
                    proficiency (finishing, tackling, positioning, etc.)
  2. Completeness -- keep attributes available for a sufficient number of players
                     across teams/years, to avoid sample bias
  3. Non-redundancy -- drop highly correlated attributes to avoid multicollinearity
"""
from __future__ import annotations

import pandas as pd

from src.data.collect_players import (
    ALL_COLUMNS,
    ATTACKING_COLUMNS,
    DEFENDING_COLUMNS,
    GOALKEEPING_COLUMNS,
)

# `nationality_name` values that differ between the FIFA player dataset and
# the international match-results dataset -- normalized here so team-year
# profiles join cleanly against match data (confirmed: FIFA calls it "Korea
# Republic", results.csv calls it "South Korea").
NAME_ALIASES = {
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "China PR": "China",
    "Chinese Taipei": "Taiwan",
    "Brunei Darussalam": "Brunei",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Guinea Bissau": "Guinea-Bissau",
    "Curacao": "Curaçao",
    "São Tomé e Príncipe": "São Tomé and Príncipe",
    "Swaziland": "Eswatini",
    "Holland": "Netherlands",
}


def filter_national_team_squad(df: pd.DataFrame, top_n: int = 23) -> pd.DataFrame:
    """Build a national-squad subset per (nationality, year).

    `nation_position` (non-null = called up) is EA's per-edition national-team
    licensing tag, not a real squad-membership record -- EA only licenses a
    subset of federations per FIFA edition (confirmed: 50 nations in FIFA 18,
    33 in FIFA 22). Unlicensed federations (Croatia, Japan, Nigeria, Senegal,
    etc.) never get a nation_position tag even though their players are real
    internationals, which would silently drop 13 of 40 2018+2022 World Cup
    teams if `nation_position` were used alone (35/128 matches survive vs.
    118/128 with this fallback).

    So: use `nation_position` where a (nationality, year) pair has it (real
    squad list, most accurate), and fall back to a proxy squad picked by
    `overall` rating where it doesn't.

    The fallback fills a positional quota (SQUAD_QUOTA) rather than taking the
    top `top_n` outright. Taking them outright looked reasonable and was not:
    EA rates goalkeepers systematically below outfielders, so a nation's best
    keeper often sits outside its top 23 entirely -- Algeria 2017's is 32nd --
    and 58% of proxy squads ended up containing no goalkeeper at all. That
    skews every squad mean toward attacking attributes, not merely the
    goalkeeping ones, and makes aggregating anything by position meaningless.

    Short groups backfill from the best remaining players of any position, so
    nations with only a handful of rated players still produce a profile.
    """
    df = df.copy()
    df["nationality_name"] = df["nationality_name"].replace(NAME_ALIASES)

    licensed = df[df["nation_position"].notna()]
    licensed_pairs = set(map(tuple, licensed[["nationality_name", "year"]].drop_duplicates().values))
    is_licensed_pair = pd.Series(list(zip(df["nationality_name"], df["year"])), index=df.index).isin(licensed_pairs)

    squad_licensed = df[is_licensed_pair & df["nation_position"].notna()]

    unlicensed_pool = df[~is_licensed_pair].copy()
    unlicensed_pool["_position_group"] = unlicensed_pool["player_positions"].map(_primary_position_group)
    unlicensed_pool = unlicensed_pool.sort_values("overall", ascending=False)

    # Explicit iteration rather than groupby().apply(): from pandas 3 the latter
    # hides the grouping columns from the callable, so the returned rows come
    # back with a null nationality_name/year and are silently dropped later.
    fallback_groups = [_quota_squad(group, top_n) for _, group in
                       unlicensed_pool.groupby(["nationality_name", "year"], sort=False)]
    # concat() rejects an empty list, which is reachable whenever every
    # (nationality, year) pair in the frame is licensed.
    squad_fallback = (
        pd.concat(fallback_groups, ignore_index=True) if fallback_groups
        else unlicensed_pool
    ).drop(columns="_position_group", errors="ignore")

    return pd.concat([squad_licensed, squad_fallback], ignore_index=True)


def _quota_squad(pool: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Best players per position group up to SQUAD_QUOTA, backfilled to `top_n`.

    `pool` must already be sorted by `overall` descending.
    """
    picked = pd.concat([
        pool[pool["_position_group"] == group].head(quota)
        for group, quota in SQUAD_QUOTA.items()
    ])
    if len(picked) < top_n:
        backfill = pool.drop(index=picked.index).head(top_n - len(picked))
        picked = pd.concat([picked, backfill])
    return picked


def clean_player_data(df: pd.DataFrame, completeness_threshold: float = 0.8) -> pd.DataFrame:
    """Drop columns with too many missing values (completeness principle) and
    rows missing critical identifiers.

    Call this AFTER filter_national_team_squad(), not before: nation_position
    is null for ~94% of the full player pool by design (most players are never
    capped), so running this first on the unfiltered pool drops the column
    before the squad filter can use it.

    Args:
        df: raw concatenated player data.
        completeness_threshold: minimum fraction of non-null values required to
            keep a column.
    """
    df = df.dropna(subset=["short_name", "nationality_name", "year"])

    keep_cols = [c for c in df.columns if df[c].isna().mean() <= (1 - completeness_threshold)]
    df = df[keep_cols]

    return df


def drop_redundant_features(df: pd.DataFrame, corr_threshold: float = 0.9) -> pd.DataFrame:
    """Non-redundancy principle: drop one of each pair of numeric columns whose
    pairwise correlation exceeds corr_threshold."""
    import numpy as np

    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr().abs()
    upper_mask = pd.DataFrame(
        np.triu(np.ones(corr.shape), k=1).astype(bool),
        index=corr.index, columns=corr.columns,
    )
    upper = corr.where(upper_mask)

    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    return df.drop(columns=to_drop)


# Primary playing position -> broad group. EA's `player_positions` is a
# comma-separated preference list ("LW, ST"); the first entry is the primary one.
POSITION_GROUPS = {
    "GK": "gk",
    "CB": "def", "LB": "def", "RB": "def", "LWB": "def", "RWB": "def",
    "CDM": "mid", "CM": "mid", "CAM": "mid", "LM": "mid", "RM": "mid",
    "LW": "att", "RW": "att", "ST": "att", "CF": "att",
}
POSITION_ORDER = ("gk", "def", "mid", "att")

# Shape of the proxy squad built for federations EA never licensed. Roughly a
# real World Cup 23: three keepers, a back eight, a midfield eight, four
# forwards. Without a quota the selection is "top 23 by overall", which leaves
# most proxy squads with no goalkeeper -- see filter_national_team_squad.
SQUAD_QUOTA = {"gk": 3, "def": 8, "mid": 8, "att": 4}


def _primary_position_group(player_positions) -> str | None:
    """Broad position group from EA's preference list, or None if unrecognised."""
    if not isinstance(player_positions, str):
        return None
    return POSITION_GROUPS.get(player_positions.split(",")[0].strip().upper())


def _profile_from_squad(squad: pd.DataFrame) -> dict:
    """One team-year profile from an already-selected squad.

    Three kinds of aggregation, because a flat squad mean answers the wrong
    question in two places:

    * **Squad means** over every numeric attribute -- the paper's original
      aggregation, and a fair summary of squad depth.

    * **Shape of the overall rating** (`overall_max`, `overall_top3`,
      `overall_spread`). The mean measures depth, which wins leagues; a single
      knockout match is often decided by the two or three best players on the
      pitch, which the mean erases. Norway 2025 is the worked example: squad
      mean 76.6 against Brazil's 84.4, but a higher best player (Haaland 91 vs
      90) and a top-3 within 1.7 points. By the mean it looks like a mismatch;
      by the shape it looks like the close game it turned out to be.

    * **Position-restricted summaries** (`goalkeeping_core`, `defending_core`,
      `attacking_core`, and `overall_<group>`). Averaging `goalkeeping_*` across
      23 players, ~20 of whom are outfielders EA gave junk keeper ratings,
      measures almost nothing. These are added *alongside* the squad-wide means
      rather than replacing them: a nation whose entire rated pool contains no
      goalkeeper yields NaN here, and that is 27% of match rows -- so
      overwriting would have traded a diluted-but-present signal for a missing
      one on a quarter of the data.

    A group with no players yields NaN, which the pipeline imputes from the
    training-split mean like any other gap.
    """
    numeric_cols = [c for c in ALL_COLUMNS
                    if c in squad.columns and pd.api.types.is_numeric_dtype(squad[c])]
    profile: dict = squad[numeric_cols].mean().to_dict()
    profile["squad_size"] = len(squad)

    groups = squad["player_positions"].map(_primary_position_group) \
        if "player_positions" in squad.columns else pd.Series(index=squad.index, dtype=object)

    if "overall" in squad.columns:
        rated = squad["overall"].dropna()
        profile["overall_max"] = rated.max() if len(rated) else float("nan")
        profile["overall_top3"] = rated.nlargest(3).mean() if len(rated) else float("nan")
        # Population SD: this describes the squad in hand, not a sample from a
        # wider population. Top-heavy squads (Norway 4.9) separate from flat
        # ones (Brazil 2.6).
        profile["overall_spread"] = rated.std(ddof=0) if len(rated) else float("nan")

        for group in POSITION_ORDER:
            members = squad.loc[groups == group, "overall"]
            profile[f"overall_{group}"] = members.mean() if len(members) else float("nan")

    for group, columns, name in (
        ("gk", GOALKEEPING_COLUMNS, "goalkeeping_core"),
        ("def", DEFENDING_COLUMNS, "defending_core"),
        ("att", ATTACKING_COLUMNS, "attacking_core"),
    ):
        members = squad[groups == group]
        cols = [c for c in columns if c in squad.columns]
        profile[name] = (
            members[cols].mean().mean() if len(members) and cols else float("nan")
        )

    return profile


def build_team_year_profile(players_df: pd.DataFrame, team: str, year: int) -> pd.Series:
    """Aggregate individual player rows into a single team-year profile.

    The paper builds *year-specific* team profiles to account for evolving
    rosters -- this is the core "player-to-team" aggregation step. Expects
    `players_df` to already be filtered to actual national-squad players (see
    filter_national_team_squad()) so the aggregation reflects the squad, not
    every player of that nationality. See `_profile_from_squad` for what is
    computed and why.
    """
    squad = players_df[(players_df["nationality_name"] == team) & (players_df["year"] == year)]
    if squad.empty:
        raise ValueError(f"No players found for {team} in {year}.")

    profile = _profile_from_squad(squad)
    profile["team"] = team
    profile["year"] = year
    return pd.Series(profile)


def build_all_team_year_profiles(players_df: pd.DataFrame) -> pd.DataFrame:
    """Build a profile for every (team, year) combination present in the data.

    One groupby pass rather than re-filtering the whole player frame per
    combination, and rows are assembled as dicts so pandas infers a real dtype
    per column -- a list of Series, each carrying a string `team`, produces
    object-dtype columns that silently push boxed Python floats into the scaler.
    """
    rows = []
    for (team, year), squad in players_df.groupby(["nationality_name", "year"], sort=False):
        profile = _profile_from_squad(squad)
        profile["team"] = team
        profile["year"] = year
        rows.append(profile)

    profiles = pd.DataFrame(rows)
    numeric = [c for c in profiles.columns if c != "team"]
    profiles[numeric] = profiles[numeric].apply(pd.to_numeric, errors="coerce")
    return profiles.reset_index(drop=True)
