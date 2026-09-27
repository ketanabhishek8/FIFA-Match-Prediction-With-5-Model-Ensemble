"""Player-level data collection.

CONFIRMED DATA SOURCE (researched Aug 2026): Stefano Leone's "FIFA Complete
Player Dataset" series on Kaggle -- one dataset per game edition, each
containing a `players_XX.csv` (e.g. players_21.csv for FIFA 21), scraped from
sofifa.com. Slugs follow the pattern:

    stefanoleone992/fifa-{edition}-complete-player-dataset

e.g. stefanoleone992/fifa-21-complete-player-dataset -> players_21.csv

Edition-to-year mapping: the paper's "year-specific" rosters (2015-2022, plus
partial 2023) line up directly with FIFA {edition} = year - 2000, e.g.
year 2021 -> FIFA 21 -> players_21.csv. This project uses that 1:1 mapping.

Real column groups confirmed from the dataset (prefixed, matching the paper's
categories almost exactly):
  - Metadata: short_name, long_name, age, height_cm, weight_kg,
    nationality_name, club_name, preferred_foot, player_positions
  - Attacking: attacking_crossing, attacking_finishing, attacking_heading_accuracy,
    attacking_short_passing, attacking_volleys
  - Skill: skill_dribbling, skill_curve, skill_fk_accuracy, skill_long_passing,
    skill_ball_control
  - Movement: movement_acceleration, movement_sprint_speed, movement_agility,
    movement_reactions, movement_balance
  - Power: power_shot_power, power_jumping, power_stamina, power_strength,
    power_long_shots
  - Mentality: mentality_aggression, mentality_interceptions,
    mentality_positioning, mentality_vision, mentality_penalties, mentality_composure
  - Defending: defending_marking_awareness, defending_standing_tackle,
    defending_sliding_tackle
  - Goalkeeping: goalkeeping_diving, goalkeeping_handling, goalkeeping_kicking,
    goalkeeping_positioning, goalkeeping_reflexes

Important modeling note: `nationality_name` alone is NOT the same as "plays for
the senior national team" -- most players with a given nationality never get
called up. The dataset also has `nation_position` (non-null only for players
currently in that edition's national squad) and `nation_jersey_number`. This
project filters on `nation_position.notna()` to build actual national-team
rosters, which is a meaningfully better signal than nationality_name alone for
World Cup squad composition. See clean.filter_national_team_squad().

TODO (Phase 2):
  - Install the Kaggle CLI (`pip install kaggle`, already in requirements.txt)
    and place your API token at ~/.kaggle/kaggle.json (or %USERPROFILE%\\.kaggle\\
    kaggle.json on Windows) -- get it from kaggle.com/settings -> API ->
    "Create New Token".
  - Run `python -m src.data.collect_players` once credentials are set up to
    pull all configured years into data/raw/.
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path

# Columns we retain, grouped per the paper's structure. These are the REAL
# column names from the stefanoleone992 dataset (confirmed via research).
METADATA_COLUMNS = ["short_name", "long_name", "age", "height_cm", "weight_kg",
                     "nationality_name", "club_name", "preferred_foot",
                     "player_positions", "nation_position", "nation_jersey_number"]
ATTACKING_COLUMNS = ["attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
                      "attacking_short_passing", "attacking_volleys"]
SKILL_COLUMNS = ["skill_dribbling", "skill_curve", "skill_fk_accuracy",
                  "skill_long_passing", "skill_ball_control"]
MOVEMENT_COLUMNS = ["movement_acceleration", "movement_sprint_speed", "movement_agility",
                     "movement_reactions", "movement_balance"]
POWER_COLUMNS = ["power_shot_power", "power_jumping", "power_stamina",
                  "power_strength", "power_long_shots"]
MENTALITY_COLUMNS = ["mentality_aggression", "mentality_interceptions", "mentality_positioning",
                      "mentality_vision", "mentality_penalties", "mentality_composure"]
DEFENDING_COLUMNS = ["defending_marking_awareness", "defending_standing_tackle",
                      "defending_sliding_tackle"]
GOALKEEPING_COLUMNS = ["goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking",
                        "goalkeeping_positioning", "goalkeeping_reflexes"]

# EA's headline rating. Long used only to *pick* the squad (the top-N fallback
# in clean.filter_national_team_squad) and then discarded, which left the single
# most informative attribute out of the model entirely. `potential` is NOT here:
# the FC 25 source carries no equivalent column, so it is 100% null for 2025 and
# would make every 2025 profile an imputed value.
RATING_COLUMNS = ["overall"]

# EA's six headline summaries. They are not redundant with the detailed
# attributes below -- EA computes them as position-weighted composites, so they
# encode how EA itself trades those attributes off. Null for goalkeepers in
# every source (keepers get the goalkeeping_* block instead), which is why the
# 80% completeness filter keeps them only just.
AGGREGATE_COLUMNS = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]

# Present in every source including FC 25. `value_eur`, `wage_eur`,
# `international_reputation` and `potential` are deliberately absent from this
# list: the FC 25 database carries none of them, so including any would make
# every 2025 profile an imputed value.
TRAIT_COLUMNS = ["skill_moves", "weak_foot"]

ALL_COLUMNS = (
    METADATA_COLUMNS + RATING_COLUMNS + AGGREGATE_COLUMNS + TRAIT_COLUMNS
    + ATTACKING_COLUMNS + SKILL_COLUMNS + MOVEMENT_COLUMNS + POWER_COLUMNS
    + MENTALITY_COLUMNS + DEFENDING_COLUMNS + GOALKEEPING_COLUMNS
)


def _edition_for_year(year: int) -> int:
    """Map a roster year (e.g. 2021) to its FIFA game edition number (e.g. 21)."""
    return year - 2000


# Editions published after stefanoleone992 stopped, which come from other
# authors with their own schemas. Mapped onto the canonical columns by
# src/data/adapters.py. Kept here rather than in the loop so `players_2025.csv`
# is reproducible from a clean checkout -- it previously had to be built by
# hand, which is how eight mappable columns stayed unmapped without anyone
# noticing.
MODERN_SOURCES = {
    2025: ("nyagami/ea-sports-fc-25-database-ratings-and-stats", "male_players.csv", "FC25"),
    2026: ("justdhia/ea-sports-fc-26-player-ratings", "ea_fc26_players.csv", "FC26"),
}


def _download_modern_edition(year: int, raw_dir, target_csv) -> None:
    """Download a post-FIFA-23 edition and normalize it into the canonical schema."""
    from src.data.adapters import FC25_COLUMN_MAP, FC26_COLUMN_MAP, normalize_modern_players

    slug, member, schema = MODERN_SOURCES[year]
    column_map = {"FC25": FC25_COLUMN_MAP, "FC26": FC26_COLUMN_MAP}[schema]

    print(f"[{year}] downloading {slug} ({schema} schema) ...")
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download", "-d", slug, "-p", str(raw_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[{year}] FAILED: {result.stderr.strip()}")
        return

    zip_path = raw_dir / f"{slug.split('/')[-1]}.zip"
    with zipfile.ZipFile(zip_path) as zf:
        if member not in zf.namelist():
            print(f"[{year}] FAILED: '{member}' not in zip; contents: {zf.namelist()}")
            zip_path.unlink(missing_ok=True)
            return
        raw = pd.read_csv(zf.open(member), low_memory=False)
    zip_path.unlink(missing_ok=True)

    normalize_modern_players(raw, column_map).to_csv(target_csv, index=False)
    print(f"[{year}] saved -> {target_csv} ({len(raw)} players)")


def download_player_data(cfg: dict | None = None) -> None:
    """Download one FIFA-edition dataset per configured year via the Kaggle
    CLI, unzip it, and keep only the players_XX.csv file in data/raw/.

    Requires the `kaggle` package (pip install kaggle) and a valid API token
    at ~/.kaggle/kaggle.json.
    """
    cfg = cfg or load_config()
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    for year in cfg["data"]["years"]:
        edition = _edition_for_year(year)
        slug = f"stefanoleone992/fifa-{edition}-complete-player-dataset"
        target_csv = raw_dir / f"players_{year}.csv"

        if target_csv.exists():
            print(f"[{year}] already downloaded -> {target_csv}")
            continue

        if year in MODERN_SOURCES:
            _download_modern_edition(year, raw_dir, target_csv)
            continue

        print(f"[{year}] downloading {slug} ...")
        # Invoke via `sys.executable -m kaggle` rather than a bare "kaggle":
        # the console script only resolves when the venv is *activated*, so a
        # bare name dies with WinError 2 / FileNotFoundError whenever this is
        # run as `venv/Scripts/python.exe -m src.data.collect_players`.
        result = subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download", "-d", slug, "-p", str(raw_dir)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"[{year}] FAILED: {result.stderr.strip()}")
            print("  (Older editions like FIFA 15-19 may need the legacy filename "
                  "players_YY.csv inside the zip -- check the dataset's file list "
                  "on Kaggle if this keeps failing.)")
            continue

        zip_path = raw_dir / f"fifa-{edition}-complete-player-dataset.zip"
        with zipfile.ZipFile(zip_path) as zf:
            # The zip contains multiple CSVs (players, teams, coaches); we only
            # want the players file for this edition.
            member = f"players_{edition}.csv"
            extracted = member in zf.namelist()
            if extracted:
                zf.extract(member, raw_dir)
                (raw_dir / member).rename(target_csv)
            else:
                print(f"[{year}] FAILED: '{member}' not found in zip; "
                      f"contents: {zf.namelist()}")
                # FIFA 23 dropped the per-edition players_XX.csv in favour of a
                # consolidated 'male_players (legacy).csv' carrying every edition
                # in one file, keyed by a `fifa_version` column. Extract that
                # file and keep the rows where fifa_version == edition to
                # produce the same thing this loop expects.
                print("  (FIFA 23+ ships 'male_players (legacy).csv' with a "
                      "fifa_version column instead -- slice that to "
                      f"fifa_version == {edition} and save as {target_csv.name}.)")
        zip_path.unlink(missing_ok=True)
        # Only claim success if a file actually landed -- this print used to sit
        # outside the if/else and reported "saved" for downloads that extracted
        # nothing.
        if extracted:
            print(f"[{year}] saved -> {target_csv}")


def load_player_data(cfg: dict | None = None) -> pd.DataFrame:
    """Load raw player data from data/raw/ into a single DataFrame, tagging
    each row with its `year` (since the paper builds year-specific profiles)."""
    cfg = cfg or load_config()
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    years = cfg["data"]["years"]

    frames = []
    for year in years:
        candidate = raw_dir / f"players_{year}.csv"
        if candidate.exists():
            df = pd.read_csv(candidate, low_memory=False)
            df["year"] = year
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No player data files found in {raw_dir}. Run download_player_data() "
            "first (needs a Kaggle API token), or manually place players_<year>.csv "
            "files there."
        )

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    download_player_data()

