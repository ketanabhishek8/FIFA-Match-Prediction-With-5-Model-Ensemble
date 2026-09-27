"""Match-level data collection.

CONFIRMED DATA SOURCE (researched Aug 2026): "International football results
from 1872 to [present]" on Kaggle (slug: martj42/international-football-
results-from-1872-to-2017 -- the slug is frozen but the dataset is actively
updated past 2017 despite the name). File: results.csv.

Real, confirmed columns: date, home_team, away_team, home_score, away_score,
tournament, city, country, neutral (bool). `tournament` includes values like
"FIFA World Cup", "FIFA World Cup qualification", "Friendly", continental
championships, etc.

Per the paper (Section IV), the WWR *baseline* is fit specifically on FIFA
World Cup history. This project follows that: filter_world_cup_matches() below
restricts to tournament == "FIFA World Cup" (finals only, not qualifiers) for
both baseline fitting and evaluation, matching the proposal's stated scope.

TODO (Phase 2):
  - Run `python -m src.data.collect_matches` once Kaggle credentials are set
    up (see collect_players.py for the same setup).
"""
from __future__ import annotations

import subprocess
import sys
import zipfile

import pandas as pd

from src.config import load_config, resolve_path

MATCH_DATASET_SLUG = "martj42/international-football-results-from-1872-to-2017"
MATCH_COLUMNS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "city", "country", "neutral",
]


def download_match_data(cfg: dict | None = None) -> None:
    """Download the raw match history dataset via the Kaggle CLI into data/raw/."""
    cfg = cfg or load_config()
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_csv = raw_dir / "results.csv"

    if target_csv.exists():
        print(f"already downloaded -> {target_csv}")
        return

    print(f"downloading {MATCH_DATASET_SLUG} ...")
    # `sys.executable -m kaggle`, not a bare "kaggle" -- the console script is
    # only on PATH when the venv is activated (see collect_players.py).
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download", "-d", MATCH_DATASET_SLUG,
         "-p", str(raw_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download failed: {result.stderr.strip()}")

    zip_path = raw_dir / "international-football-results-from-1872-to-2017.zip"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("results.csv", raw_dir)
    zip_path.unlink(missing_ok=True)
    print(f"saved -> {target_csv}")


def load_match_data(cfg: dict | None = None) -> pd.DataFrame:
    """Load raw match history from data/raw/results.csv."""
    cfg = cfg or load_config()
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    candidate = raw_dir / "results.csv"
    if not candidate.exists():
        raise FileNotFoundError(
            f"{candidate} not found. Run download_match_data() first, or manually "
            "place results.csv there."
        )
    df = pd.read_csv(candidate, parse_dates=["date"])
    df["year"] = df["date"].dt.year
    return df


def filter_world_cup_matches(df: pd.DataFrame, include_qualifiers: bool = False) -> pd.DataFrame:
    """Restrict to FIFA World Cup matches, matching the paper's baseline scope.

    Args:
        include_qualifiers: if True, also include "FIFA World Cup qualification"
            rows (useful for building richer team-history features); the paper's
            baseline itself uses World Cup *finals* history only, so keep this
            False when reproducing Table I.
    """
    mask = df["tournament"] == "FIFA World Cup"
    if include_qualifiers:
        mask = mask | (df["tournament"] == "FIFA World Cup qualification")
    return df[mask].copy()


def add_net_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add a net_score column (home_score - away_score), used as a raw feature
    and to derive the win/loss label."""
    df = df.copy()
    df["net_score"] = df["home_score"] - df["away_score"]
    return df


if __name__ == "__main__":
    download_match_data()

