"""Weighted Win Ratio (WWR) baseline, per the paper's Section IV.

Rule:
  - If two teams have played >= `min_head_to_head_matches` matches historically
    (paper uses 5, the 75th percentile of historical head-to-head counts), the
    team with the higher total number of head-to-head wins is predicted to win.
  - Otherwise, fall back to the Weighted Win Ratio:

        WWR = (v / (v + m)) * R + (m / (v + m)) * C

    where:
      v = total matches the team has played (overall, not just head-to-head)
      R = the team's win ratio (wins / v)
      m = a threshold/shrinkage parameter (config: baseline.wwr_m)
      C = the overall win ratio computed across all teams

    The team with the higher WWR is predicted to win. This is a standard
    "credibility-weighted" shrinkage estimator (same family as IMDB's weighted
    rating): teams with few matches get pulled toward the global average C,
    teams with many matches are trusted closer to their own ratio R.

This is fully implemented (unlike the other models here) since the formula is
fully specified in the paper and doesn't depend on the final feature schema.
"""
from __future__ import annotations

import pandas as pd


class WeightedWinRatioBaseline:
    def __init__(self, min_head_to_head_matches: int = 5, wwr_m: float = 10.0):
        self.min_head_to_head_matches = min_head_to_head_matches
        self.wwr_m = wwr_m
        self._team_stats: dict[str, dict[str, float]] = {}
        self._h2h_wins: dict[tuple[str, str], int] = {}
        self._h2h_played: dict[tuple[str, str], int] = {}
        self._global_win_ratio: float | None = None

    def fit(self, match_history_df: pd.DataFrame) -> "WeightedWinRatioBaseline":
        """Fit team-level win ratios and head-to-head records.

        Expects columns: home_team, away_team, home_score, away_score.
        Draws are counted as a played match for both teams but not a win for
        either, consistent with computing a simple win ratio.
        """
        played: dict[str, int] = {}
        wins: dict[str, int] = {}

        # Unplayed fixtures (NaN scores) would otherwise count as a played
        # match for both teams with no winner, quietly diluting every win ratio.
        match_history_df = match_history_df.dropna(subset=["home_score", "away_score"])

        for row in match_history_df.itertuples(index=False):
            home, away = row.home_team, row.away_team
            played[home] = played.get(home, 0) + 1
            played[away] = played.get(away, 0) + 1

            if row.home_score > row.away_score:
                wins[home] = wins.get(home, 0) + 1
            elif row.away_score > row.home_score:
                wins[away] = wins.get(away, 0) + 1

            pair = tuple(sorted((home, away)))
            self._h2h_played[pair] = self._h2h_played.get(pair, 0) + 1
            winner = home if row.home_score > row.away_score else (
                away if row.away_score > row.home_score else None
            )
            if winner is not None:
                key = (pair, winner)
                self._h2h_wins[key] = self._h2h_wins.get(key, 0) + 1

        total_matches = sum(played.values())
        total_wins = sum(wins.values())
        self._global_win_ratio = total_wins / total_matches if total_matches else 0.5

        for team, v in played.items():
            r = wins.get(team, 0) / v if v else 0.0
            self._team_stats[team] = {"v": v, "R": r}

        return self

    def _wwr(self, team: str) -> float:
        stats = self._team_stats.get(team, {"v": 0, "R": 0.0})
        v, r = stats["v"], stats["R"]
        m, c = self.wwr_m, self._global_win_ratio
        return (v / (v + m)) * r + (m / (v + m)) * c

    def predict_one(self, team_a: str, team_b: str) -> str:
        """Return the predicted winner ('team_a' or 'team_b' identity, i.e. the
        team name string) for a single matchup."""
        pair = tuple(sorted((team_a, team_b)))
        h2h_count = self._h2h_played.get(pair, 0)

        if h2h_count >= self.min_head_to_head_matches:
            wins_a = self._h2h_wins.get((pair, team_a), 0)
            wins_b = self._h2h_wins.get((pair, team_b), 0)
            if wins_a != wins_b:
                return team_a if wins_a > wins_b else team_b
            # tie in head-to-head wins -> fall through to WWR as a tiebreaker

        return team_a if self._wwr(team_a) >= self._wwr(team_b) else team_b

    def predict(self, matches_df: pd.DataFrame) -> pd.Series:
        """Vectorized convenience wrapper over predict_one() for a DataFrame
        with home_team / away_team columns."""
        return matches_df.apply(
            lambda row: self.predict_one(row["home_team"], row["away_team"]), axis=1
        )
