"""Match-history features: team strength as it stood *before* each fixture.

The player-attribute features in `build_features.py` describe squads; they say
nothing about results. The WWR baseline is the mirror image -- it knows results
and nothing about squads. On this dataset the two score within a point of each
other while disagreeing on roughly a third of matches, so neither one dominates
and the ensemble has never been given what the baseline knows.

This module closes that gap. Every feature here is computed **as of the match
date**, from matches strictly earlier than it:

  elo_*            classic Elo, updated chronologically over the whole history
  wwr_*            the baseline's own credibility-weighted win ratio
  played_*         matches played to date (the `v` term WWR shrinks on)
  h2h_winrate_*    head-to-head record against this specific opponent
  form_win_*       win rate over the team's last `form_window` matches
  form_gd_*        mean goal difference over that same window

Two properties matter more than the specific feature list:

* **No leakage, by construction rather than by split hygiene.** A match on date
  D sees only matches before D, so these features stay honest under any split
  strategy -- including the random split the pipeline currently uses, where a
  train-split-only fit would still let 2022 results inform a 2018 prediction.
  Same-day matches are excluded too: every fixture on D is snapshotted before
  any of D's results are applied, so two matches on the same day cannot see
  each other regardless of the order they appear in the file.

* **Mirror-safe naming.** The `_a` / `_b` / `_diff` convention means
  `get_feature_columns()` picks these up with no change, and the row-mirroring
  in `pipeline.run_seed()` swaps them correctly with no special-casing.
"""
from __future__ import annotations

import unicodedata
from collections import deque

import numpy as np
import pandas as pd

# Feature families emitted per team. Kept as a module constant so the mirroring
# helper and the tests can assert on it rather than re-deriving the list.
HISTORY_FEATURES = ("elo", "wwr", "played", "h2h_winrate", "form_win", "form_gd")

ELO_START = 1500.0

# Match importance, following the World Football Elo Ratings convention
# (eloratings.net). A flat K treats a June friendly as evidence equal to a World
# Cup quarter-final, which inflates the rating of nations that play many
# low-stakes matches -- Brazil's rating rests on 1,059 matches, most of them
# friendlies and continental fixtures.
ELO_WEIGHT_WORLD_CUP = 60.0
ELO_WEIGHT_CONTINENTAL = 50.0
ELO_WEIGHT_QUALIFIER = 40.0
ELO_WEIGHT_OTHER_TOURNAMENT = 30.0
ELO_WEIGHT_FRIENDLY = 20.0

# Matched against an accent-stripped, lowercased tournament name, so the
# encoding of "Copa América" in the source file cannot cause a silent miss.
_CONTINENTAL_FINALS = frozenset({
    "uefa euro", "copa america", "african cup of nations", "afc asian cup",
    "gold cup", "concacaf championship", "oceania nations cup",
    "confederations cup",
})


def _normalise_tournament(name: str) -> str:
    return (unicodedata.normalize("NFKD", str(name))
            .encode("ascii", "ignore").decode("ascii").lower().strip())


def competition_weight(tournament) -> float:
    """K-factor for a match, from the competition it was played in."""
    if not isinstance(tournament, str):
        return ELO_WEIGHT_OTHER_TOURNAMENT
    name = _normalise_tournament(tournament)
    if name == "fifa world cup":
        return ELO_WEIGHT_WORLD_CUP
    if name in _CONTINENTAL_FINALS:
        return ELO_WEIGHT_CONTINENTAL
    if "qualification" in name or "nations league" in name:
        return ELO_WEIGHT_QUALIFIER
    if name == "friendly":
        return ELO_WEIGHT_FRIENDLY
    return ELO_WEIGHT_OTHER_TOURNAMENT


def goal_difference_multiplier(goal_difference: int) -> float:
    """Scale the update by margin of victory, again per eloratings.net.

    A 4-0 win is stronger evidence than a 1-0 win, but with diminishing returns
    so a rout does not dominate the rating.
    """
    margin = abs(int(goal_difference))
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


class HistoryState:
    """Running team-strength state, advanced one match day at a time.

    `snapshot()` reads the state without changing it, so it can also serve
    inference: after a full pass over history the state holds every team's
    strength as of the last recorded match, which is exactly what a prediction
    for a future fixture needs.
    """

    def __init__(self, wwr_m: float = 10.0, form_window: int = 10,
                 elo_k: float = 20.0, elo_home_advantage: float = 60.0,
                 elo_competition_weighted: bool = False,
                 elo_goal_difference_weighted: bool = False):
        self.wwr_m = wwr_m
        self.form_window = form_window
        self.elo_k = elo_k
        self.elo_home_advantage = elo_home_advantage
        self.elo_competition_weighted = elo_competition_weighted
        self.elo_goal_difference_weighted = elo_goal_difference_weighted

        self.elo: dict[str, float] = {}
        self.played: dict[str, int] = {}
        self.wins: dict[str, int] = {}
        self.form: dict[str, deque] = {}
        self.h2h_played: dict[tuple[str, str], int] = {}
        self.h2h_wins: dict[tuple[tuple[str, str], str], int] = {}
        self.total_played = 0
        self.total_wins = 0

    # -- reads -------------------------------------------------------------

    def _wwr(self, team: str) -> float:
        """The baseline's shrinkage estimator, evaluated on history-to-date.

        Deliberately the same formula as `WeightedWinRatioBaseline._wwr` rather
        than a re-derivation: the point of this feature is to hand the ensemble
        the baseline's own signal, so a divergence here would quietly weaken the
        comparison it exists to settle.
        """
        v = self.played.get(team, 0)
        r = self.wins.get(team, 0) / v if v else 0.0
        # Global win ratio, i.e. the rate at which *some* team wins rather than
        # the match being drawn. Falls back to 0.5 before any match is recorded.
        c = self.total_wins / self.total_played if self.total_played else 0.5
        denom = v + self.wwr_m
        return (v / denom) * r + (self.wwr_m / denom) * c if denom else c

    def _h2h_winrate(self, team: str, opponent: str) -> float:
        """Share of decided head-to-head meetings won, shrunk toward 0.5.

        Shrinkage does the job an explicit "how many times have they met" column
        would: one lucky win in a single meeting lands near 0.5, a 9-3 record
        lands near its true rate. It also keeps the feature defined for a pair
        that has never met.
        """
        pair = (team, opponent) if team < opponent else (opponent, team)
        w_team = self.h2h_wins.get((pair, team), 0)
        w_opp = self.h2h_wins.get((pair, opponent), 0)
        decided = w_team + w_opp
        prior = 2.0  # meetings worth of pull toward even
        return (w_team + prior * 0.5) / (decided + prior)

    def _form(self, team: str) -> tuple[float, float]:
        """(win rate, mean goal difference) over the team's last N matches.

        Empty history returns (0.5, 0.0) -- an unknown team is treated as
        average rather than as a loser, which is what an all-zero fill would
        imply after scaling.
        """
        recent = self.form.get(team)
        if not recent:
            return 0.5, 0.0
        results = np.fromiter((r for r, _ in recent), dtype=float, count=len(recent))
        diffs = np.fromiter((d for _, d in recent), dtype=float, count=len(recent))
        return float(results.mean()), float(diffs.mean())

    def snapshot(self, home: str, away: str) -> dict[str, float]:
        """Feature dict for one fixture, from state as it currently stands."""
        out: dict[str, float] = {}
        for side, team, opponent in (("a", home, away), ("b", away, home)):
            form_win, form_gd = self._form(team)
            out[f"elo_{side}"] = self.elo.get(team, ELO_START)
            out[f"wwr_{side}"] = self._wwr(team)
            out[f"played_{side}"] = float(self.played.get(team, 0))
            out[f"h2h_winrate_{side}"] = self._h2h_winrate(team, opponent)
            out[f"form_win_{side}"] = form_win
            out[f"form_gd_{side}"] = form_gd
        for name in HISTORY_FEATURES:
            out[f"{name}_diff"] = out[f"{name}_a"] - out[f"{name}_b"]
        return out

    # -- writes ------------------------------------------------------------

    def update(self, home: str, away: str, home_score: float, away_score: float,
               neutral: bool = False, tournament=None) -> None:
        """Fold one played match into the state. Unplayed fixtures are skipped."""
        if pd.isna(home_score) or pd.isna(away_score):
            return

        # Elo first: it has to read both ratings before either is written.
        elo_home = self.elo.get(home, ELO_START)
        elo_away = self.elo.get(away, ELO_START)
        advantage = 0.0 if neutral else self.elo_home_advantage
        expected_home = 1.0 / (1.0 + 10 ** ((elo_away - (elo_home + advantage)) / 400.0))
        if home_score > away_score:
            actual_home = 1.0
        elif away_score > home_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        k = competition_weight(tournament) if self.elo_competition_weighted else self.elo_k
        if self.elo_goal_difference_weighted:
            k *= goal_difference_multiplier(home_score - away_score)

        adjustment = k * (actual_home - expected_home)
        self.elo[home] = elo_home + adjustment
        self.elo[away] = elo_away - adjustment

        self.played[home] = self.played.get(home, 0) + 1
        self.played[away] = self.played.get(away, 0) + 1
        self.total_played += 2

        pair = (home, away) if home < away else (away, home)
        self.h2h_played[pair] = self.h2h_played.get(pair, 0) + 1

        winner = home if home_score > away_score else (away if away_score > home_score else None)
        if winner is not None:
            self.wins[winner] = self.wins.get(winner, 0) + 1
            self.total_wins += 1
            self.h2h_wins[(pair, winner)] = self.h2h_wins.get((pair, winner), 0) + 1

        margin = float(home_score - away_score)
        self._push_form(home, actual_home, margin)
        self._push_form(away, 1.0 - actual_home, -margin)

    def _push_form(self, team: str, result: float, goal_diff: float) -> None:
        if team not in self.form:
            self.form[team] = deque(maxlen=self.form_window)
        self.form[team].append((result, goal_diff))


def history_feature_columns() -> list[str]:
    """Every column `build_history_features` emits, in a stable order."""
    return [f"{name}_{side}" for name in HISTORY_FEATURES for side in ("a", "b", "diff")]


def _update_from_row(state: HistoryState, row, has_neutral: bool) -> None:
    """Apply one itertuples row to the state, tolerating absent optional columns."""
    state.update(
        row.home_team, row.away_team, row.home_score, row.away_score,
        neutral=bool(getattr(row, "neutral", False)) if has_neutral else False,
        tournament=getattr(row, "tournament", None),
    )


def build_history_state(history: pd.DataFrame, **kwargs) -> HistoryState:
    """Advance a fresh state through `history` and return it, taking no
    snapshots.

    For scoring a set of fixtures from a single point in time -- e.g. the 2026
    bracket backtest, which must see the world exactly as it stood the day
    before the tournament and no later. Slice `history` by date first; the cut
    is the caller's, since only the caller knows what "before" means for its
    question.
    """
    state = HistoryState(**kwargs)
    has_neutral = "neutral" in history.columns
    for row in history.sort_values("date", kind="stable").itertuples(index=False):
        _update_from_row(state, row, has_neutral)
    return state


def build_history_features(
    targets: pd.DataFrame,
    history: pd.DataFrame,
    wwr_m: float = 10.0,
    form_window: int = 10,
    elo_k: float = 20.0,
    elo_home_advantage: float = 60.0,
    elo_competition_weighted: bool = False,
    elo_goal_difference_weighted: bool = False,
) -> tuple[pd.DataFrame, HistoryState]:
    """As-of-date history features for every row of `targets`.

    Args:
        targets: rows needing features; needs date, home_team, away_team. Its
            index is preserved in the result.
        history: the match pool the features are computed from -- normally the
            full international record, not just the World Cup, since form and
            Elo need volume to mean anything. Needs date, home_team, away_team,
            home_score, away_score, and optionally neutral.

    Returns:
        (features indexed like `targets`, final state after the whole history).
        The state is returned so inference can reuse it: it holds every team's
        strength as of the last recorded match.
    """
    state = HistoryState(wwr_m=wwr_m, form_window=form_window, elo_k=elo_k,
                         elo_home_advantage=elo_home_advantage,
                         elo_competition_weighted=elo_competition_weighted,
                         elo_goal_difference_weighted=elo_goal_difference_weighted)

    # (date, home, away) -> the target rows wanting a snapshot at that fixture.
    # A list, because the same fixture key can legitimately appear twice in the
    # source data and both rows should get the same state.
    wanted: dict[tuple, list] = {}
    for idx, date, home, away in zip(
        targets.index, targets["date"], targets["home_team"], targets["away_team"]
    ):
        wanted.setdefault((date, home, away), []).append(idx)

    history = history.sort_values("date", kind="stable")
    has_neutral = "neutral" in history.columns
    snapshots: dict = {}

    # Day-at-a-time: snapshot everything on date D, *then* apply D's results, so
    # a fixture can never see another match played the same day.
    for _, day in history.groupby("date", sort=True):
        rows = list(day.itertuples(index=False))
        for row in rows:
            key = (row.date, row.home_team, row.away_team)
            if key in wanted:
                snap = state.snapshot(row.home_team, row.away_team)
                for idx in wanted[key]:
                    snapshots[idx] = snap
        for row in rows:
            _update_from_row(state, row, has_neutral)

    missing = [idx for idx in targets.index if idx not in snapshots]
    if missing:
        # A target that is not in `history` gets the final state rather than
        # NaN; this is the inference-shaped case (a fixture not yet played).
        for idx in missing:
            row = targets.loc[idx]
            snapshots[idx] = state.snapshot(row["home_team"], row["away_team"])

    feats = pd.DataFrame.from_dict(snapshots, orient="index")
    return feats.reindex(targets.index)[history_feature_columns()], state
