"""Backtest the ensemble against the real 2026 World Cup knockout bracket.

The point of this script is that the model has never seen a 2026 match.
`config.yaml`'s `data.years` stops at 2025, so no 2026 fixture reaches training
or the WWR baseline; team strength comes from EA FC 25 squad ratings, which
were published before the tournament kicked off on 2026-06-11.

    python -m src.backtest_wc2026

Two details that make this an honest test rather than a flattering one:

  * Knockout ties can't end level, so a drawn match is resolved on penalties.
    The model predicts 90-minute home/away win and has no notion of a shootout,
    so those ties are scored against the penalty winner from shootouts.csv --
    the harder target, since a shootout is close to a coin flip.
  * Every match is scored with `neutral=True` symmetrisation (average of both
    orientations). A World Cup is played on neutral ground, and without this
    the model's learned home advantage would just favour whichever team
    results.csv happens to list first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path, set_seed
from src.data.collect_matches import load_match_data
from src.features.build_features import build_single_match_features
from src.features.history_features import build_history_state

# Group stage is the first 72 of the 104 matches (12 groups x 6); everything
# from 2026-06-28 on is knockout. Round boundaries are date ranges, since
# results.csv has no round column.
ROUNDS = [
    ("Round of 32", "2026-06-28", "2026-07-03"),
    ("Round of 16", "2026-07-04", "2026-07-07"),
    ("Quarter-finals", "2026-07-09", "2026-07-11"),
    ("Semi-finals", "2026-07-14", "2026-07-15"),
    ("Third place", "2026-07-18", "2026-07-18"),
    ("Final", "2026-07-19", "2026-07-19"),
]

PROFILE_YEAR = 2025  # latest squad data that predates the tournament
TOURNAMENT_START = "2026-06-11"  # opening match; history features are cut here


def _load_model(cfg: dict):
    """Train (or reuse the API's cache of) the ensemble on years <= 2025."""
    from api.model_cache import load_or_train
    seeds = [cfg["project"]["random_state"]]
    bundle = load_or_train(cfg, seeds)
    report = bundle["reports_by_seed"][seeds[0]]
    return bundle["profiles"], report.attrs


def predict_winner(team_a: str, team_b: str, profiles, attrs, history_state=None,
                   year: int = PROFILE_YEAR) -> tuple[str, float]:
    """Neutral-venue (order-invariant) winner + confidence for one tie.

    `history_state` must be built from matches BEFORE the tournament. Reusing
    the state the training pipeline ends on would fold the very 2026 results
    being predicted into the features doing the predicting.
    """
    def proba_a(a: str, b: str) -> np.ndarray:
        row = build_single_match_features(a, b, year, profiles, history_state, neutral=True)
        X = row[attrs["feature_cols"]].fillna(attrs["train_mean"])
        X = attrs["scaler"].transform(X)
        if attrs["pca"] is not None:
            X = attrs["pca"].transform(X)
        return np.array([est.predict_proba(X)[0][1]
                         for est in attrs["ensemble"].named_estimators_.values()])

    # Average both orientations so argument order can't decide the winner.
    p = (proba_a(team_a, team_b) + (1.0 - proba_a(team_b, team_a))) / 2.0
    mean_p = float(p.mean())
    return (team_a, mean_p) if mean_p >= 0.5 else (team_b, 1.0 - mean_p)


def actual_winner(row, shootouts: pd.DataFrame) -> str | None:
    """Who really went through. None when the result isn't recorded yet."""
    if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
        return None
    if row["home_score"] > row["away_score"]:
        return row["home_team"]
    if row["away_score"] > row["home_score"]:
        return row["away_team"]
    hit = shootouts[(shootouts["date"] == row["date"])
                    & (shootouts["home_team"] == row["home_team"])
                    & (shootouts["away_team"] == row["away_team"])]
    return hit.iloc[0]["winner"] if len(hit) else None


def compute_results(cfg: dict, verbose: bool = True) -> tuple[pd.DataFrame, list]:
    """Run every tie through the ensemble and the baseline, returning the
    per-tie results (round, correct, confidence, baseline_correct, penalties)
    plus the list of ties with no recorded result (genuine forward
    predictions). Shared by the CLI report in `run()` and by the report
    figure in `src/make_figures.py`, so both read the same numbers."""
    set_seed(cfg["project"]["random_state"])
    if 2026 in cfg["data"]["years"]:
        raise SystemExit("config data.years includes 2026 -- the backtest is meaningless "
                         "if 2026 matches were part of training.")

    profiles, attrs = _load_model(cfg)
    matches = load_match_data(cfg)
    wc = matches[(matches["tournament"] == "FIFA World Cup")
                 & (matches["year"] == 2026)].sort_values("date")
    shootouts = pd.read_csv(resolve_path("data/raw/shootouts.csv"), parse_dates=["date"])

    # Same benchmark the offline evaluation uses, held to the same rule: fit on
    # World Cup history STRICTLY BEFORE 2026, so it gets no more hindsight than
    # the ensemble does.
    from src.data.collect_matches import filter_world_cup_matches
    from src.models.baseline import WeightedWinRatioBaseline

    history = filter_world_cup_matches(matches, include_qualifiers=cfg["data"]["include_qualifiers"])
    baseline = WeightedWinRatioBaseline(
        min_head_to_head_matches=cfg["baseline"]["min_head_to_head_matches"],
        wwr_m=cfg["baseline"]["wwr_m"],
    ).fit(history[history["year"] < 2026])

    # Team strength as it stood the day before the opening match. Cut from the
    # full international record, matching how the training features were built.
    history_state = build_history_state(
        matches[matches["date"] < pd.Timestamp(TOURNAMENT_START)],
        wwr_m=cfg["baseline"]["wwr_m"],
        form_window=cfg["features"].get("history_form_window", 10),
        elo_k=cfg["features"].get("history_elo_k", 20),
        elo_home_advantage=cfg["features"].get("history_elo_home_advantage", 60),
        elo_competition_weighted=cfg["features"].get("history_elo_competition_weighted", False),
        elo_goal_difference_weighted=cfg["features"].get("history_elo_goal_difference_weighted", False),
    ) if cfg["features"].get("history_features", False) else None

    if verbose:
        print(f"Model: trained on {cfg['data']['years'][0]}-{cfg['data']['years'][-1]}, "
              f"no 2026 match seen. Team strength from EA FC {PROFILE_YEAR - 2000} squads"
              + (f", match history cut at {TOURNAMENT_START}.\n" if history_state is not None else ".\n"))

    rows, unresolved = [], []
    for name, start, end in ROUNDS:
        tie_rows = wc[(wc["date"] >= start) & (wc["date"] <= end)]
        if tie_rows.empty:
            continue
        if verbose:
            print(f"{'=' * 72}\n{name}\n{'=' * 72}")
        for _, r in tie_rows.iterrows():
            try:
                pred, conf = predict_winner(r["home_team"], r["away_team"], profiles,
                                            attrs, history_state)
            except ValueError as exc:
                if verbose:
                    print(f"  {r['home_team']} vs {r['away_team']}: SKIPPED -- {exc}")
                continue
            truth = actual_winner(r, shootouts)
            if truth is None:
                if verbose:
                    print(f"  {r['home_team']:<22} vs {r['away_team']:<22} "
                          f"-> {pred} ({conf:.0%})   [no result recorded]")
                unresolved.append((name, r["home_team"], r["away_team"], pred, conf))
                continue
            ok = pred == truth
            base_pick = baseline.predict_one(r["home_team"], r["away_team"])
            went_to_pens = r["home_score"] == r["away_score"]
            rows.append({"round": name, "correct": ok, "confidence": conf,
                         "baseline_correct": base_pick == truth, "penalties": went_to_pens})
            if verbose:
                print(f"  {r['home_team']:<22} vs {r['away_team']:<22} "
                      f"-> {pred:<22} actual {truth:<22} {'HIT ' if ok else 'MISS'} ({conf:.0%})")
        if verbose:
            print()

    return pd.DataFrame(rows), unresolved


def run() -> None:
    cfg = load_config()
    df, unresolved = compute_results(cfg)

    print(f"{'=' * 72}\nAccuracy by round\n{'=' * 72}")
    per_round = df.groupby("round", sort=False).agg(
        ties=("correct", "count"), hits=("correct", "sum"),
        baseline_hits=("baseline_correct", "sum"),
    )
    per_round["ensemble"] = (per_round["hits"] / per_round["ties"]).map("{:.0%}".format)
    per_round["baseline"] = (per_round["baseline_hits"] / per_round["ties"]).map("{:.0%}".format)
    print(per_round.to_string())

    n = len(df)
    ens, base = df["correct"].mean(), df["baseline_correct"].mean()
    se = (ens * (1 - ens) / n) ** 0.5
    print(f"\nEnsemble : {df['correct'].sum()}/{n} = {ens:.1%}  (+/- {1.96 * se:.1%} at 95%)")
    print(f"Baseline : {df['baseline_correct'].sum()}/{n} = {base:.1%}")
    print(f"Coin flip: ~50%   |   {int(df['penalties'].sum())} of {n} ties went to penalties")

    if unresolved:
        print(f"\n{'=' * 72}\nNot yet decided in this dataset -- genuine forward predictions\n{'=' * 72}")
        for name, a, b, pred, conf in unresolved:
            print(f"  {name}: {a} vs {b} -> {pred} ({conf:.0%})")


if __name__ == "__main__":
    run()
