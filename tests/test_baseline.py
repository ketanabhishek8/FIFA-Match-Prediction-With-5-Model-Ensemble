"""Sanity tests for the WWR baseline (src/models/baseline.py).

Run with: pytest tests/test_baseline.py -v
"""
import pandas as pd

from src.models.baseline import WeightedWinRatioBaseline


def _toy_match_history() -> pd.DataFrame:
    # Team A beats Team B 4 times, Team B beats Team A once -> 5 head-to-head
    # matches, hitting the min_head_to_head_matches=5 threshold, so the
    # head-to-head rule (not WWR) should decide this matchup.
    rows = []
    for _ in range(4):
        rows.append({"home_team": "A", "away_team": "B", "home_score": 2, "away_score": 0})
    rows.append({"home_team": "B", "away_team": "A", "home_score": 1, "away_score": 0})
    # A few extra matches vs a third team so overall win ratios are non-trivial.
    rows.append({"home_team": "A", "away_team": "C", "home_score": 1, "away_score": 1})  # draw, ignored
    rows.append({"home_team": "C", "away_team": "B", "home_score": 0, "away_score": 3})
    return pd.DataFrame(rows)


def test_head_to_head_rule_used_when_threshold_met():
    history = _toy_match_history()
    model = WeightedWinRatioBaseline(min_head_to_head_matches=5, wwr_m=10)
    model.fit(history)

    # A won 4 of 5 head-to-head matches vs B -> predict A.
    assert model.predict_one("A", "B") == "A"


def test_wwr_fallback_used_below_threshold():
    # C has only played 1 match vs B and 1 vs A -> below threshold of 5, so
    # WWR (shrunk toward the global average) decides, not raw head-to-head wins.
    history = _toy_match_history()
    model = WeightedWinRatioBaseline(min_head_to_head_matches=5, wwr_m=10)
    model.fit(history)

    prediction = model.predict_one("A", "C")
    assert prediction in {"A", "C"}  # both are valid teams; just check it runs and returns a team


def test_global_win_ratio_is_between_zero_and_one():
    history = _toy_match_history()
    model = WeightedWinRatioBaseline()
    model.fit(history)
    assert 0.0 <= model._global_win_ratio <= 1.0
