"""Tests for feature engineering (src/features/build_features.py).

This module decides the labels, so a silent bug here moves every reported
number without breaking anything visibly.

Run with: pytest tests/test_build_features.py -v
"""
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    build_match_features,
    build_single_match_features,
    edition_for_match_date,
    get_feature_columns,
)


@pytest.fixture
def profiles() -> pd.DataFrame:
    return pd.DataFrame([
        {"team": "A", "year": 2018, "overall": 80.0, "pace": 70.0, "squad_size": 23},
        {"team": "B", "year": 2018, "overall": 70.0, "pace": 75.0, "squad_size": 23},
    ])


def _match(home, away, hs, aws, neutral=False, date="2018-06-14"):
    return {"date": pd.Timestamp(date), "home_team": home, "away_team": away,
            "home_score": hs, "away_score": aws, "year": 2018,
            "tournament": "FIFA World Cup", "neutral": neutral}


def test_draws_are_dropped(profiles):
    matches = pd.DataFrame([
        _match("A", "B", 2, 1),
        _match("A", "B", 1, 1),   # draw -> excluded per the paper's framing
        _match("A", "B", 0, 3),
    ])
    feat = build_match_features(matches, profiles)
    assert len(feat) == 2


def test_label_one_means_the_home_side_won(profiles):
    matches = pd.DataFrame([_match("A", "B", 2, 1), _match("A", "B", 0, 3)])
    feat = build_match_features(matches, profiles).reset_index(drop=True)

    assert feat.loc[0, "label"] == 1  # home scored more
    assert feat.loc[1, "label"] == 0  # away scored more


def test_unplayed_fixtures_are_dropped_not_labelled_as_away_wins(profiles):
    """`NaN != NaN` is True, so an unplayed row survives the draw filter and
    then gets label = (NaN > NaN) = 0. It must be dropped before either step."""
    matches = pd.DataFrame([
        _match("A", "B", 2, 1),
        _match("A", "B", np.nan, np.nan),  # scheduled, not yet played
    ])
    feat = build_match_features(matches, profiles)

    assert len(feat) == 1
    assert feat["label"].tolist() == [1]


def test_diff_is_a_minus_b(profiles):
    matches = pd.DataFrame([_match("A", "B", 2, 1)])
    feat = build_match_features(matches, profiles).reset_index(drop=True)

    assert feat.loc[0, "overall_a"] == 80.0
    assert feat.loc[0, "overall_b"] == 70.0
    assert feat.loc[0, "overall_diff"] == pytest.approx(10.0)
    assert feat.loc[0, "pace_diff"] == pytest.approx(-5.0)


def test_neutral_site_is_carried_through_as_a_numeric_feature(profiles):
    matches = pd.DataFrame([_match("A", "B", 2, 1, neutral=True),
                            _match("A", "B", 2, 1, neutral=False)])
    feat = build_match_features(matches, profiles).reset_index(drop=True)

    assert feat["neutral_site"].tolist() == [1.0, 0.0]
    assert "neutral_site" in get_feature_columns(feat)


def test_feature_columns_exclude_ids_labels_and_outcomes(profiles):
    matches = pd.DataFrame([_match("A", "B", 2, 1)])
    cols = get_feature_columns(build_match_features(matches, profiles))

    for leaked in ("label", "home_score", "away_score", "home_team", "away_team",
                   "date", "year", "tournament", "squad_size_a", "squad_size_b"):
        assert leaked not in cols


def test_single_match_features_match_the_training_schema(profiles):
    """Inference must produce every column training selected, or /predict dies
    on a KeyError the moment the schema changes."""
    matches = pd.DataFrame([_match("A", "B", 2, 1)])
    train_cols = get_feature_columns(build_match_features(matches, profiles))

    row = build_single_match_features("A", "B", 2018, profiles)
    missing = [c for c in train_cols if c not in row.columns]
    assert missing == []


def test_single_match_features_default_to_a_neutral_venue(profiles):
    row = build_single_match_features("A", "B", 2018, profiles)
    assert row["neutral_site"].iloc[0] == 1.0

    row = build_single_match_features("A", "B", 2018, profiles, neutral=False)
    assert row["neutral_site"].iloc[0] == 0.0


def test_unknown_team_or_year_raises(profiles):
    with pytest.raises(ValueError):
        build_single_match_features("A", "Nowhere", 2018, profiles)
    with pytest.raises(ValueError):
        build_single_match_features("A", "B", 1998, profiles)


# -- edition-aware join --------------------------------------------------


@pytest.mark.parametrize("date,expected", [
    ("2018-01-15", 2018),   # Jan: edition 2018 shipped Sept 2017, still current
    ("2018-06-14", 2018),   # the World Cup itself -- the pre-tournament rating
    ("2018-08-31", 2018),   # last day before the next edition ships
    ("2018-09-01", 2019),   # edition 2019 has shipped; it is now the fresher one
    ("2018-11-20", 2019),
    ("2018-12-31", 2019),
])
def test_edition_for_match_date(date, expected):
    got = edition_for_match_date(pd.Series([pd.Timestamp(date)]), range(2015, 2026))
    assert got.iloc[0] == expected


def test_edition_is_clamped_to_available_years():
    """Nov 2025 wants edition 2026, which is not on disk -- it must fall back to
    the newest available rather than dropping the match from the join."""
    got = edition_for_match_date(pd.Series([pd.Timestamp("2025-11-18")]), range(2015, 2026))
    assert got.iloc[0] == 2025

    early = edition_for_match_date(pd.Series([pd.Timestamp("2010-03-01")]), range(2015, 2026))
    assert early.iloc[0] == 2015


def test_edition_aware_join_reads_the_newer_edition(profiles):
    """A November match must pick up the following edition's squad."""
    two_years = pd.concat([
        profiles,
        profiles.assign(year=2019, overall=profiles["overall"] + 10.0),
    ], ignore_index=True)
    november = pd.DataFrame([_match("A", "B", 2, 1, date="2018-11-20")])

    aware = build_match_features(november, two_years, edition_aware=True).reset_index(drop=True)
    calendar = build_match_features(november, two_years, edition_aware=False).reset_index(drop=True)

    assert aware.loc[0, "profile_year"] == 2019
    assert calendar.loc[0, "profile_year"] == 2018
    assert aware.loc[0, "overall_a"] == calendar.loc[0, "overall_a"] + 10.0


# -- representation and reliability --------------------------------------


def test_representation_diff_drops_the_levels(profiles):
    feat = build_match_features(pd.DataFrame([_match("A", "B", 2, 1)]), profiles)
    all_cols = get_feature_columns(feat, "all")
    diff_cols = get_feature_columns(feat, "diff")

    assert len(diff_cols) < len(all_cols)
    assert not any(c.endswith(("_a", "_b")) for c in diff_cols)
    assert "overall_diff" in diff_cols
    # Match context survives both representations -- it has no _diff form.
    assert "neutral_site" in diff_cols and "neutral_site" in all_cols


def test_unknown_representation_raises(profiles):
    feat = build_match_features(pd.DataFrame([_match("A", "B", 2, 1)]), profiles)
    with pytest.raises(ValueError, match="representation"):
        get_feature_columns(feat, "levels-only")


def test_squad_size_min_is_the_weaker_profile(profiles):
    thin = profiles.copy()
    thin.loc[thin["team"] == "B", "squad_size"] = 3
    feat = build_match_features(pd.DataFrame([_match("A", "B", 2, 1)]), thin).reset_index(drop=True)

    assert feat.loc[0, "squad_size_min"] == 3
    assert "squad_size_min" in get_feature_columns(feat, "diff")
    # The raw per-side sizes stay out: only the binding constraint is a feature.
    assert "squad_size_a" not in get_feature_columns(feat, "all")
