"""Tests for squad selection and team-year aggregation (src/data/clean.py).

Run with: pytest tests/test_clean_profiles.py -v
"""
import numpy as np
import pandas as pd
import pytest

from src.data.clean import (
    SQUAD_QUOTA,
    _primary_position_group,
    build_all_team_year_profiles,
    build_team_year_profile,
    filter_national_team_squad,
)


def _player(name, nation, year, overall, positions, nation_position=None):
    """One player row with every column the aggregation touches."""
    row = {"short_name": name, "nationality_name": nation, "year": year,
           "overall": overall, "player_positions": positions,
           "nation_position": nation_position,
           "goalkeeping_diving": 20.0, "goalkeeping_handling": 20.0,
           "goalkeeping_kicking": 20.0, "goalkeeping_positioning": 20.0,
           "goalkeeping_reflexes": 20.0,
           "defending_marking_awareness": 50.0, "defending_standing_tackle": 50.0,
           "defending_sliding_tackle": 50.0,
           "attacking_crossing": 60.0, "attacking_finishing": 60.0,
           "attacking_heading_accuracy": 60.0, "attacking_short_passing": 60.0,
           "attacking_volleys": 60.0}
    if positions.startswith("GK"):
        row.update({c: 80.0 for c in row if c.startswith("goalkeeping_")})
    return row


@pytest.mark.parametrize("positions,expected", [
    ("GK", "gk"), ("CB, RB", "def"), ("LWB", "def"),
    ("CDM, CM", "mid"), ("CAM", "mid"),
    ("LW, ST", "att"), ("ST", "att"), ("CF", "att"),
    (None, None), ("", None), ("XX", None),
])
def test_primary_position_group(positions, expected):
    assert _primary_position_group(positions) == expected


def _unlicensed_pool() -> pd.DataFrame:
    """40 outfielders rated above every keeper -- the real-data pattern that
    left 58% of proxy squads with no goalkeeper."""
    rows = [_player(f"out{i}", "Testland", 2020, 90 - i, "ST") for i in range(40)]
    rows += [_player(f"gk{i}", "Testland", 2020, 60 - i, "GK") for i in range(4)]
    return pd.DataFrame(rows)


def test_fallback_squad_includes_keepers_despite_low_ratings():
    squad = filter_national_team_squad(_unlicensed_pool(), top_n=23)
    groups = squad["player_positions"].map(_primary_position_group)

    assert len(squad) == 23
    assert (groups == "gk").sum() == SQUAD_QUOTA["gk"]
    # Top-23-by-overall would have taken 23 strikers and no keeper at all.
    assert (groups == "gk").sum() > 0


def test_fallback_backfills_when_a_group_is_short():
    """A pool with no midfielders still yields a full squad."""
    squad = filter_national_team_squad(_unlicensed_pool(), top_n=23)
    assert len(squad) == 23  # mid quota unfilled -> backfilled from strikers


def test_tiny_pool_still_produces_a_squad():
    tiny = pd.DataFrame([_player("a", "Smallland", 2020, 60, "ST"),
                         _player("b", "Smallland", 2020, 55, "GK")])
    squad = filter_national_team_squad(tiny, top_n=23)
    assert len(squad) == 2


def test_licensed_squads_bypass_the_quota():
    """A real call-up list is used as-is, not reshaped to the quota."""
    rows = [_player(f"p{i}", "Licensia", 2020, 80 - i, "ST", nation_position="SUB")
            for i in range(11)]
    rows += [_player("uncapped", "Licensia", 2020, 99, "ST")]
    squad = filter_national_team_squad(pd.DataFrame(rows), top_n=23)

    assert len(squad) == 11
    assert "uncapped" not in set(squad["short_name"])


def _squad_frame() -> pd.DataFrame:
    return pd.DataFrame([
        _player("star", "Shapeland", 2020, 95, "ST"),
        _player("good", "Shapeland", 2020, 85, "CAM"),
        _player("ok", "Shapeland", 2020, 80, "CB"),
        _player("weak", "Shapeland", 2020, 60, "CB"),
        _player("keeper", "Shapeland", 2020, 70, "GK"),
    ])


def test_shape_statistics_describe_star_concentration():
    p = build_team_year_profile(_squad_frame(), "Shapeland", 2020)
    overalls = [95, 85, 80, 60, 70]

    assert p["overall"] == pytest.approx(np.mean(overalls))
    assert p["overall_max"] == 95
    assert p["overall_top3"] == pytest.approx((95 + 85 + 80) / 3)
    assert p["overall_spread"] == pytest.approx(np.std(overalls))  # population SD


def test_position_group_overalls_use_only_that_group():
    p = build_team_year_profile(_squad_frame(), "Shapeland", 2020)

    assert p["overall_gk"] == 70
    assert p["overall_att"] == 95
    assert p["overall_mid"] == 85
    assert p["overall_def"] == pytest.approx((80 + 60) / 2)


def test_core_attributes_are_restricted_to_the_right_players():
    """goalkeeping_core must reflect keepers, not the 4 outfielders rated 20."""
    p = build_team_year_profile(_squad_frame(), "Shapeland", 2020)

    assert p["goalkeeping_core"] == pytest.approx(80.0)
    # The squad-wide mean is still there, still diluted -- deliberately kept so
    # teams with no rated keeper don't lose the column entirely.
    assert p["goalkeeping_diving"] == pytest.approx((80 + 20 * 4) / 5)


def test_missing_position_group_yields_nan_not_zero():
    """A squad with no keeper must not look like a squad with a terrible one."""
    no_gk = _squad_frame()
    no_gk = no_gk[no_gk["short_name"] != "keeper"]
    p = build_team_year_profile(no_gk, "Shapeland", 2020)

    assert np.isnan(p["overall_gk"])
    assert np.isnan(p["goalkeeping_core"])


def test_profiles_frame_is_numeric_not_object():
    """Rows built as dicts, so pandas infers real dtypes -- a list of Series
    each carrying a string `team` yields object columns that push boxed Python
    floats into the scaler."""
    profiles = build_all_team_year_profiles(_squad_frame())
    non_numeric = [c for c in profiles.columns
                   if c != "team" and profiles[c].dtype.kind not in "fiu"]

    assert non_numeric == []
    assert profiles["overall_max"].dtype.kind in "fiu"
    assert profiles["year"].dtype.kind in "fiu"


def test_all_profiles_matches_single_profile():
    frame = _squad_frame()
    one = build_team_year_profile(frame, "Shapeland", 2020)
    allp = build_all_team_year_profiles(frame).iloc[0]

    for col in ("overall", "overall_max", "overall_top3", "overall_gk", "goalkeeping_core"):
        assert allp[col] == pytest.approx(one[col])


def test_unknown_team_raises():
    with pytest.raises(ValueError):
        build_team_year_profile(_squad_frame(), "Nowhere", 2020)
