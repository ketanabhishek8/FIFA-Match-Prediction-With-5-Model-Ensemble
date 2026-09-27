"""Feature engineering: turn (team-year profiles + match history) into a
model-ready X, y for the classification task described in the paper
(Section III.B.1 — binary Team A win / Team B win, draws excluded).

Since each match has two inputs (Team A, Team B), every profile attribute is
represented twice per row: `<attr>_a` and `<attr>_b`. We also add the
difference `<attr>_diff = <attr>_a - <attr>_b`, which tends to help linear
models pick up on relative strength without needing interaction terms.
"""
from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def edition_for_match_date(dates: pd.Series, available_years) -> pd.Series:
    """Which FIFA edition's ratings were current when a match was played.

    The edition labelled year Y ships in about September of Y-1 and describes
    that season's squads, so it is the current one from Sept Y-1 to Aug Y.
    Joining on the match's calendar year instead means a fixture played in
    October reads ratings published thirteen months earlier while a fresher
    edition already exists -- and 63% of the World Cup matches in scope are
    played September to December.

    Clamped to the editions actually on disk, so the newest matches fall back
    to the newest available ratings rather than dropping out of the join.
    """
    years = dates.dt.year + (dates.dt.month >= 9).astype(int)
    return years.clip(lower=min(available_years), upper=max(available_years))


def _attach_team_profiles(pairs_df: pd.DataFrame, team_profiles: pd.DataFrame) -> pd.DataFrame:
    """Join team-year profiles onto each (home_team, away_team, profile_year)
    row, producing `_a` / `_b` / `_diff` suffixed columns. Shared by
    build_match_features (training, needs scores/labels) and
    build_single_match_features (inference, doesn't).

    `pairs_df` must carry `profile_year` -- the edition to read squads from,
    which is not necessarily the match's calendar year (see
    `edition_for_match_date`).
    """
    profile_cols = [c for c in team_profiles.columns if c not in ("team", "year", "squad_size")]

    a = team_profiles.add_suffix("_a").rename(
        columns={"team_a": "home_team", "year_a": "profile_year"})
    b = team_profiles.add_suffix("_b").rename(
        columns={"team_b": "away_team", "year_b": "profile_year"})

    merged = pairs_df.merge(a, on=["home_team", "profile_year"], how="inner")
    merged = merged.merge(b, on=["away_team", "profile_year"], how="inner")

    for col in profile_cols:
        merged[f"{col}_diff"] = merged[f"{col}_a"] - merged[f"{col}_b"]

    # How much squad actually backs this comparison. Nearly half of all match
    # rows have at least one side built from fewer than 11 rated players, and
    # without this the model cannot tell a full 23-man profile from a 2-player
    # one. The minimum is the right summary: the comparison is only as reliable
    # as its weaker half.
    merged["squad_size_min"] = merged[["squad_size_a", "squad_size_b"]].min(axis=1)

    return merged


def build_match_features(matches_df: pd.DataFrame, team_profiles: pd.DataFrame,
                         edition_aware: bool = True,
                         drop_draws: bool = True) -> pd.DataFrame:
    """Join team-year profiles onto each match for both the home and away team,
    producing one row per match with `_a` / `_b` / `_diff` suffixed columns.

    Args:
        matches_df: must contain home_team, away_team, date, year, and the
            outcome columns needed to build the label.
        team_profiles: output of clean.build_all_team_year_profiles().
        edition_aware: read squads from the FIFA edition current on the match
            date rather than the one labelled with the match's calendar year.
            See `edition_for_match_date`.
        drop_draws: exclude drawn matches, per the paper's binary framing. The
            scoreline model is the one caller that passes False -- it predicts a
            distribution over scorelines, in which a draw is an ordinary outcome
            rather than an unrepresentable one. Leaving them in recovers about a
            quarter of the match record. **`label` is meaningless on drawn rows
            when this is False** (it computes as 0, i.e. indistinguishable from
            an away win); anything reading `label` must filter draws itself.
    """
    matches_df = matches_df.copy()
    matches_df["profile_year"] = (
        edition_for_match_date(matches_df["date"], team_profiles["year"].unique())
        if edition_aware else matches_df["year"]
    )
    merged = _attach_team_profiles(matches_df, team_profiles)

    # Drop fixtures with no recorded result BEFORE the draw filter. results.csv
    # carries scheduled-but-unplayed matches with NaN scores, and `NaN != NaN`
    # is True in pandas -- so they'd survive the draw filter and then get
    # `label = (NaN > NaN) = False`, i.e. silently train as away wins.
    merged = merged.dropna(subset=["home_score", "away_score"])

    # Binary label: exclude draws per the paper's classification framing.
    if drop_draws:
        merged = merged[merged["home_score"] != merged["away_score"]]
    merged = merged.copy()
    merged["label"] = (merged["home_score"] > merged["away_score"]).astype(int)  # 1 = Team A (home) wins

    # Numeric copy of results.csv's `neutral` flag, so it can be a model feature
    # (see MATCH_CONTEXT_FEATURES) and so mirroring can tell which rows have an
    # arbitrary side ordering.
    merged["neutral_site"] = merged["neutral"].astype(float) if "neutral" in merged else 0.0

    return merged


def build_single_match_features(
    team_a: str, team_b: str, year: int, team_profiles: pd.DataFrame,
    history_state=None, neutral: bool = True,
) -> pd.DataFrame:
    """One-row feature frame for an arbitrary team pair with no known outcome,
    for live prediction. Raises ValueError if either team has no profile for
    that year (or they've never played, which can't happen for a single pair).

    Args:
        history_state: the `HistoryState` returned by `build_history_features`,
            holding team strength as of the last recorded match. Required
            whenever the model was trained with history features on, since the
            feature columns it expects would otherwise be missing; omit it only
            for a squad-attributes-only model.
        neutral: whether the fixture is at a neutral venue. Defaults True --
            World Cup matches are, and it is the safe default for a hypothetical
            matchup, where treating one side as "home" would invent an
            advantage nobody has.
    """
    # `year` here names the edition directly -- a caller asking for a live
    # prediction picks the squad vintage, so there is no match date to derive
    # one from.
    pairs_df = pd.DataFrame([{"home_team": team_a, "away_team": team_b,
                              "year": year, "profile_year": year}])
    merged = _attach_team_profiles(pairs_df, team_profiles)
    if merged.empty:
        raise ValueError(f"No profile data for {team_a!r} and/or {team_b!r} in {year}")

    merged["neutral_site"] = float(neutral)

    if history_state is not None:
        for col, value in history_state.snapshot(team_a, team_b).items():
            merged[col] = value
    return merged


def mirror_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Side-swapped copy of a feature frame: `_a` <-> `_b`, `_diff` negated.

    Pairs with a flipped label to teach the model that side is arbitrary. Relies
    only on the naming convention, so it covers squad attributes and history
    features alike with no per-feature knowledge.
    """
    renames = {}
    for col in X.columns:
        if col.endswith("_a"):
            renames[col] = f"{col[:-2]}_b"
        elif col.endswith("_b"):
            renames[col] = f"{col[:-2]}_a"

    mirrored = X.rename(columns=renames)
    diff_cols = [c for c in X.columns if c.endswith("_diff")]
    mirrored[diff_cols] = -mirrored[diff_cols]
    # rename() reorders nothing, but the a/b swap means positions no longer
    # match the caller's column order -- restore it so the frame stays a
    # drop-in for the original.
    return mirrored[X.columns]


# Features describing the fixture rather than either team, so they carry no
# _a/_b/_diff suffix and have to be listed explicitly.
#
# `neutral_site`: World Cup qualifiers are played home-and-away and the home
# side wins 63.1% of them, while on neutral ground it is 48.6% -- near a coin
# flip. Without this the model blends the two and applies a phantom home
# advantage to every neutral-venue fixture, which is every match at the
# tournament itself.
#
# `squad_size_min`: how much squad backs the comparison at all. Symmetric, so
# mirroring leaves it alone, and it is a level rather than a difference -- which
# is the point, since two equally thin profiles differ by zero.
MATCH_CONTEXT_FEATURES = ("neutral_site", "squad_size_min")


def get_feature_columns(df: pd.DataFrame, representation: str = "all") -> list[str]:
    """Return the engineered feature columns plus match context.

    Args:
        representation: "all" keeps `_a`, `_b` and `_diff`; "diff" keeps only
            the differences. `_diff` is exactly `_a - _b`, so the triple is
            collinear by construction and costs 3x the dimensionality for no
            extra information about which side is stronger. Measured over 5
            seeds, dropping to differences alone holds accuracy (78.66% vs
            78.93%) while cutting seed-to-seed spread from 1.89 to 0.60 -- the
            absolute levels mostly contribute variance. See config
            `features.representation`.
    """
    exclude = {"home_team", "away_team", "year", "profile_year", "date",
               "tournament", "label", "home_score", "away_score", "net_score",
               "neutral", "squad_size_a", "squad_size_b", "squad_size_diff"}

    if representation == "diff":
        suffixes = ("_diff",)
    elif representation == "all":
        suffixes = ("_a", "_b", "_diff")
    else:
        raise ValueError(f"unknown feature representation {representation!r}; "
                         "expected 'all' or 'diff'")

    suffixed = [c for c in df.columns if c not in exclude and c.endswith(suffixes)]
    return suffixed + [c for c in MATCH_CONTEXT_FEATURES if c in df.columns]


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, scaler_type: str = "standard"):
    """Fit a scaler on the training set only, then transform both splits."""
    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def apply_pca(X_train_scaled, X_test_scaled, variance_threshold: float = 0.95):
    """Fit PCA on the training set to retain `variance_threshold` of the
    explained variance, then transform both splits. Per the paper, PCA is
    applied "when beneficial" -- compare downstream CV accuracy with/without
    this step during Phase 5 model selection."""
    pca = PCA(n_components=variance_threshold, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    return X_train_pca, X_test_pca, pca
