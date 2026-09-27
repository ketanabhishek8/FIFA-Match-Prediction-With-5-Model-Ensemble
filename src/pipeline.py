"""End-to-end pipeline: connect real data -> engineer features -> train the
5-model majority-vote ensemble -> benchmark against the WWR baseline.

Run with: python -m src.pipeline
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config, set_seed
from src.data.clean import build_all_team_year_profiles, clean_player_data, filter_national_team_squad
from src.data.collect_matches import filter_world_cup_matches, load_match_data
from src.data.collect_players import load_player_data
from src.evaluation.evaluate import (
    agreement_vs_accuracy,
    challenging_cases_accuracy,
    classification_report_metrics,
    feature_importance_from_pca,
    full_comparison_report,
    mcnemar_test,
    per_model_accuracy,
    roc_auc,
)
from src.features.build_features import (
    apply_pca,
    build_match_features,
    get_feature_columns,
    mirror_feature_frame,
    scale_features,
)
from src.features.history_features import HISTORY_FEATURES, build_history_features
from src.models.baseline import WeightedWinRatioBaseline
from src.models.ensemble import build_majority_vote_ensemble
from src.models.train import tune_all_models


def build_team_profiles(cfg: dict) -> pd.DataFrame:
    players = load_player_data(cfg)
    squad = filter_national_team_squad(players)
    squad = clean_player_data(squad)
    return build_all_team_year_profiles(squad)


def split_holdout(cfg: dict, feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the feature frame into (development, holdout) by year.

    The holdout is not a test set in the usual sense -- `run_seed` already
    carves one of those out of development for each seed. It is insurance
    against everything those splits have been used for: every configuration
    compared, kept or discarded on them. See `evaluation.holdout_years`.
    """
    years = cfg["evaluation"].get("holdout_years") or []
    if not years:
        return feat, feat.iloc[0:0]
    is_holdout = feat["year"].isin(years)
    return feat[~is_holdout].copy(), feat[is_holdout].copy()


def build_match_dataset(cfg: dict, profiles: pd.DataFrame, apply_holdout: bool = True,
                        drop_draws: bool = True):
    """Returns (feature-engineered matches in project scope, full WC match
    history, history state).

    The WC history is kept around so the baseline can be fit on all available
    World Cup history, not just the years with player-feature data. The history
    state is the running team-strength snapshot after the whole record, which
    inference needs to build the same history features training saw.

    `drop_draws=False` keeps drawn matches, for the scoreline model -- see
    `build_match_features`. Everything else about the frame is identical, so the
    two tasks are built from the same join, the same history features and the
    same holdout.
    """
    matches = load_match_data(cfg)
    wc_all_history = filter_world_cup_matches(
        matches, include_qualifiers=cfg["data"].get("include_qualifiers", False)
    )
    wc_scope = wc_all_history[wc_all_history["year"].isin(cfg["data"]["years"])]
    feat = build_match_features(
        wc_scope, profiles,
        edition_aware=cfg["features"].get("edition_aware_join", False),
        drop_draws=drop_draws,
    )

    history_state = None
    if cfg["features"].get("history_features", False):
        # Computed from the FULL international record rather than World Cup
        # matches alone: Elo and form need volume, and a team's friendlies and
        # continental fixtures are evidence about it even though the baseline
        # never looks at them. As-of-date by construction, so widening the pool
        # adds information without adding leakage.
        history_feats, history_state = build_history_features(
            feat, matches,
            wwr_m=cfg["baseline"]["wwr_m"],
            form_window=cfg["features"].get("history_form_window", 10),
            elo_k=cfg["features"].get("history_elo_k", 20),
            elo_home_advantage=cfg["features"].get("history_elo_home_advantage", 60),
            elo_competition_weighted=cfg["features"].get("history_elo_competition_weighted", False),
            elo_goal_difference_weighted=cfg["features"].get("history_elo_goal_difference_weighted", False),
        )
        feat = pd.concat([feat, history_feats], axis=1)
        print(f"History features: {history_feats.shape[1]} added "
              f"({', '.join(HISTORY_FEATURES)}), as of each match date")

    if apply_holdout:
        feat, holdout = split_holdout(cfg, feat)
        if len(holdout):
            print(f"Holdout: {len(holdout)} matches from "
                  f"{cfg['evaluation']['holdout_years']} withheld -- "
                  f"{len(feat)} available for development")

    return feat, wc_all_history, history_state


@dataclass
class PreparedSplit:
    """Everything the feature-prep step produces for one train/test split."""
    idx_train: pd.Index          # after the thin-squad filter, which drops training rows
    idx_test: pd.Index
    X_train: np.ndarray          # scaled, and PCA-projected when configured
    X_test: np.ndarray
    mirror_mask: np.ndarray      # bool over the ORIGINAL idx_train, marking mirrored rows
    train_mean: pd.Series
    scaler: object
    pca: object | None


def prepare_split(cfg: dict, feat: pd.DataFrame, feature_cols: list[str],
                  idx_train: pd.Index, idx_test: pd.Index,
                  verbose: bool = True) -> PreparedSplit:
    """Impute -> mirror -> scale -> PCA for one split, every step fit on the
    training rows only.

    Shared by the classifier ensemble and the scoreline model so the two cannot
    drift apart: this is the leakage-sensitive part of the pipeline (fit the
    scaler on everything, or mirror before splitting, and the numbers get better
    for the wrong reason), and it should exist exactly once.

    Targets are deliberately not handled here. The two tasks carry different
    ones through the side-swap -- the classifier flips its label, the score
    model swaps home and away goals -- so `mirror_mask` is returned and each
    caller applies it to whatever it is predicting.
    """
    X = feat[feature_cols]

    min_squad = cfg["features"].get("min_squad_size", 0)
    if min_squad:
        # Training rows only. The test set keeps its thin-profile matches on
        # purpose: they are part of the population the model is asked about, so
        # filtering them from the evaluation would flatter the score rather than
        # improve the model.
        thin = feat.loc[idx_train, "squad_size_min"] < min_squad
        idx_train = idx_train[~thin.to_numpy()]
        if verbose:
            print(f"Squad filter: dropped {int(thin.sum())} training rows with a "
                  f"profile under {min_squad} players -> {len(idx_train)} remain")

    # Impute from the TRAINING split only. Computing the mean over the full
    # frame first would fold test-set values into the training data, the same
    # leak that scaling/PCA below are careful to avoid by fitting on train only.
    train_mean = X.loc[idx_train].mean()
    X_train, X_test = X.loc[idx_train].fillna(train_mean), X.loc[idx_test].fillna(train_mean)

    mirror_mask = np.zeros(len(idx_train), dtype=bool)
    if cfg["features"].get("mirror_training_rows", False):
        # Only neutral-venue rows. At a neutral venue which team results.csv
        # lists as "home" is arbitrary (48.6% home wins, near a coin flip), so a
        # side-swapped copy is a second true observation. Qualifiers are not
        # neutral and the home side really does win 63.1% of them -- mirroring
        # those would assert the away team had home advantage, injecting a
        # falsehood, and measurably costs ~2 points of accuracy.
        #
        # Strictly after the split, too: mirroring first would put a row and its
        # own copy on opposite sides, i.e. the same match in train and test.
        # Imputing first keeps the fill value the real training mean.
        mirror_mask = feat.loc[idx_train, "neutral_site"].astype(bool).to_numpy()
        X_train = pd.concat(
            [X_train, mirror_feature_frame(X_train[mirror_mask])], ignore_index=True
        )
        if verbose:
            print(f"Mirroring: +{int(mirror_mask.sum())} side-swapped neutral-venue rows "
                  f"-> {len(X_train)} training rows")

    X_train_s, X_test_s, scaler = scale_features(X_train, X_test, cfg["features"]["scaler"])
    pca = None
    if cfg["features"]["use_pca"]:
        X_train_f, X_test_f, pca = apply_pca(X_train_s, X_test_s, cfg["features"]["pca_variance_threshold"])
        if verbose:
            print(f"PCA: {X_train_f.shape[1]} components kept of {X_train_s.shape[1]} "
                  f"(>= {cfg['features']['pca_variance_threshold']:.0%} variance)")
    else:
        X_train_f, X_test_f = X_train_s, X_test_s

    return PreparedSplit(idx_train=idx_train, idx_test=idx_test,
                         X_train=X_train_f, X_test=X_test_f, mirror_mask=mirror_mask,
                         train_mean=train_mean, scaler=scaler, pca=pca)


def mirror_targets(values: pd.Series, mirror_mask: np.ndarray, swap_with: pd.Series | None = None,
                   flip: bool = False) -> pd.Series:
    """Extend a training target to cover the mirrored rows `prepare_split` added.

    `flip=True` gives the classifier's `1 - label`; `swap_with` gives the score
    model's home/away exchange. Exactly one of the two is meaningful per call.
    """
    if not mirror_mask.any():
        return values.reset_index(drop=True)
    extra = (1 - values[mirror_mask]) if flip else swap_with[mirror_mask]
    return pd.concat([values, extra], ignore_index=True)


def run_seed(cfg: dict, feat: pd.DataFrame, wc_all_history: pd.DataFrame, seed: int,
             idx_train=None, idx_test=None) -> pd.DataFrame:
    """Run one train/test split + fit/eval cycle at the given seed. Returns the
    comparison report (overall / high-scoring / low-scoring accuracy rows).

    Args:
        idx_train, idx_test: explicit row indices, for callers that need a split
            other than the stratified random one -- `src/holdout.py` passes the
            whole development set and the withheld years. Supplying these keeps
            imputation, mirroring, scaling, PCA, tuning and the baseline
            identical to a normal run, which is the point: the holdout number
            has to describe the same pipeline that produced everything else.
    """
    set_seed(seed)

    feature_cols = get_feature_columns(feat, cfg["features"].get("representation", "all"))
    y = feat["label"]

    if idx_train is None or idx_test is None:
        idx_train, idx_test = train_test_split(
            feat.index, test_size=cfg["models"]["test_size"],
            random_state=seed, stratify=y,
        )

    split = prepare_split(cfg, feat, feature_cols, idx_train, idx_test)
    idx_train, train_mean = split.idx_train, split.train_mean
    scaler, pca = split.scaler, split.pca
    X_train_f, X_test_f = split.X_train, split.X_test

    y_train = mirror_targets(y.loc[idx_train], split.mirror_mask, flip=True)
    y_test = y.loc[idx_test]
    test_rows = feat.loc[idx_test]

    print(f"\nTraining ensemble on {len(X_train_f)} matches, testing on {len(X_test_f)}...")
    tuned = tune_all_models(X_train_f, y_train, cfg, seed)

    # tune_all_models() returns each model's GridSearchCV.best_estimator_,
    # already refit on the full training set -- so per-model predictions are
    # free here, before they're thrown into the majority vote.
    per_model_pred = {name: est.predict(X_test_f) for name, est in tuned.items()}
    per_model_proba = {name: est.predict_proba(X_test_f)[:, 1] for name, est in tuned.items()}
    ml_proba = np.mean(list(per_model_proba.values()), axis=0)  # soft-vote average, for ROC-AUC only

    ensemble = build_majority_vote_ensemble(tuned, voting="hard")
    ensemble.fit(X_train_f, y_train)
    ml_pred = ensemble.predict(X_test_f)

    # Baseline: fit on ALL historical World Cup matches, excluding the exact
    # matches held out for testing (so it isn't scored on rows it was fit on).
    test_keys = set(zip(test_rows["date"], test_rows["home_team"], test_rows["away_team"]))
    baseline_history = wc_all_history[
        ~wc_all_history.apply(lambda r: (r["date"], r["home_team"], r["away_team"]) in test_keys, axis=1)
    ]
    baseline = WeightedWinRatioBaseline(
        min_head_to_head_matches=cfg["baseline"]["min_head_to_head_matches"],
        wwr_m=cfg["baseline"]["wwr_m"],
    )
    baseline.fit(baseline_history)
    baseline_winner = baseline.predict(test_rows)
    baseline_pred = (baseline_winner == test_rows["home_team"]).astype(int).values

    total_goals = test_rows["home_score"] + test_rows["away_score"]
    report = full_comparison_report(
        y_test.values, baseline_pred, ml_pred, total_goals,
        cfg["evaluation"]["high_scoring_goal_threshold"],
    )
    print("\n" + report.to_string())

    challenging = challenging_cases_accuracy(y_test.values, baseline_pred, ml_pred)
    if challenging["n_challenging_cases"]:
        print(f"\nChallenging cases (baseline wrong): {challenging['n_challenging_cases']}, "
              f"ML accuracy on those: {challenging['ml_accuracy_on_challenging']:.2%}")
    else:
        print("\nNo challenging cases in this test split.")

    per_model_acc = per_model_accuracy(y_test.values, per_model_pred)
    print("\nPer-model accuracy (before majority vote):\n" + per_model_acc.to_string())

    ensemble_metrics = classification_report_metrics(y_test.values, ml_pred)
    baseline_metrics = classification_report_metrics(y_test.values, baseline_pred)
    ensemble_auc = roc_auc(y_test.values, ml_proba)
    print(f"\nEnsemble: precision={ensemble_metrics['precision']:.3f} "
          f"recall={ensemble_metrics['recall']:.3f} f1={ensemble_metrics['f1']:.3f} "
          f"roc_auc={ensemble_auc:.3f}")
    print(f"Baseline: precision={baseline_metrics['precision']:.3f} "
          f"recall={baseline_metrics['recall']:.3f} f1={baseline_metrics['f1']:.3f}")

    significance = mcnemar_test(y_test.values, baseline_pred, ml_pred)
    print(f"\nMcNemar's test (baseline vs. ensemble): b={significance['b']} c={significance['c']} "
          f"p={significance['p_value']:.4f}"
          + ("  (significant at p<0.05)" if significance["p_value"] < 0.05 else "  (not significant at p<0.05)"))

    agreement_table = agreement_vs_accuracy(per_model_pred, y_test.values)
    print("\nEnsemble accuracy by vote agreement (of 5 models):\n" + agreement_table.to_string())

    feature_importance = None
    if pca is not None:
        tree_importances = np.mean(
            [tuned["random_forest"].feature_importances_, tuned["xgboost"].feature_importances_], axis=0
        )
        feature_importance = feature_importance_from_pca(tree_importances, pca.components_, feature_cols)
        print("\nTop features (Random Forest + XGBoost, mapped back through PCA):\n"
              + feature_importance.to_string())

    report.attrs.update(
        y_test=y_test.values, baseline_pred=baseline_pred, ml_pred=ml_pred, ml_proba=ml_proba,
        per_model_pred=per_model_pred, per_model_accuracy=per_model_acc,
        ensemble_metrics=ensemble_metrics, baseline_metrics=baseline_metrics, ensemble_auc=ensemble_auc,
        mcnemar=significance, agreement_table=agreement_table, feature_importance=feature_importance,
        pca_explained_variance=pca.explained_variance_ratio_ if pca is not None else None,
        ensemble=ensemble, scaler=scaler, pca=pca,
        feature_cols=feature_cols, train_mean=train_mean,
        n_train=len(X_train_f), n_test=len(X_test_f),
    )
    return report


def run(seeds: list[int] | None = None) -> None:
    """Run the pipeline once (default) or over multiple seeds if `seeds` is
    given -- profiles/features are built once and reused, since only the
    train/test split and model fitting are seed-dependent."""
    cfg = load_config()
    profiles = build_team_profiles(cfg)
    feat, wc_all_history, _history_state = build_match_dataset(cfg, profiles)
    print(f"Team-year profiles: {len(profiles)} | Matches usable for modeling: {len(feat)}")

    seeds = seeds or [cfg["project"]["random_state"]]
    reports = {}
    for seed in seeds:
        print(f"\n{'=' * 20} seed={seed} {'=' * 20}")
        reports[seed] = run_seed(cfg, feat, wc_all_history, seed)

    if len(seeds) > 1:
        # Build from plain dicts rather than pd.concat: each report carries
        # numpy arrays in .attrs (y_test, ml_pred, ...), and pandas >= 3 has
        # concat compare `obj.attrs == attrs` to decide whether to propagate
        # them -- an array-vs-array `==` returns an array, raising "truth value
        # of an array ... is ambiguous". Going through .to_dict() drops attrs
        # before pandas ever looks at them.
        overall = pd.DataFrame({s: r["Overall Accuracy"].to_dict() for s, r in reports.items()}).T
        print(f"\n{'=' * 20} summary across {len(seeds)} seeds {'=' * 20}")
        print(overall.to_string())
        print("\n" + overall.agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    run(seeds=[0, 1, 17, 42, 123])
