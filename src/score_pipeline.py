"""Scoreline pipeline: build features -> fit two Poisson regressions -> score
the resulting scoreline distribution against a no-skill floor.

The counterpart to `src/pipeline.py`, sharing its data build, its split, and its
feature prep (`prepare_split`). The only difference is the target: goals scored
by each side, rather than which side won. Draws are kept -- see
`build_match_dataset(drop_draws=False)`.

Run with: python -m src.score_pipeline
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config, set_seed
from src.evaluation.evaluate import score_model_report
from src.features.build_features import get_feature_columns
from src.models.score_model import (
    ConstantPoissonBaseline,
    PoissonScoreModel,
    outcome_index,
    outcome_proba_from_grid,
    scoreline_grid,
)
from src.pipeline import build_match_dataset, build_team_profiles, mirror_targets, prepare_split


def build_score_model(cfg: dict) -> PoissonScoreModel:
    """Construct the model from config, so the pipeline, the tests and (later)
    the API all instantiate it the same way."""
    sm = cfg.get("score_model", {})
    return PoissonScoreModel(
        max_goals=sm.get("max_goals", 10),
        alpha=sm.get("alpha", 1e-3),
        dixon_coles=sm.get("dixon_coles", True),
        rho_grid_points=sm.get("rho_grid_points", 1401),
        rho_bounds=tuple(sm.get("rho_bounds", (-0.35, 0.35))),
    )


def run_seed_scores(cfg: dict, feat: pd.DataFrame, seed: int,
                    idx_train=None, idx_test=None, verbose: bool = True) -> pd.DataFrame:
    """Fit and evaluate the scoreline model for one split. Returns a report
    indexed by model name, with the scoreline metrics as columns.

    Args:
        idx_train, idx_test: explicit indices, matching `run_seed`'s escape
            hatch, so a future `score_holdout` can score the withheld years
            through exactly this path.
    """
    set_seed(seed)
    feature_cols = get_feature_columns(feat, cfg["features"].get("representation", "all"))
    home, away = feat["home_score"], feat["away_score"]

    if idx_train is None or idx_test is None:
        # Stratify on the three-way outcome, not the binary label: with draws
        # kept, `label` collapses draws into the away-win class and would let a
        # split drift on draw rate.
        idx_train, idx_test = train_test_split(
            feat.index, test_size=cfg["models"]["test_size"], random_state=seed,
            stratify=outcome_index(home, away),
        )

    split = prepare_split(cfg, feat, feature_cols, idx_train, idx_test, verbose=verbose)
    idx_train = split.idx_train

    # The mirror swaps sides, so it swaps which target is which -- the score
    # model's equivalent of the classifier flipping its label.
    y_home = mirror_targets(home.loc[idx_train], split.mirror_mask, swap_with=away.loc[idx_train])
    y_away = mirror_targets(away.loc[idx_train], split.mirror_mask, swap_with=home.loc[idx_train])

    test_rows = feat.loc[idx_test]
    y_home_test, y_away_test = test_rows["home_score"].values, test_rows["away_score"].values

    model = build_score_model(cfg).fit(split.X_train, y_home, y_away)
    baseline = ConstantPoissonBaseline(max_goals=model.max_goals).fit(split.X_train, y_home, y_away)

    if verbose:
        print(f"\nScore model: fit on {len(split.X_train)} rows, testing on {len(split.X_test)}")
        if model.dixon_coles:
            bound = "  (AT FEASIBILITY BOUND -- check for an extreme fitted rate)" if model.rho_at_bound_ else ""
            print(f"Dixon-Coles rho = {model.rho_:+.4f} "
                  f"(fit on {model.n_low_score_rows_} low-score training rows){bound}")
        print(f"Pearson dispersion on test: {model.dispersion(split.X_test, y_home_test, y_away_test):.3f} "
              f"(~1.0 means Poisson is correctly specified)")

    # Report the uncorrected grid alongside, so the Dixon-Coles claim in
    # config.yaml can be re-checked from a normal run rather than taken on faith.
    lam, mu = model.expected_goals(split.X_test).T
    variants = {
        "Poisson + Dixon-Coles" if model.dixon_coles else "Poisson": model.predict_grid(split.X_test),
    }
    if model.dixon_coles:
        variants["Poisson (independent)"] = scoreline_grid(lam, mu, model.max_goals, rho=0.0)
    variants["Baseline (league average)"] = baseline.predict_grid(split.X_test)

    rows = {}
    for name, grid in variants.items():
        width = grid.shape[1]
        flat = grid.reshape(len(grid), -1).argmax(1)
        pred_scores = np.column_stack([flat // width, flat % width])
        eg = (baseline.expected_goals(split.X_test) if name.startswith("Baseline")
              else model.expected_goals(split.X_test))
        rows[name] = score_model_report(outcome_proba_from_grid(grid), pred_scores,
                                        y_home_test, y_away_test, expected_goals=eg)

    report = pd.DataFrame(rows).T
    if verbose:
        print("\n" + report.to_string(float_format=lambda v: f"{v:.4f}"))

    report.attrs.update(model=model, baseline=baseline, rho=model.rho_,
                        scaler=split.scaler, pca=split.pca, train_mean=split.train_mean,
                        feature_cols=feature_cols,
                        n_train=len(split.X_train), n_test=len(split.X_test))
    return report


def run(seeds: list[int] | None = None) -> None:
    cfg = load_config()
    if not cfg.get("score_model", {}).get("enabled", True):
        print("score_model.enabled is false -- nothing to do")
        return

    profiles = build_team_profiles(cfg)
    feat, _wc_all_history, _history_state = build_match_dataset(cfg, profiles, drop_draws=False)
    n_draws = int((feat["home_score"] == feat["away_score"]).sum())
    print(f"Team-year profiles: {len(profiles)} | Matches usable for scoring: {len(feat)} "
          f"({n_draws} draws, {len(feat) - n_draws} decided)")

    seeds = seeds or [cfg["project"]["random_state"]]
    reports = {}
    for seed in seeds:
        print(f"\n{'=' * 20} seed={seed} {'=' * 20}")
        reports[seed] = run_seed_scores(cfg, feat, seed)

    if len(seeds) > 1:
        print(f"\n{'=' * 20} summary across {len(seeds)} seeds {'=' * 20}")
        # Plain dicts rather than pd.concat: the reports carry fitted models in
        # .attrs, and pandas >= 3 compares attrs with `==` when deciding whether
        # to propagate them, which raises on array-valued entries. See the same
        # note in pipeline.run().
        for metric in ["RPS", "Log Loss", "Derived Win Accuracy", "Exact Scoreline",
                       "Goal MAE", "Predicted Draw Rate"]:
            table = pd.DataFrame({s: r[metric].to_dict() for s, r in reports.items()}).T
            summary = table.agg(["mean", "std"]).T
            print(f"\n{metric}:")
            for name, row in summary.iterrows():
                print(f"   {name:28s} {row['mean']:.4f} +/- {row['std']:.4f}")

        actual = np.mean([r["Actual Draw Rate"].iloc[0] for r in reports.values()])
        print(f"\n   (actual draw rate {actual:.4f} -- compare against Predicted Draw Rate above)")


if __name__ == "__main__":
    run(seeds=[0, 1, 17, 42, 123])
