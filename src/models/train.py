"""Training orchestration, per the paper's Section III.B.3 procedure:

  1. Standard train/test split.
  2. k-fold CV (k=5) on the training set, per model type, to select the best
     hyperparameters (GridSearchCV/RandomizedSearchCV over PARAM_GRIDS).
  3. Refit each model type with its best hyperparameters on the *full*
     training set.
  4. Wrap the 5 best-per-type models in the majority-voting ensemble.
  5. Evaluate on the held-out test set (see src/evaluation/evaluate.py).

This is a runnable skeleton: the pieces (config, features, models, saving) are
wired together, but it depends on src/data + src/features producing real
X_train/X_test/y_train/y_test first (Phase 2-3).
"""
from __future__ import annotations

import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.config import load_config, resolve_path, set_seed
from src.models.ensemble import PARAM_GRIDS, build_base_estimators, build_majority_vote_ensemble


def tune_all_models(X_train, y_train, cfg: dict, seed: int | None = None) -> dict:
    """Run k-fold CV hyperparameter search for each of the 5 model types and
    return {name: best_estimator}.

    Args:
        seed: seeds both the CV fold shuffle and each estimator's own
            random_state. Callers running the pipeline over multiple seeds MUST
            pass the current one -- falling back to cfg's fixed random_state
            would hold model randomness constant across "different" seeds, so
            the resulting spread would only reflect train/test split variance.
    """
    seed = cfg["project"]["random_state"] if seed is None else seed
    cv = StratifiedKFold(n_splits=cfg["models"]["cv_folds"], shuffle=True, random_state=seed)

    base_estimators = build_base_estimators(random_state=seed)
    base_estimators = {name: est for name, est in base_estimators.items() if name in cfg["models"]["classifiers"]}
    best_estimators = {}

    for name, estimator in base_estimators.items():
        search = GridSearchCV(
            estimator=estimator,
            param_grid=PARAM_GRIDS[name],
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best_estimators[name] = search.best_estimator_
        print(f"[{name}] best CV accuracy: {search.best_score_:.4f} | params: {search.best_params_}")

    return best_estimators


def run_training_pipeline(X, y, cfg: dict | None = None):
    """End-to-end: split -> tune each model -> majority-vote ensemble -> fit ->
    return (ensemble, X_test, y_test) for downstream evaluation."""
    cfg = cfg or load_config()
    set_seed(cfg["project"]["random_state"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["models"]["test_size"],
        random_state=cfg["project"]["random_state"],
        stratify=y,
    )

    best_estimators = tune_all_models(X_train, y_train, cfg)
    ensemble = build_majority_vote_ensemble(best_estimators, voting="hard")
    ensemble.fit(X_train, y_train)

    output_dir = resolve_path(cfg["evaluation"]["output_dir"]) / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, output_dir / "majority_vote_ensemble.joblib")

    return ensemble, X_test, y_test


if __name__ == "__main__":
    raise SystemExit(
        "This script expects X, y from the feature pipeline (Phase 2-3) to be "
        "assembled first -- see src/data/ and src/features/build_features.py. "
        "Wire up a notebook or a small __main__ block here once that data exists."
    )
