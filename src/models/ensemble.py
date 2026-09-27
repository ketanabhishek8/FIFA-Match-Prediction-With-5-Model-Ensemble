"""The 5-model ensemble + majority voting, per the paper's Section III.B.2 and
III.C.1: Logistic Regression, Random Forest, XGBoost, AdaBoost, KNN.

Model rationale (paper):
  - Logistic Regression: linear, interpretable baseline-within-the-ensemble.
  - Random Forest / XGBoost / AdaBoost: tree-based, strong on tabular data.
  - KNN: adds diversity via a fundamentally different decision boundary.

Each model returns a Team A / Team B win probability; final prediction is via
majority (hard) voting across the 5 classifiers.

Hyperparameter grids below are starting points for the paper's described
process: 5-fold CV on the training set to select the best hyperparameters per
model type (Section III.B.3), *before* fitting majority voting. They'll need
tuning once real feature data (dimensionality, class balance) is available.
"""
from __future__ import annotations

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Starting hyperparameter grids for GridSearchCV / RandomizedSearchCV (Phase 5).
# Keep small initially -- expand once compute budget / dataset size are known.
PARAM_GRIDS = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs"],
        "max_iter": [1000],
    },
    "random_forest": {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5],
    },
    "xgboost": {
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
    },
    "adaboost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.5, 1.0, 1.5],
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 11, 15],
        "weights": ["uniform", "distance"],
    },
}


def build_base_estimators(random_state: int = 42) -> dict:
    """Return a fresh, un-tuned instance of each of the 5 model types."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(random_state=random_state),
        "xgboost": XGBClassifier(random_state=random_state, eval_metric="logloss"),
        "adaboost": AdaBoostClassifier(random_state=random_state),
        "knn": KNeighborsClassifier(),
    }


def build_majority_vote_ensemble(tuned_estimators: dict, voting: str = "hard") -> VotingClassifier:
    """Wrap the 5 (already-tuned, best-hyperparameter) estimators in a hard
    VotingClassifier to reproduce the paper's majority-voting inference step
    (Section III.C.1).

    Args:
        tuned_estimators: dict of {name: fitted_or_unfitted_estimator}, e.g. the
            output of running GridSearchCV per model type in train.py and taking
            `.best_estimator_` for each.
    """
    estimators = [(name, est) for name, est in tuned_estimators.items()]
    return VotingClassifier(estimators=estimators, voting=voting)
