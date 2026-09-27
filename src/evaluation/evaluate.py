"""Evaluation, mirroring the paper's Section IV analysis so results are
directly comparable to Table I:

  - Overall accuracy
  - Accuracy on high-scoring vs. low-scoring matches
  - Accuracy on "challenging cases" -- matches the baseline got wrong
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# The ordered outcome convention (home win / draw / away win) belongs to the
# scoreline model, so it is defined there and reused here rather than restated.
from src.models.score_model import outcome_index


def overall_accuracy(y_true, y_pred) -> float:
    return accuracy_score(y_true, y_pred)


def per_model_accuracy(y_true, per_model_pred: dict[str, np.ndarray]) -> pd.Series:
    """Accuracy of each individual base classifier, before the majority vote
    -- shows which models in the ensemble are actually pulling weight."""
    return pd.Series({name: accuracy_score(y_true, pred) for name, pred in per_model_pred.items()})


def classification_report_metrics(y_true, y_pred) -> dict[str, float]:
    """Precision/recall/F1 (binary, positive class = home win), which accuracy
    alone hides whenever the classes aren't ~50/50 (home wins are more common
    than away wins in this dataset)."""
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def roc_auc(y_true, y_proba) -> float:
    """ROC-AUC from predicted probabilities (needs soft-vote scores, not hard
    vote labels -- see pipeline.py's ml_proba)."""
    return roc_auc_score(y_true, y_proba)


def mcnemar_test(y_true, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    """McNemar's test on paired predictions: is the accuracy difference
    between two models on the *same* test set real, or within noise?

    b = model A wrong, model B right; c = model A right, model B wrong.
    Uses the continuity-corrected chi-square statistic (1 df), the standard
    form for McNemar's test.
    """
    y_true, pred_a, pred_b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    a_correct, b_correct = pred_a == y_true, pred_b == y_true
    b = int(np.sum(~a_correct & b_correct))
    c = int(np.sum(a_correct & ~b_correct))

    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(chi2.sf(statistic, df=1))
    return {"b": b, "c": c, "statistic": float(statistic), "p_value": p_value}


def agreement_vs_accuracy(per_model_pred: dict[str, np.ndarray], y_true) -> pd.DataFrame:
    """For each test match, how many of the 5 base models agreed with the
    majority, and was the majority (ensemble) prediction actually correct?
    Answers: are unanimous predictions more trustworthy than split votes?
    """
    y_true = np.asarray(y_true)
    preds = np.stack(list(per_model_pred.values()), axis=1)  # (n_samples, n_models)
    n_models = preds.shape[1]

    majority_vote = (preds.sum(axis=1) > n_models / 2).astype(int)
    agreement = np.maximum(preds.sum(axis=1), n_models - preds.sum(axis=1))  # votes for the winning side
    correct = majority_vote == y_true

    df = pd.DataFrame({"agreement": agreement, "correct": correct})
    return df.groupby("agreement")["correct"].agg(["mean", "count"]).rename(
        columns={"mean": "accuracy", "count": "n_matches"}
    )


def feature_importance_from_pca(
    component_importances: np.ndarray, pca_components: np.ndarray, feature_names: list[str], top_n: int = 15
) -> pd.Series:
    """Map a tree model's feature_importances_ (computed in PCA-component
    space) back to the original engineered features, so results are
    interpretable -- PCA components have no real-world meaning on their own.

    Approximation: each component's importance is distributed across the
    original features in proportion to that feature's *squared* loading on
    the component (pca_components_ rows) -- squared loadings are the
    variance-share form and sum to 1 per component, matching how much of that
    component each feature actually accounts for. Summed across components
    and renormalized. This is the standard way to trace tree importances back
    through a PCA step; it is an approximation, not an exact attribution.

    Using absolute loadings instead (an earlier version of this function did)
    over-credits features with many small, weakly-loading contributions and
    flattens the whole ranking into near-indistinguishable bars -- squared
    loadings recover the sharper, more informative signal.
    """
    loadings = pca_components ** 2  # (n_components, n_features_original), sums to 1 per row
    raw_importance = component_importances @ loadings
    raw_importance = raw_importance / raw_importance.sum()
    return pd.Series(raw_importance, index=feature_names).sort_values(ascending=False).head(top_n)


def accuracy_by_scoring_bucket(
    y_true, y_pred, total_goals: pd.Series, high_scoring_threshold: int = 3
) -> dict[str, float]:
    """Split accuracy by whether a match was high-scoring or low-scoring.

    Args:
        total_goals: home_score + away_score per match, aligned index-wise with
            y_true/y_pred.
        high_scoring_threshold: combined-goals cutoff for "high-scoring"
            (config: evaluation.high_scoring_goal_threshold). The paper doesn't
            state its exact cutoff -- pick one from the goal distribution during
            EDA and document it in the report.
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "total_goals": total_goals})
    high = df[df["total_goals"] >= high_scoring_threshold]
    low = df[df["total_goals"] < high_scoring_threshold]

    return {
        "high_scoring_accuracy": accuracy_score(high["y_true"], high["y_pred"]) if len(high) else float("nan"),
        "low_scoring_accuracy": accuracy_score(low["y_true"], low["y_pred"]) if len(low) else float("nan"),
        "n_high_scoring": len(high),
        "n_low_scoring": len(low),
    }


def challenging_cases_accuracy(y_true, baseline_pred, ml_pred) -> dict[str, float]:
    """Isolate matches the baseline model got wrong ("challenging cases" in the
    paper) and report the ML model's accuracy on just that subset -- this is
    where the paper reports 7/28 = 25.00%."""
    df = pd.DataFrame({"y_true": y_true, "baseline_pred": baseline_pred, "ml_pred": ml_pred})
    challenging = df[df["baseline_pred"] != df["y_true"]]

    if challenging.empty:
        return {"n_challenging_cases": 0, "ml_accuracy_on_challenging": float("nan")}

    ml_acc = accuracy_score(challenging["y_true"], challenging["ml_pred"])
    return {
        "n_challenging_cases": len(challenging),
        "n_ml_correct_on_challenging": int((challenging["ml_pred"] == challenging["y_true"]).sum()),
        "ml_accuracy_on_challenging": ml_acc,
    }


def full_comparison_report(
    y_true, baseline_pred, ml_pred, total_goals: pd.Series, high_scoring_threshold: int = 3
) -> pd.DataFrame:
    """Build a Table-I-style comparison DataFrame: Proposed Method vs. Baseline
    across Overall / High-scoring / Low-scoring accuracy, ready to drop into the
    report."""
    rows = []
    for label, preds in [("Proposed Method", ml_pred), ("Baseline Model", baseline_pred)]:
        bucket = accuracy_by_scoring_bucket(y_true, preds, total_goals, high_scoring_threshold)
        rows.append({
            "Evaluation Metric": label,
            "Overall Accuracy": overall_accuracy(y_true, preds),
            "Accuracy (High-scoring)": bucket["high_scoring_accuracy"],
            "Accuracy (Low-scoring)": bucket["low_scoring_accuracy"],
        })
    return pd.DataFrame(rows).set_index("Evaluation Metric")


# --- Scoreline model evaluation -----------------------------------------
#
# The binary task is scored with accuracy; a scoreline model predicts a
# distribution and has to be scored as one. Accuracy on the exact scoreline is
# reportable but close to meaningless on its own -- there are 60 distinct
# scorelines in the data and always guessing the modal 1-0 already gets 10.4%,
# so the number looks catastrophic next to a 79% win-accuracy while describing
# a completely different question. RPS and log-loss are the metrics that
# actually rank these models.


def ranked_probability_score(probs: np.ndarray, obs_idx: np.ndarray) -> float:
    """Mean RPS over ordered outcomes (home win, draw, away win).

    The standard metric for football forecasts: unlike log-loss it is sensitive
    to *how far* a wrong prediction was, so putting mass on an away win when the
    home side won is penalised more than putting it on a draw. Lower is better;
    0 is a perfect confident forecast.
    """
    probs, obs_idx = np.asarray(probs, dtype=float), np.asarray(obs_idx)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(obs_idx)), obs_idx] = 1.0
    cum_p, cum_o = np.cumsum(probs, axis=1), np.cumsum(onehot, axis=1)
    # Last cumulative pair is 1 vs 1 by construction, so it contributes nothing
    # and the normaliser is (categories - 1).
    return float((((cum_p[:, :-1] - cum_o[:, :-1]) ** 2).sum(axis=1) / (probs.shape[1] - 1)).mean())


def outcome_log_loss(probs: np.ndarray, obs_idx: np.ndarray) -> float:
    """Mean negative log-likelihood of the realised outcome."""
    probs, obs_idx = np.asarray(probs, dtype=float), np.asarray(obs_idx)
    realised = probs[np.arange(len(obs_idx)), obs_idx]
    return float(-np.log(np.clip(realised, 1e-12, None)).mean())


def draw_calibration(probs: np.ndarray, obs_idx: np.ndarray) -> dict[str, float]:
    """Predicted vs. actual draw rate -- the specific failure mode of an
    independent-Poisson grid, which cannot represent score dependence at low
    scorelines and so systematically under-predicts draws."""
    probs, obs_idx = np.asarray(probs, dtype=float), np.asarray(obs_idx)
    predicted = float(probs[:, 1].mean())
    actual = float((obs_idx == 1).mean())
    return {"predicted_draw_rate": predicted, "actual_draw_rate": actual,
            "draw_gap": predicted - actual}


def exact_scoreline_accuracy(pred_scores: np.ndarray, home_score, away_score) -> float:
    """Share of matches whose exact scoreline was the model's modal prediction."""
    pred_scores = np.asarray(pred_scores)
    home_score, away_score = np.asarray(home_score), np.asarray(away_score)
    return float(((pred_scores[:, 0] == home_score) & (pred_scores[:, 1] == away_score)).mean())


def derived_win_accuracy(probs: np.ndarray, home_score, away_score) -> dict[str, float]:
    """Win-accuracy implied by the scoreline distribution, on non-draw matches.

    This is the only number directly comparable to the classifier ensemble, and
    only on this subset: the binary task never sees draws, so scoring the score
    model on rows the classifier is not asked about would compare two different
    questions. Draw mass is ignored rather than redistributed -- the comparison
    is "given it was decided, which side did the model favour".
    """
    probs = np.asarray(probs, dtype=float)
    home_score, away_score = np.asarray(home_score), np.asarray(away_score)
    decided = home_score != away_score
    if not decided.any():
        return {"derived_win_accuracy": float("nan"), "n_decided": 0}

    picked_home = probs[decided, 0] > probs[decided, 2]
    actual_home = home_score[decided] > away_score[decided]
    return {"derived_win_accuracy": float((picked_home == actual_home).mean()),
            "n_decided": int(decided.sum())}


def score_model_report(probs: np.ndarray, pred_scores: np.ndarray,
                       home_score, away_score,
                       expected_goals: np.ndarray | None = None) -> pd.Series:
    """Every scoreline metric in one row, for stacking across seeds.

    Args:
        expected_goals: optional (n, 2) array of predicted goal rates, for MAE.
            MAE is scored against the rate rather than the modal scoreline --
            the rate is the model's actual point estimate, and rounding it first
            would measure the rounding as much as the model.
    """
    obs = outcome_index(home_score, away_score)
    out = {
        "RPS": ranked_probability_score(probs, obs),
        "Log Loss": outcome_log_loss(probs, obs),
        "Exact Scoreline": exact_scoreline_accuracy(pred_scores, home_score, away_score),
        "Derived Win Accuracy": derived_win_accuracy(probs, home_score, away_score)["derived_win_accuracy"],
    }
    if expected_goals is not None:
        expected_goals = np.asarray(expected_goals, dtype=float)
        out["Goal MAE"] = float(
            (np.abs(expected_goals[:, 0] - np.asarray(home_score)).mean()
             + np.abs(expected_goals[:, 1] - np.asarray(away_score)).mean()) / 2
        )
    cal = draw_calibration(probs, obs)
    out.update({"Predicted Draw Rate": cal["predicted_draw_rate"],
                "Actual Draw Rate": cal["actual_draw_rate"]})
    return pd.Series(out)
