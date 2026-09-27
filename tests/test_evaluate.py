"""Sanity tests for the new evaluation metrics (src/evaluation/evaluate.py).

Run with: pytest tests/test_evaluate.py -v
"""
import numpy as np

from src.evaluation.evaluate import (
    agreement_vs_accuracy,
    classification_report_metrics,
    feature_importance_from_pca,
    mcnemar_test,
    per_model_accuracy,
    roc_auc,
)


def test_per_model_accuracy_matches_hand_count():
    y_true = np.array([1, 0, 1, 0])
    per_model_pred = {
        "perfect": np.array([1, 0, 1, 0]),
        "always_one": np.array([1, 1, 1, 1]),
    }
    acc = per_model_accuracy(y_true, per_model_pred)
    assert acc["perfect"] == 1.0
    assert acc["always_one"] == 0.5


def test_classification_report_metrics_perfect_predictions():
    y_true = np.array([1, 0, 1, 0])
    metrics = classification_report_metrics(y_true, y_true)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_roc_auc_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(y_true, y_proba) == 1.0


def test_mcnemar_test_no_disagreement_is_not_significant():
    y_true = np.array([1, 0, 1, 0])
    result = mcnemar_test(y_true, y_true, y_true)  # identical predictions -> b=c=0
    assert result["b"] == 0 and result["c"] == 0
    assert result["p_value"] == 1.0


def test_mcnemar_test_counts_asymmetric_disagreement():
    y_true = np.array([1, 1, 1, 1, 1, 1])
    pred_a = np.array([0, 0, 0, 0, 0, 1])  # wrong 5x
    pred_b = np.array([1, 1, 1, 1, 1, 1])  # always right
    result = mcnemar_test(y_true, pred_a, pred_b)
    assert result["b"] == 5  # a wrong, b right
    assert result["c"] == 0  # a right, b wrong


def test_agreement_vs_accuracy_unanimous_group_is_always_correct():
    y_true = np.array([1, 1, 0])
    per_model_pred = {f"m{i}": np.array([1, 1, 0]) for i in range(5)}  # all models always correct
    table = agreement_vs_accuracy(per_model_pred, y_true)
    assert table.loc[5, "accuracy"] == 1.0
    assert table.loc[5, "n_matches"] == 3


def test_feature_importance_from_pca_sums_to_one_and_ranks_dominant_feature():
    # 2 components, 3 original features. Component 0 loads entirely on
    # feature "x" and carries all the model importance -> "x" should dominate.
    component_importances = np.array([1.0, 0.0])
    pca_components = np.array([
        [1.0, 0.0, 0.0],   # component 0: pure "x"
        [0.0, 1.0, 1.0],   # component 1: split "y"/"z"
    ])
    result = feature_importance_from_pca(
        component_importances, pca_components, feature_names=["x", "y", "z"], top_n=3
    )
    assert np.isclose(result.sum(), 1.0)
    assert result.idxmax() == "x"
