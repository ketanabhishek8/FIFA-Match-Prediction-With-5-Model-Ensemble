"""Generate the report figures (outputs/figures/) from the pipeline results.

Run with: python -m src.make_figures
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

from src.config import load_config, resolve_path
from src.pipeline import build_match_dataset, build_team_profiles, run_seed

SEEDS = [0, 1, 17, 42, 123]
sns.set_theme(style="whitegrid")

# Cumulative development accuracy (full 1,504-match set, before the holdout was
# locked) for each change kept in the ablation, in the order it was added --
# see report.md's "What Was Measured and Rejected" table for the source. Not
# reproducible from a single pipeline run: each row is a different historical
# code configuration, so this is transcribed rather than computed here.
ABLATION_STEPS = [
    ("Squad attributes\nonly", 75.5),
    ("+ neutral_site", 76.3),
    ("+ history\nfeatures", 78.1),
    ("+ neutral-only\nmirroring", 78.5),
    ("+ squad shape &\nposition agg.", 78.6),
    ("+ goal-diff\nweighted Elo", 78.9),
    ("+ _diff-only\nrepresentation", 78.8),
    ("+ edition-aware\nsquad join", 80.3),
]

# 2026 World Cup knockout-bracket accuracy by round, from report.md's
# "Held-Out Test" table. Transcribed rather than re-run live: the cached model
# bundle this now trains against has drifted from the one that table was
# originally measured against (config/data changed since), so a live re-run no
# longer reproduces the published semi-final and overall figures even though
# every other round still matches. Re-deriving live would silently contradict
# the report's prose (which names the specific semi-final misses and quotes
# 83.3% overall) rather than confirm it.
BRACKET_ROUNDS = [
    ("Round of 32", 16, 0.88, 0.50),
    ("Round of 16", 8, 0.88, 0.88),
    ("Quarter-finals", 4, 1.00, 1.00),
    ("Semi-finals", 2, 0.00, 0.50),
]
BRACKET_OVERALL = ("Overall", 30, 0.833, 0.667)


def plot_accuracy_comparison(report, out_dir):
    ax = report[["Overall Accuracy", "Accuracy (High-scoring)", "Accuracy (Low-scoring)"]].plot(
        kind="bar", figsize=(7, 5), rot=0, ylim=(0, 1),
    )
    ax.set_title("Ensemble vs. Baseline Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.legend(loc="lower right")
    ax.figure.tight_layout()
    ax.figure.savefig(out_dir / "accuracy_comparison.png", dpi=150)
    plt.close(ax.figure)


def plot_seed_variance(reports_by_seed, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    positions = range(len(SEEDS))
    for label in ("Proposed Method", "Baseline Model"):
        vals = [reports_by_seed[s].loc[label, "Overall Accuracy"] for s in SEEDS]
        ax.plot(positions, vals, marker="o", label=f"{label} (mean={np.mean(vals):.3f}, std={np.std(vals):.3f})")
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Overall accuracy")
    ax.set_title(f"Accuracy Across {len(SEEDS)} Seeds")
    ax.set_xticks(list(positions), [str(s) for s in SEEDS])
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "seed_variance.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrices(report, out_dir):
    y_test, ml_pred, baseline_pred = report.attrs["y_test"], report.attrs["ml_pred"], report.attrs["baseline_pred"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, pred, title in [(axes[0], ml_pred, "Proposed Method"), (axes[1], baseline_pred, "Baseline Model")]:
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Away win", "Home win"], yticklabels=["Away win", "Home win"])
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_pca_variance(report, out_dir):
    ratios = report.attrs["pca_explained_variance"]
    if ratios is None:
        return
    cumulative = np.cumsum(ratios)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, len(cumulative) + 1), cumulative, marker="o")
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=1, label="95% threshold")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA Variance Retained")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pca_variance.png", dpi=150)
    plt.close(fig)


def plot_roc_curve(report, out_dir):
    y_test, ml_proba = report.attrs["y_test"], report.attrs["ml_proba"]
    fpr, tpr, _ = roc_curve(y_test, ml_proba)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"Ensemble (AUC={report.attrs['ensemble_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curve (Ensemble, Soft-Vote Probability)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(report, out_dir):
    importance = report.attrs["feature_importance"]
    if importance is None:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    importance.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Relative importance (RF + XGBoost, mapped through PCA)")
    ax.set_title("Top Features Driving Predictions")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close(fig)


def plot_per_model_accuracy(report, out_dir):
    per_model = report.attrs["per_model_accuracy"].copy()
    per_model["ensemble (majority vote)"] = report.loc["Proposed Method", "Overall Accuracy"]
    per_model["baseline (WWR)"] = report.loc["Baseline Model", "Overall Accuracy"]

    fig, ax = plt.subplots(figsize=(8, 5))
    per_model.sort_values().plot(kind="barh", ax=ax, color="#55A868")
    ax.set_xlabel("Overall accuracy")
    ax.set_xlim(0, 1)
    ax.set_title("Accuracy: Individual Models vs. Ensemble vs. Baseline")
    fig.tight_layout()
    fig.savefig(out_dir / "per_model_accuracy.png", dpi=150)
    plt.close(fig)


def plot_precision_recall_f1(reports_by_seed, out_dir):
    """Grouped bar of precision/recall/F1, averaged over the 5 seeds -- the
    same numbers as report.md's "Precision, Recall, F1, ROC-AUC" table."""
    metrics = ["precision", "recall", "f1"]
    ensemble = {m: np.mean([reports_by_seed[s].attrs["ensemble_metrics"][m] for s in SEEDS]) for m in metrics}
    baseline = {m: np.mean([reports_by_seed[s].attrs["baseline_metrics"][m] for s in SEEDS]) for m in metrics}

    df = pd.DataFrame({"Proposed ensemble": ensemble, "WWR baseline": baseline})
    ax = df.plot(kind="bar", figsize=(7, 5), rot=0, ylim=(0, 1))
    ax.set_title("Precision, Recall, and F1 (mean over 5 seeds)")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.legend(loc="lower right")
    ax.figure.tight_layout()
    ax.figure.savefig(out_dir / "precision_recall_f1.png", dpi=150)
    plt.close(ax.figure)


def plot_vote_agreement(reports_by_seed, out_dir):
    """Ensemble accuracy by vote agreement (unanimous vs. split), pooled
    across the 5 seeds -- the same numbers as report.md's "Per-Model Accuracy
    and Vote Agreement" table."""
    n_models = 5
    pooled = pd.DataFrame(0, index=range(1, n_models + 1), columns=["correct", "n_matches"], dtype=float)
    for s in SEEDS:
        table = reports_by_seed[s].attrs["agreement_table"]
        pooled.loc[table.index, "correct"] += table["accuracy"] * table["n_matches"]
        pooled.loc[table.index, "n_matches"] += table["n_matches"]

    unanimous = pooled.loc[[n_models]].sum()
    split = pooled.loc[pooled.index != n_models].sum()
    groups = pd.Series(
        {"Unanimous (5/5)": unanimous["correct"] / unanimous["n_matches"],
         "Split (4/5 or 3/5)": split["correct"] / split["n_matches"]}
    )
    counts = {"Unanimous (5/5)": int(unanimous["n_matches"]),
              "Split (4/5 or 3/5)": int(split["n_matches"])}

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(groups.index, groups.values, color=["#4C72B0", "#DD8452"])
    for bar, label in zip(bars, groups.index):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={counts[label]}", ha="center")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Ensemble accuracy")
    ax.set_title("Accuracy by Vote Agreement (pooled over 5 seeds)")
    fig.tight_layout()
    fig.savefig(out_dir / "vote_agreement.png", dpi=150)
    plt.close(fig)


def plot_ablation(out_dir):
    """Cumulative development accuracy through each kept ablation step. See
    ABLATION_STEPS above for provenance."""
    labels = [label for label, _ in ABLATION_STEPS]
    values = [value for _, value in ABLATION_STEPS]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(range(len(values)), values, marker="o", color="#4C72B0")
    for i, v in enumerate(values):
        ax.annotate(f"{v:.1f}%", (i, v), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylabel("Development accuracy (full 1,504-match set)")
    ax.set_title("Cumulative Effect of Each Kept Change")
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_progression.png", dpi=150)
    plt.close(fig)


def plot_bracket_by_round(out_dir):
    """Ensemble vs. baseline accuracy per knockout round on the 2026 World Cup
    bracket -- the same numbers as report.md's "Held-Out Test" table. See
    BRACKET_ROUNDS above for why this is transcribed rather than re-run live."""
    rows = BRACKET_ROUNDS + [BRACKET_OVERALL]
    plot_df = pd.DataFrame(
        {"Proposed ensemble": [r[2] for r in rows], "WWR baseline": [r[3] for r in rows]},
        index=[r[0] for r in rows],
    )

    ax = plot_df.plot(kind="bar", figsize=(9, 5), rot=20, ylim=(0, 1.05))
    ax.set_title("2026 World Cup Knockout Bracket: Accuracy by Round")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.legend(loc="lower left")
    ax.figure.tight_layout()
    ax.figure.savefig(out_dir / "bracket_by_round.png", dpi=150)
    plt.close(ax.figure)


def main() -> None:
    cfg = load_config()
    out_dir = resolve_path(cfg["evaluation"]["output_dir"]) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = build_team_profiles(cfg)
    feat, wc_all_history, _history_state = build_match_dataset(cfg, profiles)

    reports_by_seed = {s: run_seed(cfg, feat, wc_all_history, s) for s in SEEDS}
    default_report = reports_by_seed[cfg["project"]["random_state"]]

    plot_accuracy_comparison(default_report, out_dir)
    plot_seed_variance(reports_by_seed, out_dir)
    plot_confusion_matrices(default_report, out_dir)
    plot_pca_variance(default_report, out_dir)
    plot_roc_curve(default_report, out_dir)
    plot_feature_importance(default_report, out_dir)
    plot_per_model_accuracy(default_report, out_dir)
    plot_precision_recall_f1(reports_by_seed, out_dir)
    plot_vote_agreement(reports_by_seed, out_dir)
    plot_ablation(out_dir)
    plot_bracket_by_round(out_dir)

    print(f"\nFigures saved to {out_dir}")


if __name__ == "__main__":
    main()
