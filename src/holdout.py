"""Score the withheld years once.

    python -m src.holdout

Everything else in this project -- `src.pipeline`, `src.make_figures`, the
served model -- trains and reports on development data only, because
`build_match_dataset()` drops `evaluation.holdout_years` before returning.
This module is the one place those years are read.

Why bother. Roughly fifteen configurations were compared against the same five
random splits, each kept or discarded on how it scored there. That is model
selection on a test set, and its effect compounds invisibly: over the same
stretch the random-split figure rose 78.9% -> 80.3% while an untuned temporal
split moved 79.7% -> 79.5%. The random-split number is not wrong, it is just no
longer an estimate of anything out-of-sample.

So treat the number this prints as a one-shot measurement. Running it is
cheap; running it repeatedly while changing the model is how the guarantee is
spent. If you tune against it, it becomes another development split and you are
back where you started -- pick new holdout years instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config, set_seed
from src.evaluation.evaluate import classification_report_metrics, mcnemar_test
from src.pipeline import build_match_dataset, build_team_profiles, run_seed, split_holdout


def run() -> None:
    cfg = load_config()
    years = cfg["evaluation"].get("holdout_years") or []
    if not years:
        raise SystemExit(
            "evaluation.holdout_years is empty -- nothing is being withheld, so "
            "there is no holdout to score. Set it in config/config.yaml."
        )

    seed = cfg["project"]["random_state"]
    set_seed(seed)

    profiles = build_team_profiles(cfg)
    # apply_holdout=False: this is the one caller allowed to see the full frame.
    feat_all, wc_all_history, _ = build_match_dataset(cfg, profiles, apply_holdout=False)
    dev, holdout = split_holdout(cfg, feat_all)

    if holdout.empty:
        raise SystemExit(f"No matches found in holdout years {years}.")

    print("=" * 72)
    print(f"  HOLDOUT EVALUATION -- years {years}, scored once")
    print("=" * 72)
    print(f"development: {len(dev)} matches ({dev.year.min()}-{dev.year.max()})")
    print(f"holdout    : {len(holdout)} matches, "
          f"{holdout.date.min().date()} .. {holdout.date.max().date()}")

    # Train on ALL development data -- no inner split, since model selection is
    # already finished by the time this runs.
    report = run_seed(cfg, feat_all, wc_all_history, seed,
                      idx_train=dev.index, idx_test=holdout.index)
    attrs = report.attrs
    y = np.asarray(attrs["y_test"])
    ml, base = np.asarray(attrs["ml_pred"]), np.asarray(attrs["baseline_pred"])

    majority = max(y.mean(), 1 - y.mean())
    print("\n" + "=" * 72)
    print("  RESULT")
    print("=" * 72)
    print(f"  ensemble          {(ml == y).mean():.4f}")
    print(f"  WWR baseline      {(base == y).mean():.4f}")
    print(f"  always-home       {majority:.4f}   (the floor worth beating)")
    print(f"  margin over baseline {((ml == y).mean() - (base == y).mean()) * 100:+.2f} pp")

    sig = mcnemar_test(y, base, ml)
    print(f"\n  McNemar b={sig['b']} c={sig['c']} p={sig['p_value']:.4f}"
          + ("  (significant at p<0.05)" if sig["p_value"] < 0.05 else "  (not significant)"))

    print("\n  per-model accuracy, before the vote:")
    print(pd.Series(attrs["per_model_accuracy"]).to_string())

    metrics = classification_report_metrics(y, ml)
    print(f"\n  precision {metrics['precision']:.3f}  recall {metrics['recall']:.3f}  "
          f"f1 {metrics['f1']:.3f}  ROC-AUC {attrs['ensemble_auc']:.3f}")

    # The headline claim is about World Cup matches; qualifiers are the easier
    # majority of the data, so report them apart rather than blended.
    by_tournament = pd.DataFrame({
        "tournament": holdout["tournament"].to_numpy(),
        "correct": (ml == y),
    }).groupby("tournament")["correct"].agg(["size", "mean"])
    print("\n  by tournament:")
    print(by_tournament.to_string())

    print("\n" + "-" * 72)
    print("  This number has not been optimised against. Do not tune on it --")
    print("  re-running after changing the model spends what makes it worth having.")
    print("-" * 72)


if __name__ == "__main__":
    run()
