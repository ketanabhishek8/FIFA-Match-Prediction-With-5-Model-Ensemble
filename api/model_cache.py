"""Train-once-cache-to-disk for the API: the pipeline (profile building +
5-seed train/eval) is expensive, so we run it once and cache the result --
subsequent server starts load from disk unless the training-relevant config
changed.
"""
from __future__ import annotations

import hashlib
import json

import joblib

from src.config import resolve_path

CACHE_DIR = resolve_path("outputs/models")


# Config sections that change what gets trained. Anything listed here
# invalidates the cache when it changes; anything outside it (project.name, say)
# does not. `score_model` is here because the bundle now carries a fitted
# scoreline model too -- without it, editing rho bounds or max_goals would
# silently keep serving the previously cached one.
CACHE_KEY_SECTIONS = ("data", "features", "baseline", "models", "evaluation", "score_model")


def _cache_key(cfg: dict, seeds: list[int]) -> str:
    relevant = {k: cfg[k] for k in CACHE_KEY_SECTIONS if k in cfg}
    relevant["seeds"] = seeds
    # random_state picks which seed's report the API serves as the default
    # (see `default_seed` below), so it has to be part of the key -- otherwise
    # changing it to another value already in `seeds` hits a stale bundle and
    # silently keeps serving the previous seed's results.
    relevant["random_state"] = cfg["project"]["random_state"]
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


def load_or_train(cfg: dict, seeds: list[int]) -> dict:
    from src.pipeline import build_match_dataset, build_team_profiles, run_seed
    from src.score_pipeline import run_seed_scores

    cache_file = CACHE_DIR / f"{_cache_key(cfg, seeds)}.joblib"
    if cache_file.exists():
        return joblib.load(cache_file)

    profiles = build_team_profiles(cfg)
    feat, wc_all_history, history_state = build_match_dataset(cfg, profiles)
    reports_by_seed = {s: run_seed(cfg, feat, wc_all_history, s) for s in seeds}

    # The scoreline model trains on a draws-inclusive frame, so it needs its own
    # dataset build and its own scaler/PCA -- different rows mean a different
    # fit, and reusing the classifier's would transform inference rows with
    # statistics from a different training population. Built separately rather
    # than by filtering draws out of one shared frame, so each task's split is
    # bit-for-bit what `python -m src.pipeline` / `src.score_pipeline` produce.
    score_reports_by_seed = {}
    if cfg.get("score_model", {}).get("enabled", True):
        score_feat, _, _ = build_match_dataset(cfg, profiles, drop_draws=False)
        score_reports_by_seed = {
            s: run_seed_scores(cfg, score_feat, s, verbose=False) for s in seeds
        }

    bundle = {
        "profiles": profiles,
        "reports_by_seed": reports_by_seed,
        "score_reports_by_seed": score_reports_by_seed,
        "default_seed": cfg["project"]["random_state"],
        # Team strength as of the last recorded match. /predict needs it to
        # build the same history features training saw; None when the model was
        # trained with features.history_features off.
        "history_state": history_state,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, cache_file)
    return bundle
